package conversation

import (
	"bytes"
	"encoding/json"
	"strings"
)

// OpenAIResponseParser implements RequestParser for OpenAI Responses API.
// It parses /v1/responses request/response bodies.
type OpenAIResponseParser struct{}

// APIType returns "openai_response".
func (p *OpenAIResponseParser) APIType() string {
	return "openai_response"
}

// MatchURL returns true for OpenAI Responses endpoints.
func (p *OpenAIResponseParser) MatchURL(url string) bool {
	path := url
	if idx := strings.Index(path, "?"); idx >= 0 {
		path = path[:idx]
	}
	return strings.HasSuffix(path, "/v1/responses")
}

type openAIResponseRequest struct {
	Model        string          `json:"model"`
	Instructions string          `json:"instructions"`
	Input        json.RawMessage `json:"input"`
	Stream       bool            `json:"stream"`
}

type openAIResponseInputItem struct {
	Type      string          `json:"type"`
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	Output    json.RawMessage `json:"output"`
}

type openAIResponseEnvelope struct {
	Output   []openAIResponseOutputItem `json:"output"`
	Response *openAIResponseResponse    `json:"response"`
}

type openAIResponseResponse struct {
	Output []openAIResponseOutputItem `json:"output"`
}

type openAIResponseOutputItem struct {
	ID        string                   `json:"id"`
	Type      string                   `json:"type"`
	Role      string                   `json:"role"`
	Status    string                   `json:"status"`
	Content   json.RawMessage          `json:"content"`
	CallID    string                   `json:"call_id"`
	Name      string                   `json:"name"`
	Arguments string                   `json:"arguments"`
	Output    json.RawMessage          `json:"output"`
	Summary   []openAIResponseTextPart `json:"summary"`
}

type openAIResponseTextPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL json.RawMessage `json:"image_url,omitempty"`
}

type openAIResponseStreamEvent struct {
	Type        string                   `json:"type"`
	OutputIndex int                      `json:"output_index"`
	Delta       string                   `json:"delta"`
	Text        string                   `json:"text"`
	Arguments   string                   `json:"arguments"`
	Item        openAIResponseOutputItem `json:"item"`
	Response    *openAIResponseResponse  `json:"response"`
}

type openAIResponseStreamItemState struct {
	OutputIndex      int
	Item             openAIResponseOutputItem
	Text             string
	Arguments        string
	TextBuilder      strings.Builder
	ArgumentsBuilder strings.Builder
}

// ParseRequest parses an OpenAI Responses request body.
func (p *OpenAIResponseParser) ParseRequest(body []byte) *ParsedRequest {
	if len(body) == 0 {
		return nil
	}

	var req openAIResponseRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil
	}

	messages := make([]UnifiedMessage, 0, 8)
	if strings.TrimSpace(req.Instructions) != "" {
		messages = append(messages, UnifiedMessage{
			Role:    RoleSystem,
			Content: req.Instructions,
			Source:  "request",
		})
	}

	messages = append(messages, parseOpenAIResponseInput(req.Input)...)
	if len(messages) == 0 {
		return nil
	}

	var firstUserContent string
	var firstContent string
	for i := range messages {
		messages[i].MsgIndex = i
		messages[i].Source = "request"

		content := strings.TrimSpace(messages[i].Content)
		if firstContent == "" && content != "" {
			firstContent = content
		}
		if firstUserContent == "" && messages[i].Role == RoleUser && content != "" {
			firstUserContent = content
		}
	}

	conversationKey := deriveConversationKey(firstUserContent)
	if conversationKey == "" {
		conversationKey = deriveConversationKey(firstContent)
	}

	return &ParsedRequest{
		ConversationKey: conversationKey,
		Model:           req.Model,
		Messages:        messages,
		Stream:          req.Stream,
		APIType:         p.APIType(),
	}
}

// ParseResponse parses an OpenAI Responses response body and extracts output messages.
func (p *OpenAIResponseParser) ParseResponse(body []byte) []UnifiedMessage {
	if len(body) == 0 {
		return nil
	}

	if messages := parseOpenAIResponseJSONBody(body); len(messages) > 0 {
		return messages
	}

	return parseOpenAIResponseSSEBody(body)
}

func parseOpenAIResponseInput(raw json.RawMessage) []UnifiedMessage {
	if len(raw) == 0 {
		return nil
	}

	var inputText string
	if err := json.Unmarshal(raw, &inputText); err == nil {
		if strings.TrimSpace(inputText) == "" {
			return nil
		}
		return []UnifiedMessage{{
			Role:    RoleUser,
			Content: inputText,
		}}
	}

	var items []openAIResponseInputItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil
	}

	messages := make([]UnifiedMessage, 0, len(items))
	for _, item := range items {
		itemType := normalizeOpenAIResponseItemType(item.Type, item.Role)
		switch itemType {
		case "message":
			content, contentParts := parseOpenAIResponseMessageContent(item.Content)
			role := strings.TrimSpace(item.Role)
			if role == "" {
				role = string(RoleUser)
			}
			messages = append(messages, UnifiedMessage{
				Role:         MessageRole(role),
				Content:      content,
				ContentParts: contentParts,
			})
		case "function_call":
			messages = append(messages, UnifiedMessage{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{{
					ID:   item.CallID,
					Type: "function",
					Function: FunctionCall{
						Name:      item.Name,
						Arguments: item.Arguments,
					},
				}},
			})
		case "function_call_output":
			messages = append(messages, UnifiedMessage{
				Role:       RoleTool,
				Content:    parseOpenAIResponseRawString(item.Output),
				ToolCallID: item.CallID,
			})
		}
	}

	return messages
}

func parseOpenAIResponseJSONBody(body []byte) []UnifiedMessage {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || !json.Valid(trimmed) {
		return nil
	}

	var envelope openAIResponseEnvelope
	if err := json.Unmarshal(trimmed, &envelope); err != nil {
		return nil
	}

	if len(envelope.Output) > 0 {
		return parseOpenAIResponseOutputItems(envelope.Output, "response")
	}
	if envelope.Response != nil && len(envelope.Response.Output) > 0 {
		return parseOpenAIResponseOutputItems(envelope.Response.Output, "response")
	}

	return nil
}

func parseOpenAIResponseSSEBody(body []byte) []UnifiedMessage {
	states := make(map[int]*openAIResponseStreamItemState)
	order := make([]int, 0, 8)

	ensureState := func(outputIndex int) *openAIResponseStreamItemState {
		if state, ok := states[outputIndex]; ok {
			return state
		}
		state := &openAIResponseStreamItemState{OutputIndex: outputIndex}
		states[outputIndex] = state
		order = append(order, outputIndex)
		return state
	}

	for _, line := range splitSSELines(body) {
		data := strings.TrimSpace(extractSSEData(line))
		if data == "" || data == "[DONE]" || !json.Valid([]byte(data)) {
			continue
		}

		var event openAIResponseStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		if event.Type == "response.completed" && event.Response != nil && len(event.Response.Output) > 0 {
			return parseOpenAIResponseOutputItems(event.Response.Output, "response")
		}

		switch event.Type {
		case "response.output_item.added", "response.output_item.done":
			state := ensureState(event.OutputIndex)
			applyOpenAIResponseOutputState(state, event.Item)
		case "response.output_text.delta":
			state := ensureState(event.OutputIndex)
			if strings.TrimSpace(state.Item.Type) == "" {
				state.Item.Type = "message"
			}
			if strings.TrimSpace(state.Item.Role) == "" {
				state.Item.Role = string(RoleAssistant)
			}
			state.TextBuilder.WriteString(event.Delta)
		case "response.output_text.done":
			state := ensureState(event.OutputIndex)
			if strings.TrimSpace(state.Item.Type) == "" {
				state.Item.Type = "message"
			}
			if strings.TrimSpace(state.Item.Role) == "" {
				state.Item.Role = string(RoleAssistant)
			}
			if event.Text != "" {
				state.Text = event.Text
			}
		case "response.function_call_arguments.delta":
			state := ensureState(event.OutputIndex)
			if strings.TrimSpace(state.Item.Type) == "" {
				state.Item.Type = "function_call"
			}
			state.ArgumentsBuilder.WriteString(event.Delta)
		case "response.function_call_arguments.done":
			state := ensureState(event.OutputIndex)
			if strings.TrimSpace(state.Item.Type) == "" {
				state.Item.Type = "function_call"
			}
			if event.Arguments != "" {
				state.Arguments = event.Arguments
			}
		}
	}

	if len(order) == 0 {
		return nil
	}

	sortOpenAIResponseIndexes(order)

	items := make([]openAIResponseOutputItem, 0, len(order))
	for _, outputIndex := range order {
		state := states[outputIndex]
		if state == nil {
			continue
		}

		item := state.Item
		switch strings.TrimSpace(item.Type) {
		case "message":
			if isEmptyOpenAIResponseRaw(item.Content) {
				text := state.Text
				if text == "" {
					text = state.TextBuilder.String()
				}
				if text != "" {
					item.Content = marshalOpenAIResponseTextContent(text)
				}
			}
		case "function_call":
			if item.Arguments == "" {
				args := state.Arguments
				if args == "" {
					args = state.ArgumentsBuilder.String()
				}
				item.Arguments = args
			}
		}

		items = append(items, item)
	}

	return parseOpenAIResponseOutputItems(items, "response")
}

func parseOpenAIResponseOutputItems(items []openAIResponseOutputItem, source string) []UnifiedMessage {
	if len(items) == 0 {
		return nil
	}

	messages := make([]UnifiedMessage, 0, len(items))
	for _, item := range items {
		switch strings.TrimSpace(item.Type) {
		case "message":
			content, contentParts := parseOpenAIResponseMessageContent(item.Content)
			role := strings.TrimSpace(item.Role)
			if role == "" {
				role = string(RoleAssistant)
			}
			messages = append(messages, UnifiedMessage{
				Role:         MessageRole(role),
				Content:      content,
				ContentParts: contentParts,
				MsgIndex:     len(messages),
				Source:       source,
			})
		case "function_call":
			messages = append(messages, UnifiedMessage{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{{
					ID:   item.CallID,
					Type: "function",
					Function: FunctionCall{
						Name:      item.Name,
						Arguments: item.Arguments,
					},
				}},
				MsgIndex: len(messages),
				Source:   source,
			})
		case "function_call_output":
			messages = append(messages, UnifiedMessage{
				Role:       RoleTool,
				Content:    parseOpenAIResponseRawString(item.Output),
				ToolCallID: item.CallID,
				MsgIndex:   len(messages),
				Source:     source,
			})
		}
	}

	return messages
}

func parseOpenAIResponseMessageContent(raw json.RawMessage) (string, []ContentPart) {
	if len(raw) == 0 {
		return "", nil
	}

	var content string
	if err := json.Unmarshal(raw, &content); err == nil {
		return content, nil
	}

	var parts []openAIResponseTextPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		return convertOpenAIResponseContentParts(parts)
	}

	var part openAIResponseTextPart
	if err := json.Unmarshal(raw, &part); err == nil && hasOpenAIResponseContent(part) {
		return convertOpenAIResponseContentParts([]openAIResponseTextPart{part})
	}

	return string(bytes.TrimSpace(raw)), nil
}

func convertOpenAIResponseContentParts(parts []openAIResponseTextPart) (string, []ContentPart) {
	if len(parts) == 0 {
		return "", nil
	}

	contentParts := make([]ContentPart, 0, len(parts))
	textParts := make([]string, 0, len(parts))

	for _, part := range parts {
		partType := strings.TrimSpace(part.Type)
		if partType == "" {
			if part.Text != "" {
				partType = "input_text"
			} else if len(part.ImageURL) > 0 {
				partType = "input_image"
			}
		}

		switch partType {
		case "input_text", "output_text", "text", "summary_text":
			if part.Text == "" {
				continue
			}
			contentParts = append(contentParts, ContentPart{
				Type: "text",
				Text: part.Text,
			})
			textParts = append(textParts, part.Text)
		case "input_image", "image_url":
			imageURL := parseOpenAIResponseImageURL(part.ImageURL)
			if imageURL == nil {
				continue
			}
			contentParts = append(contentParts, ContentPart{
				Type:     "image_url",
				ImageURL: imageURL,
			})
		}
	}

	if len(contentParts) == 0 {
		return "", nil
	}

	return strings.Join(textParts, "\n"), contentParts
}

func parseOpenAIResponseImageURL(raw json.RawMessage) *ImageURL {
	if len(raw) == 0 {
		return nil
	}

	var url string
	if err := json.Unmarshal(raw, &url); err == nil {
		if strings.TrimSpace(url) == "" {
			return nil
		}
		return &ImageURL{URL: url}
	}

	var imageURL ImageURL
	if err := json.Unmarshal(raw, &imageURL); err == nil {
		if strings.TrimSpace(imageURL.URL) == "" {
			return nil
		}
		return &imageURL
	}

	return nil
}

func parseOpenAIResponseRawString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	var value string
	if err := json.Unmarshal(raw, &value); err == nil {
		return value
	}

	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}

	return string(trimmed)
}

func isEmptyOpenAIResponseRaw(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) || bytes.Equal(trimmed, []byte("[]"))
}

func applyOpenAIResponseOutputState(state *openAIResponseStreamItemState, item openAIResponseOutputItem) {
	if state == nil {
		return
	}

	if strings.TrimSpace(item.ID) != "" {
		state.Item.ID = item.ID
	}
	if strings.TrimSpace(item.Type) != "" {
		state.Item.Type = item.Type
	}
	if strings.TrimSpace(item.Role) != "" {
		state.Item.Role = item.Role
	}
	if strings.TrimSpace(item.Status) != "" {
		state.Item.Status = item.Status
	}
	if len(item.Content) > 0 {
		state.Item.Content = append(json.RawMessage(nil), item.Content...)
	}
	if strings.TrimSpace(item.CallID) != "" {
		state.Item.CallID = item.CallID
	}
	if strings.TrimSpace(item.Name) != "" {
		state.Item.Name = item.Name
	}
	if item.Arguments != "" {
		state.Item.Arguments = item.Arguments
	}
	if len(item.Output) > 0 {
		state.Item.Output = append(json.RawMessage(nil), item.Output...)
	}
}

func marshalOpenAIResponseTextContent(text string) json.RawMessage {
	data, err := json.Marshal([]openAIResponseTextPart{{
		Type: "output_text",
		Text: text,
	}})
	if err != nil {
		return nil
	}
	return data
}

func normalizeOpenAIResponseItemType(itemType, role string) string {
	trimmedType := strings.TrimSpace(itemType)
	if trimmedType != "" {
		return trimmedType
	}
	if strings.TrimSpace(role) != "" {
		return "message"
	}
	return ""
}

func hasOpenAIResponseContent(part openAIResponseTextPart) bool {
	return strings.TrimSpace(part.Type) != "" || part.Text != "" || len(part.ImageURL) > 0
}

func sortOpenAIResponseIndexes(indexes []int) {
	for i := 0; i < len(indexes); i++ {
		for j := i + 1; j < len(indexes); j++ {
			if indexes[j] < indexes[i] {
				indexes[i], indexes[j] = indexes[j], indexes[i]
			}
		}
	}
}
