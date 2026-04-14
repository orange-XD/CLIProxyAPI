package conversation

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"
)

// ClaudeMessagesParser implements RequestParser for Anthropic Messages API.
// It parses /v1/messages request/response bodies.
type ClaudeMessagesParser struct{}

// APIType returns "claude_messages".
func (p *ClaudeMessagesParser) APIType() string {
	return "claude_messages"
}

// MatchURL returns true for Anthropic Messages endpoints.
func (p *ClaudeMessagesParser) MatchURL(url string) bool {
	path := url
	if idx := strings.Index(path, "?"); idx >= 0 {
		path = path[:idx]
	}
	path = strings.TrimRight(path, "/")
	return strings.HasSuffix(path, "/v1/messages")
}

type claudeMessagesRequest struct {
	Model    string                  `json:"model"`
	System   json.RawMessage         `json:"system"`
	Messages []claudeMessagesMessage `json:"messages"`
	Stream   bool                    `json:"stream"`
}

type claudeMessagesMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type claudeMessagesResponse struct {
	Type    string          `json:"type"`
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type claudeContentBlock struct {
	Type      string             `json:"type"`
	Text      string             `json:"text"`
	ID        string             `json:"id"`
	Name      string             `json:"name"`
	Input     json.RawMessage    `json:"input"`
	Source    *claudeImageSource `json:"source"`
	URL       string             `json:"url"`
	Content   json.RawMessage    `json:"content"`
	ToolUseID string             `json:"tool_use_id"`
}

type claudeImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
	URL       string `json:"url"`
}

type claudeStreamEvent struct {
	Type         string          `json:"type"`
	Index        int             `json:"index"`
	ContentBlock json.RawMessage `json:"content_block"`
	Delta        json.RawMessage `json:"delta"`
}

type claudeStreamDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	PartialJSON string `json:"partial_json"`
}

type claudeStreamBlockState struct {
	Type                  string
	Text                  strings.Builder
	Image                 *ContentPart
	ToolCall              ToolCall
	ToolArguments         strings.Builder
	ToolArgumentsFallback string
}

// ParseRequest parses an Anthropic Messages request body.
func (p *ClaudeMessagesParser) ParseRequest(body []byte) *ParsedRequest {
	if len(body) == 0 {
		return nil
	}

	var req claudeMessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil
	}

	if len(req.Messages) == 0 {
		return nil
	}

	messages := make([]UnifiedMessage, 0, len(req.Messages)+1)
	messages = append(messages, parseClaudeRoleContent(RoleSystem, req.System, "request")...)
	for _, message := range req.Messages {
		role := MessageRole(strings.TrimSpace(message.Role))
		if role == "" {
			continue
		}
		messages = append(messages, parseClaudeRoleContent(role, message.Content, "request")...)
	}
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

// ParseResponse parses an Anthropic Messages response body and extracts assistant messages.
func (p *ClaudeMessagesParser) ParseResponse(body []byte) []UnifiedMessage {
	if len(body) == 0 {
		return nil
	}

	if messages := parseClaudeMessagesJSONResponse(body); len(messages) > 0 {
		return messages
	}

	return parseClaudeMessagesSSEResponse(body)
}

func parseClaudeMessagesJSONResponse(body []byte) []UnifiedMessage {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || !json.Valid(trimmed) {
		return nil
	}

	var resp claudeMessagesResponse
	if err := json.Unmarshal(trimmed, &resp); err != nil {
		return nil
	}

	if resp.Type != "" && resp.Type != "message" {
		return nil
	}

	role := MessageRole(strings.TrimSpace(resp.Role))
	if role == "" {
		role = RoleAssistant
	}

	messages := parseClaudeRoleContent(role, resp.Content, "response")
	for i := range messages {
		messages[i].MsgIndex = i
		messages[i].Source = "response"
	}
	return messages
}

func parseClaudeMessagesSSEResponse(body []byte) []UnifiedMessage {
	states := make(map[int]*claudeStreamBlockState)
	order := make([]int, 0, 8)

	ensureState := func(index int) *claudeStreamBlockState {
		if state, ok := states[index]; ok {
			return state
		}
		state := &claudeStreamBlockState{}
		states[index] = state
		order = append(order, index)
		return state
	}

	for _, line := range splitSSELines(body) {
		data := strings.TrimSpace(extractSSEData(line))
		if data == "" || data == "[DONE]" || !json.Valid([]byte(data)) {
			continue
		}

		var event claudeStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "content_block_start":
			var block claudeContentBlock
			if err := json.Unmarshal(event.ContentBlock, &block); err != nil {
				continue
			}
			state := ensureState(event.Index)
			state.Type = strings.TrimSpace(block.Type)

			switch state.Type {
			case "text":
				state.Text.WriteString(block.Text)
			case "image":
				state.Image = parseClaudeImageContentPart(block)
			case "tool_use":
				state.ToolCall = ToolCall{
					ID:   block.ID,
					Type: "function",
					Function: FunctionCall{
						Name: block.Name,
					},
				}
				state.ToolArgumentsFallback = normalizeClaudeArguments(block.Input)
			}
		case "content_block_delta":
			var delta claudeStreamDelta
			if err := json.Unmarshal(event.Delta, &delta); err != nil {
				continue
			}
			state := ensureState(event.Index)

			switch delta.Type {
			case "text_delta":
				if state.Type == "" {
					state.Type = "text"
				}
				state.Text.WriteString(delta.Text)
			case "input_json_delta":
				if state.Type == "" {
					state.Type = "tool_use"
					state.ToolCall.Type = "function"
				}
				state.ToolArguments.WriteString(delta.PartialJSON)
			}
		}
	}

	if len(order) == 0 {
		return nil
	}

	sort.Ints(order)

	contentParts := make([]ContentPart, 0, len(order))
	textParts := make([]string, 0, len(order))
	toolCalls := make([]ToolCall, 0, len(order))

	for _, index := range order {
		state := states[index]
		if state == nil {
			continue
		}

		switch state.Type {
		case "text":
			text := state.Text.String()
			if text == "" {
				continue
			}
			contentParts = append(contentParts, ContentPart{
				Type: "text",
				Text: text,
			})
			textParts = append(textParts, text)
		case "image":
			if state.Image != nil {
				contentParts = append(contentParts, *state.Image)
			}
		case "tool_use":
			if strings.TrimSpace(state.ToolCall.ID) == "" && strings.TrimSpace(state.ToolCall.Function.Name) == "" {
				continue
			}
			toolCall := state.ToolCall
			args := state.ToolArguments.String()
			if args == "" {
				args = state.ToolArgumentsFallback
			}
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			if toolCall.Type == "" {
				toolCall.Type = "function"
			}
			toolCall.Function.Arguments = args
			toolCalls = append(toolCalls, toolCall)
		}
	}

	if len(contentParts) == 0 && len(toolCalls) == 0 {
		return nil
	}

	msg := UnifiedMessage{
		Role:     RoleAssistant,
		Content:  strings.Join(textParts, "\n"),
		MsgIndex: 0,
		Source:   "response",
	}
	if len(contentParts) > 0 {
		msg.ContentParts = contentParts
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	return []UnifiedMessage{msg}
}

func parseClaudeRoleContent(role MessageRole, raw json.RawMessage, source string) []UnifiedMessage {
	trimmed := bytes.TrimSpace(raw)
	if role == "" || len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}

	var content string
	if err := json.Unmarshal(trimmed, &content); err == nil {
		if strings.TrimSpace(content) == "" {
			return nil
		}
		return []UnifiedMessage{{
			Role:    role,
			Content: content,
			Source:  source,
		}}
	}

	var blocks []claudeContentBlock
	if err := json.Unmarshal(trimmed, &blocks); err != nil {
		return nil
	}

	contentParts := make([]ContentPart, 0, len(blocks))
	textParts := make([]string, 0, len(blocks))
	toolCalls := make([]ToolCall, 0, len(blocks))
	toolResults := make([]UnifiedMessage, 0, len(blocks))

	for _, block := range blocks {
		switch strings.TrimSpace(block.Type) {
		case "text":
			if block.Text == "" {
				continue
			}
			contentParts = append(contentParts, ContentPart{
				Type: "text",
				Text: block.Text,
			})
			textParts = append(textParts, block.Text)
		case "image":
			if imagePart := parseClaudeImageContentPart(block); imagePart != nil {
				contentParts = append(contentParts, *imagePart)
			}
		case "tool_use":
			if role != RoleAssistant {
				continue
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: FunctionCall{
					Name:      block.Name,
					Arguments: normalizeClaudeArguments(block.Input),
				},
			})
		case "tool_result":
			if role == RoleAssistant {
				continue
			}
			toolResult, ok := parseClaudeToolResultBlock(block, source)
			if ok {
				toolResults = append(toolResults, toolResult)
			}
		}
	}

	result := make([]UnifiedMessage, 0, len(toolResults)+1)
	if role != RoleAssistant && len(toolResults) > 0 {
		result = append(result, toolResults...)
	}

	if len(contentParts) == 0 && len(toolCalls) == 0 {
		return result
	}

	msg := UnifiedMessage{
		Role:    role,
		Content: strings.Join(textParts, "\n"),
		Source:  source,
	}
	if len(contentParts) > 0 {
		msg.ContentParts = contentParts
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	result = append(result, msg)
	return result
}

func parseClaudeToolResultBlock(block claudeContentBlock, source string) (UnifiedMessage, bool) {
	content, contentParts := parseClaudeToolResultContent(block.Content)
	msg := UnifiedMessage{
		Role:       RoleTool,
		Content:    content,
		ToolCallID: block.ToolUseID,
		Source:     source,
	}
	if len(contentParts) > 0 {
		msg.ContentParts = contentParts
	}
	if strings.TrimSpace(msg.ToolCallID) == "" && strings.TrimSpace(msg.Content) == "" && len(msg.ContentParts) == 0 {
		return UnifiedMessage{}, false
	}
	return msg, true
}

func parseClaudeToolResultContent(raw json.RawMessage) (string, []ContentPart) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", nil
	}

	var content string
	if err := json.Unmarshal(trimmed, &content); err == nil {
		return content, nil
	}

	var blocks []claudeContentBlock
	if err := json.Unmarshal(trimmed, &blocks); err == nil {
		contentParts := make([]ContentPart, 0, len(blocks))
		textParts := make([]string, 0, len(blocks))
		for _, block := range blocks {
			switch strings.TrimSpace(block.Type) {
			case "text":
				if block.Text == "" {
					continue
				}
				contentParts = append(contentParts, ContentPart{
					Type: "text",
					Text: block.Text,
				})
				textParts = append(textParts, block.Text)
			case "image":
				if imagePart := parseClaudeImageContentPart(block); imagePart != nil {
					contentParts = append(contentParts, *imagePart)
				}
			}
		}
		if len(contentParts) > 0 {
			return strings.Join(textParts, "\n"), contentParts
		}
		return string(trimmed), nil
	}

	var block claudeContentBlock
	if err := json.Unmarshal(trimmed, &block); err == nil {
		switch strings.TrimSpace(block.Type) {
		case "text":
			if block.Text != "" {
				return block.Text, []ContentPart{{
					Type: "text",
					Text: block.Text,
				}}
			}
		case "image":
			if imagePart := parseClaudeImageContentPart(block); imagePart != nil {
				return "", []ContentPart{*imagePart}
			}
		}
	}

	return string(trimmed), nil
}

func parseClaudeImageContentPart(block claudeContentBlock) *ContentPart {
	imageURL := ""
	if block.Source != nil {
		switch strings.TrimSpace(block.Source.Type) {
		case "base64":
			data := strings.TrimSpace(block.Source.Data)
			if data != "" {
				mediaType := strings.TrimSpace(block.Source.MediaType)
				if mediaType == "" {
					mediaType = "application/octet-stream"
				}
				imageURL = "data:" + mediaType + ";base64," + data
			}
		case "url":
			imageURL = strings.TrimSpace(block.Source.URL)
		}
	}
	if imageURL == "" {
		imageURL = strings.TrimSpace(block.URL)
	}
	if imageURL == "" {
		return nil
	}
	return &ContentPart{
		Type: "image_url",
		ImageURL: &ImageURL{
			URL: imageURL,
		},
	}
}

func normalizeClaudeArguments(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "{}"
	}
	return string(trimmed)
}
