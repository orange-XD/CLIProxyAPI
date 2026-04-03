package conversation

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
)

// OpenAIChatParser implements RequestParser for OpenAI Chat Completions API.
// It parses /v1/chat/completions request/response bodies.
type OpenAIChatParser struct{}

// APIType returns "openai_chat".
func (p *OpenAIChatParser) APIType() string {
	return "openai_chat"
}

// MatchURL returns true for OpenAI Chat Completions endpoints.
func (p *OpenAIChatParser) MatchURL(url string) bool {
	return strings.HasSuffix(url, "/v1/chat/completions") || strings.Contains(url, "/chat/completions")
}

// openAIChatRequest represents the JSON structure of an OpenAI Chat Completions request.
type openAIChatRequest struct {
	Model    string              `json:"model"`
	Messages []openAIChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
}

// openAIChatMessage represents a single message in the OpenAI Chat format.
type openAIChatMessage struct {
	Role       string           `json:"role"`
	Content    json.RawMessage  `json:"content"` // Can be string or array
	Name       string           `json:"name,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

// openAIToolCall represents a tool call in the OpenAI Chat format.
type openAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openAIFunctionCall `json:"function"`
}

// openAIFunctionCall represents a function call within a tool call.
type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// openAIChatResponse represents the JSON structure of an OpenAI Chat Completions response.
type openAIChatResponse struct {
	Choices []openAIChatChoice `json:"choices"`
	Model   string             `json:"model"`
}

// openAIChatChoice represents a single choice in the response.
type openAIChatChoice struct {
	Message openAIChatMessage `json:"message"`
}

// ParseRequest parses an OpenAI Chat Completions request body.
func (p *OpenAIChatParser) ParseRequest(body []byte) *ParsedRequest {
	if len(body) == 0 {
		return nil
	}

	var req openAIChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil
	}

	if len(req.Messages) == 0 {
		return nil
	}

	messages := make([]UnifiedMessage, 0, len(req.Messages))
	var firstUserContent string

	for i, msg := range req.Messages {
		um := UnifiedMessage{
			Role:       MessageRole(msg.Role),
			MsgIndex:   i,
			Source:     "request",
			ToolCallID: msg.ToolCallID,
			Name:       msg.Name,
		}

		// Parse content: can be string or array of content parts
		contentStr, contentParts := parseContent(msg.Content)
		um.Content = contentStr
		um.ContentParts = contentParts

		// Parse tool calls
		if len(msg.ToolCalls) > 0 {
			um.ToolCalls = make([]ToolCall, 0, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				um.ToolCalls = append(um.ToolCalls, ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
		}

		messages = append(messages, um)

		// Capture first user message for conversation key
		if firstUserContent == "" && msg.Role == "user" {
			firstUserContent = contentStr
		}
	}

	// Derive conversation key from first user message
	conversationKey := deriveConversationKey(firstUserContent)

	return &ParsedRequest{
		ConversationKey: conversationKey,
		Model:           req.Model,
		Messages:        messages,
		Stream:          req.Stream,
		APIType:         p.APIType(),
	}
}

// ParseResponse parses an OpenAI Chat Completions response body and extracts assistant messages.
func (p *OpenAIChatParser) ParseResponse(body []byte) []UnifiedMessage {
	if len(body) == 0 {
		return nil
	}

	var resp openAIChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil
	}

	if len(resp.Choices) == 0 {
		return nil
	}

	result := make([]UnifiedMessage, 0, len(resp.Choices))
	for i, choice := range resp.Choices {
		um := UnifiedMessage{
			Role:       MessageRole(choice.Message.Role),
			MsgIndex:   i,
			Source:     "response",
			ToolCallID: choice.Message.ToolCallID,
			Name:       choice.Message.Name,
		}

		contentStr, contentParts := parseContent(choice.Message.Content)
		um.Content = contentStr
		um.ContentParts = contentParts

		if len(choice.Message.ToolCalls) > 0 {
			um.ToolCalls = make([]ToolCall, 0, len(choice.Message.ToolCalls))
			for _, tc := range choice.Message.ToolCalls {
				um.ToolCalls = append(um.ToolCalls, ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
		}

		result = append(result, um)
	}

	return result
}

// parseContent handles the OpenAI content field which can be either a string or
// an array of content parts.
func parseContent(raw json.RawMessage) (string, []ContentPart) {
	if len(raw) == 0 {
		return "", nil
	}

	// Try string first
	var str string
	if err := json.Unmarshal(raw, &str); err == nil {
		return str, nil
	}

	// Try array of content parts
	var parts []ContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		// Extract text from parts for the simple content field
		var textParts []string
		for _, p := range parts {
			if p.Type == "text" && p.Text != "" {
				textParts = append(textParts, p.Text)
			}
		}
		content := strings.Join(textParts, "\n")
		return content, parts
	}

	// Fallback: return raw as string
	return string(raw), nil
}

// deriveConversationKey generates a unique key from the first user message content
// using SHA256 hash (first 16 bytes hex-encoded).
func deriveConversationKey(content string) string {
	if content == "" {
		return ""
	}
	h := sha256.Sum256([]byte(content))
	return fmt.Sprintf("%x", h[:16])
}
