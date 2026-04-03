// Package conversation provides PostgreSQL-based conversation logging with message deduplication.
// It implements the RequestLogger interface to capture OpenAI Chat Completions (and future API types)
// request/response data, parse messages, and store them in PostgreSQL for session restoration.
package conversation

import "time"

// MessageRole represents the role of a message sender in a conversation.
type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleTool      MessageRole = "tool"
	RoleFunction  MessageRole = "function"
	RoleDeveloper MessageRole = "developer"
)

// ContentPart represents a single content part within a message (for multi-part messages).
type ContentPart struct {
	Type     string    `json:"type"`                // "text", "image_url", etc.
	Text     string    `json:"text,omitempty"`      // For type="text"
	ImageURL *ImageURL `json:"image_url,omitempty"` // For type="image_url"
}

// ImageURL represents an image URL within a content part.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// ToolCall represents a tool/function call within an assistant message.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"` // "function"
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call with name and arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// UnifiedMessage represents a single message in a conversation, normalized across API types.
// This is the common model that all parsers produce, enabling storage in a unified schema.
type UnifiedMessage struct {
	// Role is the sender role (system, user, assistant, tool, function, developer).
	Role MessageRole `json:"role"`

	// Content is the text content of the message (for simple text messages).
	Content string `json:"content,omitempty"`

	// ContentParts holds multi-part content (text + images, etc.).
	ContentParts []ContentPart `json:"content_parts,omitempty"`

	// ToolCalls holds tool/function calls (for assistant messages).
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// ToolCallID is the ID of the tool call this message responds to (for tool messages).
	ToolCallID string `json:"tool_call_id,omitempty"`

	// Name is the function name (for function messages).
	Name string `json:"name,omitempty"`

	// MsgIndex is the positional index of this message in the messages array.
	MsgIndex int `json:"msg_index"`

	// Source indicates where this message came from: "request" or "response".
	Source string `json:"source"`
}

// ParsedRequest holds the result of parsing an API request body.
type ParsedRequest struct {
	// ConversationKey is a unique identifier derived from the first user message content.
	// It's used to group related requests into the same conversation.
	ConversationKey string `json:"conversation_key"`

	// Model is the model name from the request.
	Model string `json:"model"`

	// Messages are the parsed messages from the request body.
	Messages []UnifiedMessage `json:"messages"`

	// Stream indicates whether this is a streaming request.
	Stream bool `json:"stream"`

	// APIType identifies the API type that was parsed (e.g., "openai_chat", "claude_messages").
	APIType string `json:"api_type"`
}

// ConversationRecord represents a conversation record in the database.
type ConversationRecord struct {
	ID              string    `json:"id"`
	ConversationKey string    `json:"conversation_key"`
	Model           string    `json:"model"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
	MessageCount    int       `json:"message_count"`
}

// MessageRecord represents a deduplicated message record in the database.
type MessageRecord struct {
	ID             string    `json:"id"`
	ConversationID string    `json:"conversation_id"`
	MsgIndex       int       `json:"msg_index"`
	Source         string    `json:"source"` // "request" or "response"
	Role           string    `json:"role"`
	Content        string    `json:"content"`
	ContentJSON    string    `json:"content_json,omitempty"` // JSON for complex content parts
	ToolCallsJSON  string    `json:"tool_calls_json,omitempty"`
	ToolCallID     string    `json:"tool_call_id,omitempty"`
	Name           string    `json:"name,omitempty"`
	CreatedAt      time.Time `json:"created_at"`
}

// RequestLogRecord represents a full request/response snapshot in the database.
type RequestLogRecord struct {
	ID             string    `json:"id"`
	ConversationID string    `json:"conversation_id"`
	RequestMethod  string    `json:"request_method"`
	RequestURL     string    `json:"request_url"`
	RequestBody    []byte    `json:"request_body,omitempty"`
	ResponseStatus int       `json:"response_status"`
	ResponseBody   []byte    `json:"response_body,omitempty"`
	Model          string    `json:"model"`
	Stream         bool      `json:"stream"`
	RequestID      string    `json:"request_id"`
	CreatedAt      time.Time `json:"created_at"`
	DurationMs     int64     `json:"duration_ms"`
}
