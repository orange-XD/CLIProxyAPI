package conversation

import "testing"

func TestClaudeMessagesParserMatchURL(t *testing.T) {
	parser := &ClaudeMessagesParser{}

	tests := []struct {
		url  string
		want bool
	}{
		{url: "/v1/messages", want: true},
		{url: "/v1/messages?beta=true", want: true},
		{url: "/v1/messages/count_tokens", want: false},
		{url: "/v1/chat/completions", want: false},
	}

	for _, tt := range tests {
		if got := parser.MatchURL(tt.url); got != tt.want {
			t.Fatalf("MatchURL(%q) = %v, want %v", tt.url, got, tt.want)
		}
	}
}

func TestClaudeMessagesParserParseRequest(t *testing.T) {
	parser := &ClaudeMessagesParser{}

	body := []byte(`{
		"model":"claude-sonnet-4-5",
		"system":"You are helpful.",
		"stream":true,
		"messages":[
			{
				"role":"assistant",
				"content":[
					{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"query":"docs"}}
				]
			},
			{
				"role":"user",
				"content":[
					{"type":"text","text":"before"},
					{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"tool ok"}]},
					{"type":"text","text":"after"},
					{
						"type":"image",
						"source":{
							"type":"url",
							"url":"https://example.com/image.png"
						}
					}
				]
			}
		]
	}`)

	parsed := parser.ParseRequest(body)
	if parsed == nil {
		t.Fatal("ParseRequest returned nil")
	}

	if parsed.APIType != "claude_messages" {
		t.Fatalf("APIType = %q, want %q", parsed.APIType, "claude_messages")
	}
	if parsed.Model != "claude-sonnet-4-5" {
		t.Fatalf("Model = %q, want %q", parsed.Model, "claude-sonnet-4-5")
	}
	if !parsed.Stream {
		t.Fatal("Stream = false, want true")
	}
	if parsed.ConversationKey != deriveConversationKey("before\nafter") {
		t.Fatalf("ConversationKey = %q, want hash of first user content", parsed.ConversationKey)
	}
	if len(parsed.Messages) != 4 {
		t.Fatalf("len(Messages) = %d, want 4", len(parsed.Messages))
	}

	if parsed.Messages[0].Role != RoleSystem || parsed.Messages[0].Content != "You are helpful." {
		t.Fatalf("system message = %+v", parsed.Messages[0])
	}

	if parsed.Messages[1].Role != RoleAssistant {
		t.Fatalf("assistant role = %q, want %q", parsed.Messages[1].Role, RoleAssistant)
	}
	if len(parsed.Messages[1].ToolCalls) != 1 {
		t.Fatalf("len(tool calls) = %d, want 1", len(parsed.Messages[1].ToolCalls))
	}
	if parsed.Messages[1].ToolCalls[0].ID != "toolu_1" {
		t.Fatalf("tool call id = %q, want %q", parsed.Messages[1].ToolCalls[0].ID, "toolu_1")
	}
	if parsed.Messages[1].ToolCalls[0].Function.Arguments != `{"query":"docs"}` {
		t.Fatalf("tool call arguments = %q", parsed.Messages[1].ToolCalls[0].Function.Arguments)
	}

	if parsed.Messages[2].Role != RoleTool {
		t.Fatalf("tool result role = %q, want %q", parsed.Messages[2].Role, RoleTool)
	}
	if parsed.Messages[2].ToolCallID != "toolu_1" {
		t.Fatalf("tool result ToolCallID = %q, want %q", parsed.Messages[2].ToolCallID, "toolu_1")
	}
	if parsed.Messages[2].Content != "tool ok" {
		t.Fatalf("tool result content = %q, want %q", parsed.Messages[2].Content, "tool ok")
	}

	userMessage := parsed.Messages[3]
	if userMessage.Role != RoleUser {
		t.Fatalf("user role = %q, want %q", userMessage.Role, RoleUser)
	}
	if userMessage.Content != "before\nafter" {
		t.Fatalf("user content = %q, want %q", userMessage.Content, "before\nafter")
	}
	if len(userMessage.ContentParts) != 3 {
		t.Fatalf("len(user content parts) = %d, want 3", len(userMessage.ContentParts))
	}
	if userMessage.ContentParts[2].ImageURL == nil || userMessage.ContentParts[2].ImageURL.URL != "https://example.com/image.png" {
		t.Fatalf("user image part = %+v", userMessage.ContentParts[2])
	}
}

func TestClaudeMessagesParserParseResponse(t *testing.T) {
	parser := &ClaudeMessagesParser{}

	body := []byte(`{
		"id":"msg_123",
		"type":"message",
		"role":"assistant",
		"content":[
			{"type":"thinking","text":"internal"},
			{"type":"text","text":"hello"},
			{"type":"tool_use","id":"toolu_2","name":"lookup","input":{"query":"parser"}}
		]
	}`)

	messages := parser.ParseResponse(body)
	if len(messages) != 1 {
		t.Fatalf("len(messages) = %d, want 1", len(messages))
	}

	if messages[0].Role != RoleAssistant {
		t.Fatalf("messages[0].Role = %q, want %q", messages[0].Role, RoleAssistant)
	}
	if messages[0].Content != "hello" {
		t.Fatalf("messages[0].Content = %q, want %q", messages[0].Content, "hello")
	}
	if len(messages[0].ContentParts) != 1 {
		t.Fatalf("len(messages[0].ContentParts) = %d, want 1", len(messages[0].ContentParts))
	}
	if len(messages[0].ToolCalls) != 1 {
		t.Fatalf("len(messages[0].ToolCalls) = %d, want 1", len(messages[0].ToolCalls))
	}
	if messages[0].ToolCalls[0].ID != "toolu_2" {
		t.Fatalf("messages[0].ToolCalls[0].ID = %q, want %q", messages[0].ToolCalls[0].ID, "toolu_2")
	}
	if messages[0].ToolCalls[0].Function.Arguments != `{"query":"parser"}` {
		t.Fatalf("messages[0].ToolCalls[0].Function.Arguments = %q", messages[0].ToolCalls[0].Function.Arguments)
	}
}

func TestClaudeMessagesParserParseResponseSSE(t *testing.T) {
	parser := &ClaudeMessagesParser{}

	body := []byte(
		"event: message_start\n" +
			"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n" +
			"event: content_block_start\n" +
			"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Hello\"}}\n\n" +
			"event: content_block_delta\n" +
			"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n" +
			"event: content_block_stop\n" +
			"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: content_block_start\n" +
			"data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_3\",\"name\":\"lookup\"}}\n\n" +
			"event: content_block_delta\n" +
			"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"query\\\":\\\"docs\\\"}\"}}\n\n" +
			"event: content_block_stop\n" +
			"data: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
			"event: message_stop\n" +
			"data: {\"type\":\"message_stop\"}\n\n",
	)

	messages := parser.ParseResponse(body)
	if len(messages) != 1 {
		t.Fatalf("len(messages) = %d, want 1", len(messages))
	}

	if messages[0].Content != "Hello world" {
		t.Fatalf("messages[0].Content = %q, want %q", messages[0].Content, "Hello world")
	}
	if messages[0].Role != RoleAssistant {
		t.Fatalf("messages[0].Role = %q, want %q", messages[0].Role, RoleAssistant)
	}
	if len(messages[0].ToolCalls) != 1 {
		t.Fatalf("len(messages[0].ToolCalls) = %d, want 1", len(messages[0].ToolCalls))
	}
	if messages[0].ToolCalls[0].ID != "toolu_3" {
		t.Fatalf("messages[0].ToolCalls[0].ID = %q, want %q", messages[0].ToolCalls[0].ID, "toolu_3")
	}
	if messages[0].ToolCalls[0].Function.Name != "lookup" {
		t.Fatalf("messages[0].ToolCalls[0].Function.Name = %q, want %q", messages[0].ToolCalls[0].Function.Name, "lookup")
	}
	if messages[0].ToolCalls[0].Function.Arguments != "{\"query\":\"docs\"}" {
		t.Fatalf("messages[0].ToolCalls[0].Function.Arguments = %q, want %q", messages[0].ToolCalls[0].Function.Arguments, "{\"query\":\"docs\"}")
	}
}
