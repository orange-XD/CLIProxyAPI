package conversation

import "testing"

func TestOpenAIResponseParserParseRequest(t *testing.T) {
	parser := &OpenAIResponseParser{}

	body := []byte(`{
		"model":"gpt-5.4",
		"instructions":"You are helpful.",
		"stream":true,
		"input":[
			{
				"type":"message",
				"role":"user",
				"content":[
					{"type":"input_text","text":"Show me the file"},
					{"type":"input_image","image_url":"https://example.com/image.png"}
				]
			},
			{
				"type":"function_call",
				"call_id":"call_123",
				"name":"read_file",
				"arguments":"{\"path\":\"README.md\"}"
			},
			{
				"type":"function_call_output",
				"call_id":"call_123",
				"output":{"status":"ok","content":"done"}
			}
		]
	}`)

	parsed := parser.ParseRequest(body)
	if parsed == nil {
		t.Fatal("ParseRequest returned nil")
	}

	if parsed.APIType != "openai_response" {
		t.Fatalf("APIType = %q, want %q", parsed.APIType, "openai_response")
	}
	if parsed.Model != "gpt-5.4" {
		t.Fatalf("Model = %q, want %q", parsed.Model, "gpt-5.4")
	}
	if !parsed.Stream {
		t.Fatal("Stream = false, want true")
	}
	if parsed.ConversationKey != deriveConversationKey("Show me the file") {
		t.Fatalf("ConversationKey = %q, want hash of first user content", parsed.ConversationKey)
	}
	if len(parsed.Messages) != 4 {
		t.Fatalf("len(Messages) = %d, want 4", len(parsed.Messages))
	}

	if parsed.Messages[0].Role != RoleSystem || parsed.Messages[0].Content != "You are helpful." {
		t.Fatalf("system message = %+v", parsed.Messages[0])
	}

	userMessage := parsed.Messages[1]
	if userMessage.Role != RoleUser {
		t.Fatalf("user role = %q, want %q", userMessage.Role, RoleUser)
	}
	if userMessage.Content != "Show me the file" {
		t.Fatalf("user content = %q, want %q", userMessage.Content, "Show me the file")
	}
	if len(userMessage.ContentParts) != 2 {
		t.Fatalf("len(user content parts) = %d, want 2", len(userMessage.ContentParts))
	}
	if userMessage.ContentParts[1].ImageURL == nil || userMessage.ContentParts[1].ImageURL.URL != "https://example.com/image.png" {
		t.Fatalf("user image part = %+v", userMessage.ContentParts[1])
	}

	toolCallMessage := parsed.Messages[2]
	if toolCallMessage.Role != RoleAssistant {
		t.Fatalf("tool call role = %q, want %q", toolCallMessage.Role, RoleAssistant)
	}
	if len(toolCallMessage.ToolCalls) != 1 {
		t.Fatalf("len(tool calls) = %d, want 1", len(toolCallMessage.ToolCalls))
	}
	if toolCallMessage.ToolCalls[0].ID != "call_123" || toolCallMessage.ToolCalls[0].Function.Name != "read_file" {
		t.Fatalf("tool call = %+v", toolCallMessage.ToolCalls[0])
	}

	toolResultMessage := parsed.Messages[3]
	if toolResultMessage.Role != RoleTool {
		t.Fatalf("tool result role = %q, want %q", toolResultMessage.Role, RoleTool)
	}
	if toolResultMessage.ToolCallID != "call_123" {
		t.Fatalf("tool result ToolCallID = %q, want %q", toolResultMessage.ToolCallID, "call_123")
	}
	if toolResultMessage.Content != `{"status":"ok","content":"done"}` {
		t.Fatalf("tool result content = %q", toolResultMessage.Content)
	}
}

func TestOpenAIResponseParserParseResponse(t *testing.T) {
	parser := &OpenAIResponseParser{}

	body := []byte(`{
		"id":"resp_123",
		"output":[
			{
				"type":"reasoning",
				"summary":[{"type":"summary_text","text":"internal summary"}]
			},
			{
				"type":"message",
				"role":"assistant",
				"content":[
					{"type":"output_text","text":"hello"},
					{"type":"output_text","text":"world"}
				]
			},
			{
				"type":"function_call",
				"call_id":"call_456",
				"name":"search_docs",
				"arguments":"{\"query\":\"parser\"}"
			}
		]
	}`)

	messages := parser.ParseResponse(body)
	if len(messages) != 2 {
		t.Fatalf("len(messages) = %d, want 2", len(messages))
	}

	if messages[0].Role != RoleAssistant {
		t.Fatalf("messages[0].Role = %q, want %q", messages[0].Role, RoleAssistant)
	}
	if messages[0].Content != "hello\nworld" {
		t.Fatalf("messages[0].Content = %q, want %q", messages[0].Content, "hello\nworld")
	}
	if len(messages[0].ContentParts) != 2 {
		t.Fatalf("len(messages[0].ContentParts) = %d, want 2", len(messages[0].ContentParts))
	}

	if messages[1].Role != RoleAssistant {
		t.Fatalf("messages[1].Role = %q, want %q", messages[1].Role, RoleAssistant)
	}
	if len(messages[1].ToolCalls) != 1 {
		t.Fatalf("len(messages[1].ToolCalls) = %d, want 1", len(messages[1].ToolCalls))
	}
	if messages[1].ToolCalls[0].ID != "call_456" {
		t.Fatalf("messages[1].ToolCalls[0].ID = %q, want %q", messages[1].ToolCalls[0].ID, "call_456")
	}
	if messages[1].ToolCalls[0].Function.Arguments != "{\"query\":\"parser\"}" {
		t.Fatalf("messages[1].ToolCalls[0].Function.Arguments = %q", messages[1].ToolCalls[0].Function.Arguments)
	}
}

func TestOpenAIResponseParserParseResponseSSE(t *testing.T) {
	parser := &OpenAIResponseParser{}

	body := []byte(
		"event: response.output_item.added\n" +
			"data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}\n\n" +
			"event: response.output_text.delta\n" +
			"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"Hello\"}\n\n" +
			"event: response.output_text.delta\n" +
			"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\" world\"}\n\n" +
			"event: response.output_item.added\n" +
			"data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"fc_call_1\",\"type\":\"function_call\",\"status\":\"in_progress\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n" +
			"event: response.function_call_arguments.delta\n" +
			"data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":1,\"item_id\":\"fc_call_1\",\"delta\":\"{\\\"query\\\":\\\"docs\\\"}\"}\n\n" +
			"event: response.output_text.done\n" +
			"data: {\"type\":\"response.output_text.done\",\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"text\":\"Hello world\"}\n\n" +
			"event: response.output_item.done\n" +
			"data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"fc_call_1\",\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"docs\\\"}\"}}\n\n",
	)

	messages := parser.ParseResponse(body)
	if len(messages) != 2 {
		t.Fatalf("len(messages) = %d, want 2", len(messages))
	}

	if messages[0].Content != "Hello world" {
		t.Fatalf("messages[0].Content = %q, want %q", messages[0].Content, "Hello world")
	}
	if messages[0].Role != RoleAssistant {
		t.Fatalf("messages[0].Role = %q, want %q", messages[0].Role, RoleAssistant)
	}

	if len(messages[1].ToolCalls) != 1 {
		t.Fatalf("len(messages[1].ToolCalls) = %d, want 1", len(messages[1].ToolCalls))
	}
	if messages[1].ToolCalls[0].ID != "call_1" {
		t.Fatalf("messages[1].ToolCalls[0].ID = %q, want %q", messages[1].ToolCalls[0].ID, "call_1")
	}
	if messages[1].ToolCalls[0].Function.Name != "lookup" {
		t.Fatalf("messages[1].ToolCalls[0].Function.Name = %q, want %q", messages[1].ToolCalls[0].Function.Name, "lookup")
	}
	if messages[1].ToolCalls[0].Function.Arguments != "{\"query\":\"docs\"}" {
		t.Fatalf("messages[1].ToolCalls[0].Function.Arguments = %q", messages[1].ToolCalls[0].Function.Arguments)
	}
}
