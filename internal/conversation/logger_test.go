package conversation

import (
	"strings"
	"testing"
)

func TestBuildInsertMessagesQuery(t *testing.T) {
	messages := []UnifiedMessage{
		{
			Role:     RoleUser,
			Content:  "hello",
			MsgIndex: 0,
			Source:   "request",
		},
		{
			Role:     RoleAssistant,
			Content:  "done",
			MsgIndex: 1,
			Source:   "response",
			ContentParts: []ContentPart{
				{Type: "text", Text: "done"},
			},
			ToolCalls: []ToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: FunctionCall{
						Name:      "lookup",
						Arguments: `{"q":"docs"}`,
					},
				},
			},
			ToolCallID: "call_1",
			Name:       "lookup",
		},
	}

	query, args := buildInsertMessagesQuery(`"public".conversation_message`, "cid-123", messages)

	if !strings.Contains(query, `INSERT INTO "public".conversation_message`) {
		t.Fatalf("query = %q, want insert target", query)
	}
	if !strings.Contains(query, `($1, $2, $3, $4, $5, NULLIF($6, ''), NULLIF($7, ''), NULLIF($8, ''), NULLIF($9, ''))`) {
		t.Fatalf("query = %q, want first row placeholders", query)
	}
	if !strings.Contains(query, `($10, $11, $12, $13, $14, NULLIF($15, ''), NULLIF($16, ''), NULLIF($17, ''), NULLIF($18, ''))`) {
		t.Fatalf("query = %q, want second row placeholders", query)
	}
	if !strings.Contains(query, `ON CONFLICT (conversation_id, msg_index, source) DO NOTHING`) {
		t.Fatalf("query = %q, want conflict clause", query)
	}

	if len(args) != 18 {
		t.Fatalf("len(args) = %d, want 18", len(args))
	}

	if args[0] != "cid-123" || args[1] != 0 || args[2] != "request" || args[3] != "user" || args[4] != "hello" {
		t.Fatalf("first row args = %#v", args[:5])
	}
	if args[5] != "" || args[6] != "" || args[7] != "" || args[8] != "" {
		t.Fatalf("first row nullable args = %#v", args[5:9])
	}

	if args[9] != "cid-123" || args[10] != 1 || args[11] != "response" || args[12] != "assistant" || args[13] != "done" {
		t.Fatalf("second row args = %#v", args[9:14])
	}
	if args[14] != `[{"type":"text","text":"done"}]` {
		t.Fatalf("content_json = %#v", args[14])
	}
	if args[15] != `[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"docs\"}"}}]` {
		t.Fatalf("tool_calls_json = %#v", args[15])
	}
	if args[16] != "call_1" || args[17] != "lookup" {
		t.Fatalf("second row tail args = %#v", args[16:18])
	}
}
