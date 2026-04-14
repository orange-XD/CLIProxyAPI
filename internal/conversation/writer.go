package conversation

import (
	"context"
	"encoding/json"
	"sync"
	"time"
)

// ConversationStreamWriter implements the streaming log writer that collects
// SSE chunks for a streaming request and persists the assembled conversation
// to PostgreSQL on Close().
type ConversationStreamWriter struct {
	logger    *ConversationLogger
	parser    RequestParser
	parsed    *ParsedRequest
	reqBody   []byte
	method    string
	url       string
	requestID string
	startTime time.Time

	mu                   sync.Mutex
	chunks               [][]byte
	status               int
	headers              map[string][]string
	apiRequest           []byte
	apiResponse          []byte
	apiWebsocketTimeline []byte
	firstChunk           time.Time
	closed               bool
}

// WriteChunkAsync appends a streaming response chunk.
func (w *ConversationStreamWriter) WriteChunkAsync(chunk []byte) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return
	}
	if len(chunk) > 0 {
		cp := make([]byte, len(chunk))
		copy(cp, chunk)
		w.chunks = append(w.chunks, cp)
	}
}

// WriteStatus records the response status code and headers.
func (w *ConversationStreamWriter) WriteStatus(status int, headers map[string][]string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.status = status
	w.headers = headers
	return nil
}

// WriteAPIRequest records the upstream API request data.
func (w *ConversationStreamWriter) WriteAPIRequest(apiRequest []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(apiRequest) > 0 {
		w.apiRequest = make([]byte, len(apiRequest))
		copy(w.apiRequest, apiRequest)
	}
	return nil
}

// WriteAPIResponse records the upstream API response data.
func (w *ConversationStreamWriter) WriteAPIResponse(apiResponse []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(apiResponse) > 0 {
		w.apiResponse = make([]byte, len(apiResponse))
		copy(w.apiResponse, apiResponse)
	}
	return nil
}

// WriteAPIWebsocketTimeline records the upstream websocket event timeline.
func (w *ConversationStreamWriter) WriteAPIWebsocketTimeline(apiWebsocketTimeline []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(apiWebsocketTimeline) > 0 {
		w.apiWebsocketTimeline = make([]byte, len(apiWebsocketTimeline))
		copy(w.apiWebsocketTimeline, apiWebsocketTimeline)
	}
	return nil
}

// SetFirstChunkTimestamp records the TTFB timestamp.
func (w *ConversationStreamWriter) SetFirstChunkTimestamp(timestamp time.Time) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.firstChunk = timestamp
}

// Close finalizes the streaming log: assembles response, parses messages,
// and persists the full conversation to PostgreSQL asynchronously.
func (w *ConversationStreamWriter) Close() error {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return nil
	}
	w.closed = true

	// Snapshot under lock
	chunks := w.chunks
	status := w.status
	reqBody := w.reqBody
	apiResp := w.apiResponse
	parsed := w.parsed
	method := w.method
	url := w.url
	requestID := w.requestID
	startTime := w.startTime
	firstChunk := w.firstChunk
	w.mu.Unlock()

	// Assemble raw streaming response body
	respBody := collectStreamingResponse(chunks)

	// Truncate if needed
	if w.logger.cfg.LogRequestBody {
		reqBody = w.logger.truncateBody(reqBody)
	} else {
		reqBody = nil
	}

	var respBodyToLog []byte
	if w.logger.cfg.LogResponseBody {
		respBodyToLog = w.logger.truncateBody(respBody)
	}

	// Parse response messages from the assembled body
	var responseMessages []UnifiedMessage
	for _, candidate := range [][]byte{respBody, apiResp} {
		if len(candidate) == 0 {
			continue
		}
		responseMessages = w.parser.ParseResponse(candidate)
		if len(responseMessages) > 0 {
			break
		}
	}

	// If still no parsed messages, try parsing the streaming SSE data
	if len(responseMessages) == 0 && len(chunks) > 0 {
		responseMessages = w.parseStreamingChunks(chunks)
	}

	// Calculate duration
	durationMs := int64(0)
	if !firstChunk.IsZero() {
		durationMs = time.Since(startTime).Milliseconds()
	}

	// Enqueue async write
	w.logger.enqueueWrite(func(ctx context.Context) error {
		return w.logger.writeConversation(ctx, parsed, reqBody, respBodyToLog, method, url, status, requestID, durationMs, responseMessages)
	})

	return nil
}

// parseStreamingChunks attempts to extract the final assembled message from
// SSE streaming chunks for OpenAI Chat Completions format.
// It looks for the final non-[DONE] chunk containing "choices" with finish_reason.
func (w *ConversationStreamWriter) parseStreamingChunks(chunks [][]byte) []UnifiedMessage {
	// Collect all SSE data lines
	var assembledContent string
	var role string
	var toolCalls []ToolCall
	toolCallMap := make(map[int]*ToolCall)

	for _, chunk := range chunks {
		// Parse SSE lines
		lines := splitSSELines(chunk)
		for _, line := range lines {
			data := extractSSEData(line)
			if data == "" || data == "[DONE]" {
				continue
			}

			var sseObj struct {
				Choices []struct {
					Delta struct {
						Role      string `json:"role"`
						Content   string `json:"content"`
						ToolCalls []struct {
							Index    int    `json:"index"`
							ID       string `json:"id"`
							Type     string `json:"type"`
							Function struct {
								Name      string `json:"name"`
								Arguments string `json:"arguments"`
							} `json:"function"`
						} `json:"tool_calls"`
					} `json:"delta"`
					FinishReason *string `json:"finish_reason"`
				} `json:"choices"`
			}

			if err := json.Unmarshal([]byte(data), &sseObj); err != nil {
				continue
			}

			for _, choice := range sseObj.Choices {
				if choice.Delta.Role != "" {
					role = choice.Delta.Role
				}
				assembledContent += choice.Delta.Content

				// Accumulate tool calls by index
				for _, tc := range choice.Delta.ToolCalls {
					existing, ok := toolCallMap[tc.Index]
					if !ok {
						existing = &ToolCall{
							ID:   tc.ID,
							Type: tc.Type,
							Function: FunctionCall{
								Name:      tc.Function.Name,
								Arguments: tc.Function.Arguments,
							},
						}
						toolCallMap[tc.Index] = existing
					} else {
						if tc.ID != "" {
							existing.ID = tc.ID
						}
						if tc.Type != "" {
							existing.Type = tc.Type
						}
						if tc.Function.Name != "" {
							existing.Function.Name = tc.Function.Name
						}
						existing.Function.Arguments += tc.Function.Arguments
					}
				}
			}
		}
	}

	// Convert tool call map to slice
	if len(toolCallMap) > 0 {
		toolCalls = make([]ToolCall, 0, len(toolCallMap))
		for i := 0; i < len(toolCallMap); i++ {
			if tc, ok := toolCallMap[i]; ok {
				toolCalls = append(toolCalls, *tc)
			}
		}
	}

	if assembledContent == "" && len(toolCalls) == 0 {
		return nil
	}

	if role == "" {
		role = "assistant"
	}

	return []UnifiedMessage{
		{
			Role:      MessageRole(role),
			Content:   assembledContent,
			ToolCalls: toolCalls,
			Source:    "response",
			MsgIndex:  0, // Will be re-indexed by writeConversation
		},
	}
}

// splitSSELines splits raw SSE data into individual lines.
func splitSSELines(data []byte) []string {
	var lines []string
	start := 0
	for i, b := range data {
		if b == '\n' {
			line := string(data[start:i])
			if line != "" {
				lines = append(lines, line)
			}
			start = i + 1
		}
	}
	if start < len(data) {
		line := string(data[start:])
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

// extractSSEData extracts the data payload from an SSE line.
func extractSSEData(line string) string {
	if len(line) > 6 && line[:6] == "data: " {
		return line[6:]
	}
	if len(line) > 5 && line[:5] == "data:" {
		return line[5:]
	}
	return ""
}
