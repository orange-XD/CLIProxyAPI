package conversation

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/logging"
	log "github.com/sirupsen/logrus"

	_ "github.com/jackc/pgx/v5/stdlib"
)

// ConversationLogger implements the RequestLogger interface for PostgreSQL-based
// conversation logging with message deduplication.
type ConversationLogger struct {
	cfg      config.ConversationLogConfig
	db       *sql.DB
	schema   string
	registry *ParserRegistry
	enabled  bool
	mu       sync.RWMutex

	// writeQueue is a buffered channel for async database writes.
	writeQueue chan writeOp
	wg         sync.WaitGroup
	cancel     context.CancelFunc
}

var _ logging.RequestLogger = (*ConversationLogger)(nil)
var _ logging.StreamingLogWriter = (*ConversationStreamWriter)(nil)

const messageInsertBatchSize = 500

// writeOp represents an asynchronous database write operation.
type writeOp struct {
	fn   func(ctx context.Context) error
	done chan error
}

// NewConversationLogger creates a new ConversationLogger.
// It connects to PostgreSQL, ensures schema, and starts the async write worker.
func NewConversationLogger(cfg config.ConversationLogConfig) (*ConversationLogger, error) {
	dsn := strings.TrimSpace(cfg.DSN)
	if dsn == "" {
		dsn = strings.TrimSpace(os.Getenv("PGSTORE_DSN"))
	}
	if dsn == "" {
		return nil, fmt.Errorf("conversation: DSN is required (set conversation-log.dsn or PGSTORE_DSN env)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return nil, fmt.Errorf("conversation: open database: %w", err)
	}
	if err = db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("conversation: ping database: %w", err)
	}

	// Set connection pool defaults
	db.SetMaxOpenConns(5)
	db.SetMaxIdleConns(2)
	db.SetConnMaxLifetime(30 * time.Minute)

	schema := strings.TrimSpace(cfg.Schema)
	if schema == "" {
		schema = "public"
	}

	if err = EnsureSchema(ctx, db, schema); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("conversation: ensure schema: %w", err)
	}

	logger := &ConversationLogger{
		cfg:        cfg,
		db:         db,
		schema:     schema,
		registry:   NewParserRegistry(),
		enabled:    true,
		writeQueue: make(chan writeOp, 256),
	}
	// Start async write worker
	workerCtx, workerCancel := context.WithCancel(context.Background())
	logger.cancel = workerCancel
	logger.wg.Add(1)

	go logger.writeWorker(workerCtx)

	log.Infof("conversation logger initialized (schema=%s)", schema)
	return logger, nil
}

// writeWorker processes async database writes from the queue.
func (l *ConversationLogger) writeWorker(ctx context.Context) {
	defer l.wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case op := <-l.writeQueue:
			writeErr := op.fn(ctx)
			if writeErr != nil {
				log.Debugf("conversation: async write error: %v", writeErr)
			}
			if op.done != nil {
				op.done <- writeErr
			}
		}
	}
}

// enqueueWrite adds a write operation to the async queue.
func (l *ConversationLogger) enqueueWrite(fn func(ctx context.Context) error) {
	select {
	case l.writeQueue <- writeOp{fn: fn}:
	default:
		log.Debug("conversation: write queue full, dropping operation")
	}
}

// Close shuts down the logger, draining pending writes and closing the DB connection.
func (l *ConversationLogger) Close() error {
	if l.cancel != nil {
		l.cancel()
	}
	l.wg.Wait()
	if l.db != nil {
		return l.db.Close()
	}
	return nil
}

// SetEnabled dynamically enables or disables conversation logging.
func (l *ConversationLogger) SetEnabled(enabled bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.enabled = enabled
}

// IsEnabled returns whether conversation logging is currently active.
func (l *ConversationLogger) IsEnabled() bool {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.enabled
}

// IsEnabled implements the RequestLogger interface.
func (l *ConversationLogger) LogRequest(
	url, method string,
	requestHeaders map[string][]string,
	body []byte,
	statusCode int,
	responseHeaders map[string][]string,
	response, websocketTimeline, apiRequest, apiResponse, apiWebsocketTimeline []byte,
	apiResponseErrors []*interfaces.ErrorMessage,
	requestID string,
	requestTimestamp, apiResponseTimestamp time.Time,
) error {
	if !l.IsEnabled() {
		return nil
	}

	// Find matching parser
	parser := l.registry.FindParser(url)
	if parser == nil {
		return nil // Not a conversation-type endpoint
	}

	// Parse request
	parsed := parser.ParseRequest(body)
	if parsed == nil || parsed.ConversationKey == "" {
		return nil
	}

	// Check model exclusion
	if l.isModelExcluded(parsed.Model) {
		return nil
	}

	// Truncate bodies if needed
	reqBody := body
	respBody := response
	if !l.cfg.LogRequestBody {
		reqBody = nil
	}
	if !l.cfg.LogResponseBody {
		respBody = nil
	}
	reqBody = l.truncateBody(reqBody)
	respBody = l.truncateBody(respBody)

	// Parse response messages
	var responseMessages []UnifiedMessage
	for _, candidate := range [][]byte{response, apiResponse} {
		if len(candidate) == 0 {
			continue
		}
		responseMessages = parser.ParseResponse(candidate)
		if len(responseMessages) > 0 {
			break
		}
	}

	// Calculate duration
	durationMs := int64(0)
	if !apiResponseTimestamp.IsZero() && !requestTimestamp.IsZero() {
		durationMs = apiResponseTimestamp.Sub(requestTimestamp).Milliseconds()
	}

	clientIP := extractClientIP(requestHeaders)

	// Enqueue async write
	l.enqueueWrite(func(ctx context.Context) error {
		return l.writeConversation(ctx, parsed, reqBody, respBody, method, url, statusCode, requestID, durationMs, responseMessages, clientIP)
	})

	return nil
}

// LogStreamingRequest initiates logging for a streaming request and returns a StreamingLogWriter.
func (l *ConversationLogger) LogStreamingRequest(
	url, method string,
	headers map[string][]string,
	body []byte,
	requestID string,
) (logging.StreamingLogWriter, error) {
	if !l.IsEnabled() {
		return nil, nil
	}

	parser := l.registry.FindParser(url)
	if parser == nil {
		return nil, nil
	}

	parsed := parser.ParseRequest(body)
	if parsed == nil || parsed.ConversationKey == "" {
		return nil, nil
	}

	if l.isModelExcluded(parsed.Model) {
		return nil, nil
	}

	reqBody := body
	if !l.cfg.LogRequestBody {
		reqBody = nil
	}
	reqBody = l.truncateBody(reqBody)

	clientIP := extractClientIP(headers)

	writer := &ConversationStreamWriter{
		logger:    l,
		parser:    parser,
		parsed:    parsed,
		reqBody:   reqBody,
		method:    method,
		url:       url,
		requestID: requestID,
		startTime: time.Now(),
		chunks:    make([][]byte, 0, 64),
		clientIP:  clientIP,
	}

	return writer, nil
}

// writeConversation persists a complete conversation turn to the database.
func (l *ConversationLogger) writeConversation(
	ctx context.Context,
	parsed *ParsedRequest,
	reqBody, respBody []byte,
	method, url string,
	statusCode int,
	requestID string,
	durationMs int64,
	responseMessages []UnifiedMessage,
	clientIP string,
) error {
	tx, err := l.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer func() {
		if errRollback := tx.Rollback(); errRollback != nil && errRollback != sql.ErrTxDone {
			log.Debugf("conversation: rollback transaction: %v", errRollback)
		}
	}()

	// Upsert conversation
	conversationID, err := l.upsertConversation(ctx, tx, parsed, clientIP)
	if err != nil {
		return fmt.Errorf("upsert conversation: %w", err)
	}

	// Clear existing messages for the conversation to handle updates
	if err := l.clearMessagesByConversationID(ctx, tx, conversationID); err != nil {
		return fmt.Errorf("clear messages: %w", err)
	}

	allMessages := make([]UnifiedMessage, 0, len(parsed.Messages)+len(responseMessages))
	allMessages = append(allMessages, parsed.Messages...)
	if len(responseMessages) > 0 {
		// Re-index response messages starting after the last request message
		offset := len(parsed.Messages)
		for i, msg := range responseMessages {
			msg.MsgIndex = offset + i
			allMessages = append(allMessages, msg)
		}
	}

	// Insert request/response messages with deduplication.
	if err := l.insertMessages(ctx, tx, conversationID, allMessages); err != nil {
		return fmt.Errorf("insert messages: %w", err)
	}

	// Insert request log
	// if err := l.insertRequestLog(ctx, tx, conversationID, reqBody, respBody, method, url, statusCode, parsed, requestID, durationMs); err != nil {
	// 	log.Debugf("conversation: insert request log: %v", err)
	// }

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}

	return nil
}

// upsertConversation creates or retrieves a conversation by conversation_key.
func (l *ConversationLogger) upsertConversation(ctx context.Context, tx *sql.Tx, parsed *ParsedRequest, clientIP string) (string, error) {
	table := quoteID(l.schema) + ".conversation"

	var id string
	err := tx.QueryRowContext(ctx, fmt.Sprintf(`
		INSERT INTO %s (conversation_key, model, api_type, updated_at, message_count, client_ip, request_count)
		VALUES ($1, $2, $3, NOW(), $4, $5, 1)
		ON CONFLICT (conversation_key) DO UPDATE SET
			model = EXCLUDED.model,
			updated_at = NOW(),
			message_count = EXCLUDED.message_count,
			client_ip = EXCLUDED.client_ip,
			request_count = %s.request_count + 1
		RETURNING id
	`, table, table), parsed.ConversationKey, parsed.Model, parsed.APIType, len(parsed.Messages), clientIP).Scan(&id)
	if err != nil {
		return "", err
	}
	return id, nil
}

// insertMessages inserts messages with ON CONFLICT DO NOTHING for deduplication.
func (l *ConversationLogger) insertMessages(ctx context.Context, tx *sql.Tx, conversationID string, messages []UnifiedMessage) error {
	if len(messages) == 0 {
		return nil
	}

	table := quoteID(l.schema) + ".conversation_message"

	for start := 0; start < len(messages); start += messageInsertBatchSize {
		end := start + messageInsertBatchSize
		if end > len(messages) {
			end = len(messages)
		}

		query, args := buildInsertMessagesQuery(table, conversationID, messages[start:end])
		if _, err := tx.ExecContext(ctx, query, args...); err != nil {
			return err
		}
	}

	return nil
}

// clearMessagesByConversationID removes all stored messages for a conversation and resets its message count.
func (l *ConversationLogger) clearMessagesByConversationID(ctx context.Context, tx *sql.Tx, conversationID string) error {
	if strings.TrimSpace(conversationID) == "" {
		return fmt.Errorf("conversation: conversationID is required")
	}

	messageTable := quoteID(l.schema) + ".conversation_message"

	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`
		DELETE FROM %s
		WHERE conversation_id = $1
	`, messageTable), conversationID); err != nil {
		return err
	}

	return nil
}

// extractClientIP extracts the client IP from request headers.
func extractClientIP(headers map[string][]string) string {
	if values, ok := headers["X-Internal-Client-IP"]; ok && len(values) > 0 {
		return values[0]
	}
	return ""
}

// insertRequestLog inserts a request log record.
func (l *ConversationLogger) insertRequestLog(
	ctx context.Context,
	tx *sql.Tx,
	conversationID string,
	reqBody, respBody []byte,
	method, url string,
	statusCode int,
	parsed *ParsedRequest,
	requestID string,
	durationMs int64,
) error {
	table := quoteID(l.schema) + ".request_log"

	_, err := tx.ExecContext(ctx, fmt.Sprintf(`
		INSERT INTO %s (conversation_id, request_method, request_url, request_body, response_status, response_body, model, stream, request_id, duration_ms)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NULLIF($9, ''), $10)
	`, table),
		conversationID, method, url, reqBody, statusCode, respBody,
		parsed.Model, parsed.Stream, requestID, durationMs,
	)
	return err
}

// truncateBody truncates a body to the configured max size.
func (l *ConversationLogger) truncateBody(body []byte) []byte {
	if body == nil {
		return nil
	}
	if l.cfg.MaxBodySize <= 0 || len(body) <= l.cfg.MaxBodySize {
		return body
	}
	truncated := make([]byte, l.cfg.MaxBodySize)
	copy(truncated, body[:l.cfg.MaxBodySize])
	// Append truncation notice
	notice := []byte("\n...[TRUNCATED]")
	truncated = append(truncated[:l.cfg.MaxBodySize-len(notice)], notice...)
	return truncated
}

// isModelExcluded checks if a model name matches any exclusion pattern.
func (l *ConversationLogger) isModelExcluded(model string) bool {
	for _, pattern := range l.cfg.ExcludeModels {
		if matchModelPattern(model, pattern) {
			return true
		}
	}
	return false
}

// matchModelPattern checks if a model name matches a pattern (supports * wildcard).
func matchModelPattern(model, pattern string) bool {
	if pattern == "*" {
		return true
	}
	model = strings.ToLower(model)
	pattern = strings.ToLower(pattern)

	if strings.HasPrefix(pattern, "*") && strings.HasSuffix(pattern, "*") {
		return strings.Contains(model, pattern[1:len(pattern)-1])
	}
	if strings.HasPrefix(pattern, "*") {
		return strings.HasSuffix(model, pattern[1:])
	}
	if strings.HasSuffix(pattern, "*") {
		return strings.HasPrefix(model, pattern[:len(pattern)-1])
	}
	return model == pattern
}

// isPathExcluded checks if a URL path matches any exclusion pattern.
func (l *ConversationLogger) isPathExcluded(url string) bool {
	for _, pattern := range l.cfg.ExcludePaths {
		if strings.Contains(url, pattern) {
			return true
		}
	}
	return false
}

// collectStreamingResponse collects streaming chunks and returns the assembled response.
func collectStreamingResponse(chunks [][]byte) []byte {
	if len(chunks) == 0 {
		return nil
	}
	return bytes.Join(chunks, nil)
}

func buildInsertMessagesQuery(table string, conversationID string, messages []UnifiedMessage) (string, []any) {
	args := make([]any, 0, len(messages)*9)
	var builder strings.Builder

	builder.WriteString(`
			INSERT INTO `)
	builder.WriteString(table)
	builder.WriteString(` (conversation_id, msg_index, source, role, content, content_json, tool_calls_json, tool_call_id, name)
			VALUES `)

	for i, msg := range messages {
		if i > 0 {
			builder.WriteString(", ")
		}

		placeholderStart := i*9 + 1
		builder.WriteString(fmt.Sprintf(
			"($%d, $%d, $%d, $%d, $%d, NULLIF($%d, ''), NULLIF($%d, ''), NULLIF($%d, ''), NULLIF($%d, ''))",
			placeholderStart,
			placeholderStart+1,
			placeholderStart+2,
			placeholderStart+3,
			placeholderStart+4,
			placeholderStart+5,
			placeholderStart+6,
			placeholderStart+7,
			placeholderStart+8,
		))

		args = append(
			args,
			conversationID,
			msg.MsgIndex,
			msg.Source,
			string(msg.Role),
			msg.Content,
			marshalContentParts(msg.ContentParts),
			marshalToolCalls(msg.ToolCalls),
			msg.ToolCallID,
			msg.Name,
		)
	}

	builder.WriteString(`
			ON CONFLICT (conversation_id, msg_index, source) DO NOTHING
		`)

	return builder.String(), args
}

func marshalContentParts(parts []ContentPart) string {
	if len(parts) == 0 {
		return ""
	}

	data, err := json.Marshal(parts)
	if err != nil {
		return ""
	}

	return string(data)
}

func marshalToolCalls(toolCalls []ToolCall) string {
	if len(toolCalls) == 0 {
		return ""
	}

	data, err := json.Marshal(toolCalls)
	if err != nil {
		return ""
	}

	return string(data)
}
