package conversation

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
)

// SchemaName is the default PostgreSQL schema for conversation logging tables.
const SchemaName = "public"

// Schema DDL statements for conversation logging tables.
const ddlConversation = `
CREATE TABLE IF NOT EXISTS %s.conversation (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_key TEXT    NOT NULL,
    model           TEXT    NOT NULL DEFAULT '',
    api_type        TEXT    NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    message_count   INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_key ON %s.conversation (conversation_key);
`

const ddlConversationMessage = `
CREATE TABLE IF NOT EXISTS %s.conversation_message (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID    NOT NULL REFERENCES %s.conversation(id) ON DELETE CASCADE,
    msg_index       INTEGER NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'request',
    role            TEXT    NOT NULL,
    content         TEXT    DEFAULT '',
    content_json    TEXT    DEFAULT '',
    tool_calls_json TEXT    DEFAULT '',
    tool_call_id    TEXT    DEFAULT '',
    name            TEXT    DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (conversation_id, msg_index, source)
);
CREATE INDEX IF NOT EXISTS idx_conversation_message_cid ON %s.conversation_message (conversation_id);
`

const ddlRequestLog = `
CREATE TABLE IF NOT EXISTS %s.request_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID    NOT NULL REFERENCES %s.conversation(id) ON DELETE CASCADE,
    request_method  TEXT    NOT NULL DEFAULT '',
    request_url     TEXT    NOT NULL DEFAULT '',
    request_body    BYTEA   DEFAULT NULL,
    response_status INTEGER DEFAULT 0,
    response_body   BYTEA   DEFAULT NULL,
    model           TEXT    DEFAULT '',
    stream          BOOLEAN DEFAULT FALSE,
    request_id      TEXT    DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_ms     BIGINT  DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_request_log_cid ON %s.request_log (conversation_id);
CREATE INDEX IF NOT EXISTS idx_request_log_created ON %s.request_log (created_at);
`

// EnsureSchema creates all required tables for conversation logging.
func EnsureSchema(ctx context.Context, db *sql.DB, schema string) error {
	if db == nil {
		return fmt.Errorf("conversation: database connection is nil")
	}
	schema = strings.TrimSpace(schema)
	if schema == "" {
		schema = SchemaName
	}

	// Create schema if not default
	if schema != "public" {
		if _, err := db.ExecContext(ctx, fmt.Sprintf("CREATE SCHEMA IF NOT EXISTS %s", quoteID(schema))); err != nil {
			return fmt.Errorf("conversation: create schema: %w", err)
		}
	}

	// Enable pgcrypto extension for gen_random_uuid()
	if _, err := db.ExecContext(ctx, `CREATE EXTENSION IF NOT EXISTS pgcrypto`); err != nil {
		// Non-fatal: some databases may already have it or use built-in gen_random_uuid()
		_ = err
	}

	// Create tables
	q := fmt.Sprintf(ddlConversation, quoteID(schema), quoteID(schema))
	if _, err := db.ExecContext(ctx, q); err != nil {
		return fmt.Errorf("conversation: create conversation table: %w", err)
	}

	q = fmt.Sprintf(ddlConversationMessage, quoteID(schema), quoteID(schema), quoteID(schema))
	if _, err := db.ExecContext(ctx, q); err != nil {
		return fmt.Errorf("conversation: create conversation_message table: %w", err)
	}

	q = fmt.Sprintf(ddlRequestLog, quoteID(schema), quoteID(schema), quoteID(schema), quoteID(schema))
	if _, err := db.ExecContext(ctx, q); err != nil {
		return fmt.Errorf("conversation: create request_log table: %w", err)
	}

	return nil
}

// quoteID quotes a PostgreSQL identifier to prevent injection.
func quoteID(id string) string {
	return `"` + strings.ReplaceAll(id, `"`, `""`) + `"`
}
