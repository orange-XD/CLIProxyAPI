package conversation

// RequestParser defines the interface for parsing API request/response bodies
// into unified conversation messages. Each API type (OpenAI Chat, Claude Messages, etc.)
// implements this interface to enable extensible conversation logging.
type RequestParser interface {
	// ParseRequest parses a request body into a ParsedRequest containing
	// conversation key, model, messages, and metadata.
	// Returns nil if the body cannot be parsed by this parser.
	ParseRequest(body []byte) *ParsedRequest

	// ParseResponse parses a non-streaming response body and returns
	// assistant messages extracted from it.
	ParseResponse(body []byte) []UnifiedMessage

	// APIType returns the unique identifier for this parser's API type
	// (e.g., "openai_chat", "claude_messages", "gemini_generate").
	APIType() string

	// MatchURL returns true if the given URL path should be handled by this parser.
	MatchURL(url string) bool
}

// ParserRegistry manages registered request parsers and routes URLs to the appropriate parser.
type ParserRegistry struct {
	parsers []RequestParser
}

// NewParserRegistry creates a new parser registry with default parsers registered.
func NewParserRegistry() *ParserRegistry {
	registry := &ParserRegistry{}
	registry.Register(&OpenAIChatParser{})
	return registry
}

// Register adds a parser to the registry.
func (r *ParserRegistry) Register(parser RequestParser) {
	r.parsers = append(r.parsers, parser)
}

// FindParser returns the first parser that matches the given URL, or nil if none match.
func (r *ParserRegistry) FindParser(url string) RequestParser {
	for _, p := range r.parsers {
		if p.MatchURL(url) {
			return p
		}
	}
	return nil
}
