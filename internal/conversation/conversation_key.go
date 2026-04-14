package conversation

import (
	"crypto/sha256"
	"fmt"
)

// deriveConversationKey generates a unique key from message content
// using SHA256 hash (first 16 bytes hex-encoded).
func deriveConversationKey(content string) string {
	if content == "" {
		return ""
	}
	h := sha256.Sum256([]byte(content))
	return fmt.Sprintf("%x", h[:16])
}
