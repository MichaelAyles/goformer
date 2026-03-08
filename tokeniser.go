package goformer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// tokeniser implements WordPiece tokenisation from a HuggingFace tokenizer.json.
type tokeniser struct {
	vocab    map[string]int
	idToWord map[int]string
	clsID    int
	sepID    int
	padID    int
	unkID    int
	maxLen   int
	doLower  bool
}

// tokenizerJSON is the top-level structure of a HuggingFace tokenizer.json.
type tokenizerJSON struct {
	Model         tokenizerModel          `json:"model"`
	AddedTokens   []addedToken            `json:"added_tokens"`
	PreTokenizer  *json.RawMessage        `json:"pre_tokenizer"`
	Normalizer    *tokenizerNormalizer     `json:"normalizer"`
	TruncParams   *tokenizerTruncation    `json:"truncation"`
}

type tokenizerModel struct {
	Type    string         `json:"type"`
	Vocab   map[string]int `json:"vocab"`
}

type addedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type tokenizerNormalizer struct {
	Type      string `json:"type"`
	Lowercase bool   `json:"lowercase"`
}

type tokenizerTruncation struct {
	MaxLength int `json:"max_length"`
}

// loadTokeniser reads tokenizer.json from a model directory.
func loadTokeniser(dir string) (*tokeniser, error) {
	path := filepath.Join(dir, "tokenizer.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokeniser: %w", err)
	}

	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("tokeniser: invalid JSON: %w", err)
	}

	if tj.Model.Type != "WordPiece" {
		return nil, fmt.Errorf("tokeniser: unsupported model type %q (need WordPiece)", tj.Model.Type)
	}

	t := &tokeniser{
		vocab:    tj.Model.Vocab,
		idToWord: make(map[int]string, len(tj.Model.Vocab)),
		maxLen:   512,
		unkID:    0,
	}

	for word, id := range tj.Model.Vocab {
		t.idToWord[id] = word
	}

	// Extract special token IDs.
	for _, at := range tj.AddedTokens {
		switch at.Content {
		case "[CLS]":
			t.clsID = at.ID
		case "[SEP]":
			t.sepID = at.ID
		case "[PAD]":
			t.padID = at.ID
		case "[UNK]":
			t.unkID = at.ID
		}
	}

	if tj.TruncParams != nil && tj.TruncParams.MaxLength > 0 {
		t.maxLen = tj.TruncParams.MaxLength
	}

	if tj.Normalizer != nil && tj.Normalizer.Lowercase {
		t.doLower = true
	}

	return t, nil
}

// tokenise converts text to token IDs and an attention mask.
func (t *tokeniser) tokenise(text string) (ids []int, mask []int) {
	words := t.preTokenise(text)

	tokens := make([]int, 0, len(words)+2)
	tokens = append(tokens, t.clsID)

	for _, word := range words {
		wpTokens := t.wordpiece(word)
		tokens = append(tokens, wpTokens...)
	}

	// Truncate to maxLen-1 to leave room for [SEP].
	if len(tokens) > t.maxLen-1 {
		tokens = tokens[:t.maxLen-1]
	}
	tokens = append(tokens, t.sepID)

	mask = make([]int, len(tokens))
	for i := range mask {
		mask[i] = 1
	}

	return tokens, mask
}

// tokeniseBatch tokenises multiple texts, padding to the longest sequence.
func (t *tokeniser) tokeniseBatch(texts []string) (ids [][]int, masks [][]int) {
	ids = make([][]int, len(texts))
	masks = make([][]int, len(texts))

	maxLen := 0
	for i, text := range texts {
		ids[i], masks[i] = t.tokenise(text)
		if len(ids[i]) > maxLen {
			maxLen = len(ids[i])
		}
	}

	// Pad to maxLen.
	for i := range ids {
		for len(ids[i]) < maxLen {
			ids[i] = append(ids[i], t.padID)
			masks[i] = append(masks[i], 0)
		}
	}

	return ids, masks
}

// preTokenise splits text into words by whitespace and punctuation.
func (t *tokeniser) preTokenise(text string) []string {
	if t.doLower {
		text = strings.ToLower(text)
	}

	var words []string
	var current []rune
	for _, r := range text {
		if unicode.IsSpace(r) {
			if len(current) > 0 {
				words = append(words, string(current))
				current = current[:0]
			}
		} else if unicode.IsPunct(r) || isChinesePunct(r) {
			if len(current) > 0 {
				words = append(words, string(current))
				current = current[:0]
			}
			words = append(words, string(r))
		} else {
			current = append(current, r)
		}
	}
	if len(current) > 0 {
		words = append(words, string(current))
	}
	return words
}

// wordpiece applies WordPiece tokenisation to a single word.
func (t *tokeniser) wordpiece(word string) []int {
	if _, ok := t.vocab[word]; ok {
		return []int{t.vocab[word]}
	}

	tokens := make([]int, 0, 4)
	runes := []rune(word)
	start := 0

	for start < len(runes) {
		end := len(runes)
		found := false
		for end > start {
			var sub string
			if start == 0 {
				sub = string(runes[start:end])
			} else {
				sub = "##" + string(runes[start:end])
			}
			if id, ok := t.vocab[sub]; ok {
				tokens = append(tokens, id)
				start = end
				found = true
				break
			}
			end--
		}
		if !found {
			tokens = append(tokens, t.unkID)
			start++
		}
	}
	return tokens
}

// isChinesePunct checks if a rune is a CJK character (treated as punctuation for splitting).
func isChinesePunct(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F)
}
