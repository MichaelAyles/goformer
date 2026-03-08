package goformer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"
)

// tokeniser implements WordPiece tokenisation from a HuggingFace tokenizer.json.
type tokeniser struct {
	vocab        map[string]int
	idToWord     map[int]string
	clsID        int
	sepID        int
	padID        int
	unkID        int
	maxLen       int
	doLower      bool
	stripAccents bool
	cleanText    bool
}

// tokenizerJSON is the top-level structure of a HuggingFace tokenizer.json.
type tokenizerJSON struct {
	Model        tokenizerModel       `json:"model"`
	AddedTokens  []addedToken         `json:"added_tokens"`
	PreTokenizer *json.RawMessage     `json:"pre_tokenizer"`
	Normalizer   *tokenizerNormalizer `json:"normalizer"`
	TruncParams  *tokenizerTruncation `json:"truncation"`
}

type tokenizerModel struct {
	Type  string         `json:"type"`
	Vocab map[string]int `json:"vocab"`
}

type addedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type tokenizerNormalizer struct {
	Type         string `json:"type"`
	Lowercase    bool   `json:"lowercase"`
	StripAccents *bool  `json:"strip_accents"`
	CleanText    bool   `json:"clean_text"`
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

	if tj.Normalizer != nil {
		t.doLower = tj.Normalizer.Lowercase
		t.cleanText = tj.Normalizer.CleanText
		// BertNormalizer: strip_accents defaults to same as lowercase when null.
		if tj.Normalizer.StripAccents != nil {
			t.stripAccents = *tj.Normalizer.StripAccents
		} else {
			t.stripAccents = tj.Normalizer.Lowercase
		}
	}

	return t, nil
}

// normalise applies BERT text normalisation.
func (t *tokeniser) normalise(text string) string {
	if t.cleanText {
		text = cleanText(text)
	}
	if t.doLower {
		text = strings.ToLower(text)
	}
	if t.stripAccents {
		text = stripAccents(text)
	}
	return text
}

// cleanText removes control characters and replaces whitespace with spaces.
func cleanText(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		if r == 0 || r == utf8.RuneError || unicode.Is(unicode.Cc, r) {
			if r == '\t' || r == '\n' || r == '\r' {
				b.WriteRune(' ')
			}
			i += size
			continue
		}
		b.WriteRune(r)
		i += size
	}
	return b.String()
}

// stripAccents removes combining diacritical marks after NFD decomposition.
// Uses Go's built-in unicode tables rather than x/text/unicode/norm.
func stripAccents(text string) string {
	// Manual NFD-like decomposition for common accented Latin characters,
	// then strip any combining marks.
	decomposed := nfdDecompose(text)
	var b strings.Builder
	b.Grow(len(decomposed))
	for _, r := range decomposed {
		if !unicode.Is(unicode.Mn, r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// nfdDecompose performs canonical decomposition of a string.
// This covers the common accented Latin characters used in Western European languages.
func nfdDecompose(s string) string {
	var b strings.Builder
	b.Grow(len(s) * 2)
	for _, r := range s {
		if decomp, ok := nfdTable[r]; ok {
			for _, dr := range decomp {
				b.WriteRune(dr)
			}
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// nfdTable maps precomposed characters to their NFD decomposition.
//
//nolint:gochecknoglobals
var nfdTable = map[rune][]rune{
	'À': {'A', '\u0300'}, 'Á': {'A', '\u0301'}, 'Â': {'A', '\u0302'}, 'Ã': {'A', '\u0303'}, 'Ä': {'A', '\u0308'},
	'Å': {'A', '\u030A'}, 'Ç': {'C', '\u0327'}, 'È': {'E', '\u0300'}, 'É': {'E', '\u0301'}, 'Ê': {'E', '\u0302'},
	'Ë': {'E', '\u0308'}, 'Ì': {'I', '\u0300'}, 'Í': {'I', '\u0301'}, 'Î': {'I', '\u0302'}, 'Ï': {'I', '\u0308'},
	'Ñ': {'N', '\u0303'}, 'Ò': {'O', '\u0300'}, 'Ó': {'O', '\u0301'}, 'Ô': {'O', '\u0302'}, 'Õ': {'O', '\u0303'},
	'Ö': {'O', '\u0308'}, 'Ù': {'U', '\u0300'}, 'Ú': {'U', '\u0301'}, 'Û': {'U', '\u0302'}, 'Ü': {'U', '\u0308'},
	'Ý': {'Y', '\u0301'}, 'à': {'a', '\u0300'}, 'á': {'a', '\u0301'}, 'â': {'a', '\u0302'}, 'ã': {'a', '\u0303'},
	'ä': {'a', '\u0308'}, 'å': {'a', '\u030A'}, 'ç': {'c', '\u0327'}, 'è': {'e', '\u0300'}, 'é': {'e', '\u0301'},
	'ê': {'e', '\u0302'}, 'ë': {'e', '\u0308'}, 'ì': {'i', '\u0300'}, 'í': {'i', '\u0301'}, 'î': {'i', '\u0302'},
	'ï': {'i', '\u0308'}, 'ñ': {'n', '\u0303'}, 'ò': {'o', '\u0300'}, 'ó': {'o', '\u0301'}, 'ô': {'o', '\u0302'},
	'õ': {'o', '\u0303'}, 'ö': {'o', '\u0308'}, 'ù': {'u', '\u0300'}, 'ú': {'u', '\u0301'}, 'û': {'u', '\u0302'},
	'ü': {'u', '\u0308'}, 'ý': {'y', '\u0301'}, 'ÿ': {'y', '\u0308'}, 'Ā': {'A', '\u0304'}, 'ā': {'a', '\u0304'},
	'Ă': {'A', '\u0306'}, 'ă': {'a', '\u0306'}, 'Ą': {'A', '\u0328'}, 'ą': {'a', '\u0328'}, 'Ć': {'C', '\u0301'},
	'ć': {'c', '\u0301'}, 'Č': {'C', '\u030C'}, 'č': {'c', '\u030C'}, 'Ď': {'D', '\u030C'}, 'ď': {'d', '\u030C'},
	'Ē': {'E', '\u0304'}, 'ē': {'e', '\u0304'}, 'Ė': {'E', '\u0307'}, 'ė': {'e', '\u0307'}, 'Ę': {'E', '\u0328'},
	'ę': {'e', '\u0328'}, 'Ě': {'E', '\u030C'}, 'ě': {'e', '\u030C'}, 'Ğ': {'G', '\u0306'}, 'ğ': {'g', '\u0306'},
	'Ģ': {'G', '\u0327'}, 'ģ': {'g', '\u0327'}, 'Ī': {'I', '\u0304'}, 'ī': {'i', '\u0304'}, 'Į': {'I', '\u0328'},
	'į': {'i', '\u0328'}, 'İ': {'I', '\u0307'}, 'Ķ': {'K', '\u0327'}, 'ķ': {'k', '\u0327'}, 'Ļ': {'L', '\u0327'},
	'ļ': {'l', '\u0327'}, 'Ľ': {'L', '\u030C'}, 'ľ': {'l', '\u030C'}, 'Ń': {'N', '\u0301'}, 'ń': {'n', '\u0301'},
	'Ņ': {'N', '\u0327'}, 'ņ': {'n', '\u0327'}, 'Ň': {'N', '\u030C'}, 'ň': {'n', '\u030C'}, 'Ō': {'O', '\u0304'},
	'ō': {'o', '\u0304'}, 'Ő': {'O', '\u030B'}, 'ő': {'o', '\u030B'}, 'Ŕ': {'R', '\u0301'}, 'ŕ': {'r', '\u0301'},
	'Ř': {'R', '\u030C'}, 'ř': {'r', '\u030C'}, 'Ś': {'S', '\u0301'}, 'ś': {'s', '\u0301'}, 'Ş': {'S', '\u0327'},
	'ş': {'s', '\u0327'}, 'Š': {'S', '\u030C'}, 'š': {'s', '\u030C'}, 'Ţ': {'T', '\u0327'}, 'ţ': {'t', '\u0327'},
	'Ť': {'T', '\u030C'}, 'ť': {'t', '\u030C'}, 'Ū': {'U', '\u0304'}, 'ū': {'u', '\u0304'}, 'Ů': {'U', '\u030A'},
	'ů': {'u', '\u030A'}, 'Ű': {'U', '\u030B'}, 'ű': {'u', '\u030B'}, 'Ų': {'U', '\u0328'}, 'ų': {'u', '\u0328'},
	'Ź': {'Z', '\u0301'}, 'ź': {'z', '\u0301'}, 'Ż': {'Z', '\u0307'}, 'ż': {'z', '\u0307'}, 'Ž': {'Z', '\u030C'},
	'ž': {'z', '\u030C'},
}

// tokenise converts text to token IDs and an attention mask.
func (t *tokeniser) tokenise(text string) (ids []int, mask []int) {
	text = t.normalise(text)
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
// This does NOT apply lowercasing/accent stripping — that's done in normalise().
func (t *tokeniser) preTokenise(text string) []string {
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
