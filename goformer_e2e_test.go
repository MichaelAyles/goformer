package goformer

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type referenceCase struct {
	Text          string    `json:"text"`
	TokenIDs      []int     `json:"token_ids"`
	AttentionMask []int     `json:"attention_mask"`
	Embedding     []float32 `json:"embedding"`
}

func loadReferences(t *testing.T) []referenceCase {
	t.Helper()
	data, err := os.ReadFile("testdata/references.json")
	if err != nil {
		t.Skipf("reference data not found (run testdata/generate_references.py): %v", err)
	}
	var refs []referenceCase
	if err := json.Unmarshal(data, &refs); err != nil {
		t.Fatalf("invalid reference JSON: %v", err)
	}
	return refs
}

func loadTestModel(t *testing.T) *Model {
	t.Helper()
	m, err := Load("models/bge-small-en-v1.5")
	if err != nil {
		t.Skipf("model not found (download bge-small-en-v1.5): %v", err)
	}
	return m
}

func cosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func TestTokenisation(t *testing.T) {
	m := loadTestModel(t)
	refs := loadReferences(t)

	for _, ref := range refs {
		t.Run(ref.Text, func(t *testing.T) {
			ids, mask := m.tok.tokenise(ref.Text)
			if len(ids) != len(ref.TokenIDs) {
				t.Fatalf("token count mismatch: got %d, want %d\ngot:  %v\nwant: %v", len(ids), len(ref.TokenIDs), ids, ref.TokenIDs)
			}
			for i := range ids {
				if ids[i] != ref.TokenIDs[i] {
					t.Errorf("token[%d] = %d, want %d\ngot:  %v\nwant: %v", i, ids[i], ref.TokenIDs[i], ids, ref.TokenIDs)
					break
				}
			}
			for i := range mask {
				if mask[i] != ref.AttentionMask[i] {
					t.Errorf("mask[%d] = %d, want %d", i, mask[i], ref.AttentionMask[i])
					break
				}
			}
		})
	}
}

func TestEmbeddings(t *testing.T) {
	m := loadTestModel(t)
	refs := loadReferences(t)

	for _, ref := range refs {
		t.Run(ref.Text, func(t *testing.T) {
			got, err := m.Embed(ref.Text)
			if err != nil {
				t.Fatalf("Embed(%q): %v", ref.Text, err)
			}

			if len(got) != len(ref.Embedding) {
				t.Fatalf("embedding dim mismatch: got %d, want %d", len(got), len(ref.Embedding))
			}

			cos := cosineSimilarity(got, ref.Embedding)
			t.Logf("cosine similarity: %.6f", cos)
			if cos < 0.99 {
				t.Errorf("cosine similarity %.6f < 0.99", cos)
			}

			// Check element-wise tolerance.
			var maxDiff float64
			for i := range got {
				diff := math.Abs(float64(got[i] - ref.Embedding[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("max element-wise diff: %.6f", maxDiff)
		})
	}
}

func TestEmbedBatch(t *testing.T) {
	m := loadTestModel(t)
	refs := loadReferences(t)

	texts := make([]string, len(refs))
	for i, ref := range refs {
		texts[i] = ref.Text
	}

	embeddings, err := m.EmbedBatch(texts)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}

	for i, ref := range refs {
		cos := cosineSimilarity(embeddings[i], ref.Embedding)
		t.Logf("[%d] %q cosine similarity: %.6f", i, ref.Text, cos)
		if cos < 0.99 {
			t.Errorf("[%d] cosine similarity %.6f < 0.99", i, cos)
		}
	}
}

func TestDimsAndMaxSeqLen(t *testing.T) {
	m := loadTestModel(t)
	if m.Dims() != 384 {
		t.Errorf("Dims() = %d, want 384", m.Dims())
	}
	if m.MaxSeqLen() != 512 {
		t.Errorf("MaxSeqLen() = %d, want 512", m.MaxSeqLen())
	}
}
