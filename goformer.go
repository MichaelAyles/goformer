package goformer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Model holds a loaded BERT-family transformer model.
type Model struct {
	config    modelConfig
	tok       *tokeniser
	wordEmbed *tensor
	posEmbed  *tensor
	typeEmbed *tensor
	embLNGamma, embLNBeta *tensor
	layers    []*transformerLayer
}

// modelConfig holds values parsed from config.json.
type modelConfig struct {
	VocabSize          int     `json:"vocab_size"`
	HiddenSize         int     `json:"hidden_size"`
	NumHiddenLayers    int     `json:"num_hidden_layers"`
	NumAttentionHeads  int     `json:"num_attention_heads"`
	IntermediateSize   int     `json:"intermediate_size"`
	MaxPositionEmbed   int     `json:"max_position_embeddings"`
	LayerNormEps       float64 `json:"layer_norm_eps"`
}

// Load reads model weights, config, and tokeniser from a HuggingFace model directory.
func Load(path string) (*Model, error) {
	// Parse config.json.
	cfgData, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("goformer: reading config.json: %w", err)
	}
	var cfg modelConfig
	if err := json.Unmarshal(cfgData, &cfg); err != nil {
		return nil, fmt.Errorf("goformer: parsing config.json: %w", err)
	}
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-12
	}

	// Load tokeniser.
	tok, err := loadTokeniser(path)
	if err != nil {
		return nil, fmt.Errorf("goformer: %w", err)
	}
	tok.maxLen = cfg.MaxPositionEmbed

	// Load weights.
	sfPath, err := findSafetensorsFile(path)
	if err != nil {
		return nil, fmt.Errorf("goformer: %w", err)
	}
	wm, err := loadSafetensors(sfPath)
	if err != nil {
		return nil, fmt.Errorf("goformer: %w", err)
	}

	m := &Model{config: cfg, tok: tok}

	// Load embedding weights.
	var errs []error
	m.wordEmbed, err = wm.get("embeddings.word_embeddings.weight")
	errs = append(errs, err)
	m.posEmbed, err = wm.get("embeddings.position_embeddings.weight")
	errs = append(errs, err)
	m.typeEmbed, err = wm.get("embeddings.token_type_embeddings.weight")
	errs = append(errs, err)
	m.embLNGamma, err = wm.get("embeddings.LayerNorm.weight")
	errs = append(errs, err)
	m.embLNBeta, err = wm.get("embeddings.LayerNorm.bias")
	errs = append(errs, err)

	for _, e := range errs {
		if e != nil {
			return nil, fmt.Errorf("goformer: loading embeddings: %w", e)
		}
	}

	// Load transformer layers.
	m.layers = make([]*transformerLayer, cfg.NumHiddenLayers)
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layer, err := loadTransformerLayer(wm, i)
		if err != nil {
			return nil, fmt.Errorf("goformer: loading layer %d: %w", i, err)
		}
		m.layers[i] = layer
	}

	return m, nil
}

// loadTransformerLayer loads all weights for a single transformer layer.
func loadTransformerLayer(wm *weightMap, idx int) (*transformerLayer, error) {
	prefix := fmt.Sprintf("encoder.layer.%d", idx)
	get := func(name string) (*tensor, error) {
		return wm.get(prefix + "." + name)
	}

	var err error
	layer := &transformerLayer{attn: &attentionWeights{}}

	// Attention weights.
	layer.attn.qW, err = get("attention.self.query.weight")
	if err != nil {
		return nil, err
	}
	layer.attn.qB, err = get("attention.self.query.bias")
	if err != nil {
		return nil, err
	}
	layer.attn.kW, err = get("attention.self.key.weight")
	if err != nil {
		return nil, err
	}
	layer.attn.kB, err = get("attention.self.key.bias")
	if err != nil {
		return nil, err
	}
	layer.attn.vW, err = get("attention.self.value.weight")
	if err != nil {
		return nil, err
	}
	layer.attn.vB, err = get("attention.self.value.bias")
	if err != nil {
		return nil, err
	}
	layer.attn.outW, err = get("attention.output.dense.weight")
	if err != nil {
		return nil, err
	}
	layer.attn.outB, err = get("attention.output.dense.bias")
	if err != nil {
		return nil, err
	}

	// Attention LayerNorm.
	layer.attnLNGamma, err = get("attention.output.LayerNorm.weight")
	if err != nil {
		return nil, err
	}
	layer.attnLNBeta, err = get("attention.output.LayerNorm.bias")
	if err != nil {
		return nil, err
	}

	// FFN weights.
	layer.ffnW1, err = get("intermediate.dense.weight")
	if err != nil {
		return nil, err
	}
	layer.ffnB1, err = get("intermediate.dense.bias")
	if err != nil {
		return nil, err
	}
	layer.ffnW2, err = get("output.dense.weight")
	if err != nil {
		return nil, err
	}
	layer.ffnB2, err = get("output.dense.bias")
	if err != nil {
		return nil, err
	}

	// FFN LayerNorm.
	layer.ffnLNGamma, err = get("output.LayerNorm.weight")
	if err != nil {
		return nil, err
	}
	layer.ffnLNBeta, err = get("output.LayerNorm.bias")
	if err != nil {
		return nil, err
	}

	return layer, nil
}

// Embed produces a normalised embedding vector for the input text.
func (m *Model) Embed(text string) ([]float32, error) {
	results, err := m.EmbedBatch([]string{text})
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// EmbedBatch produces embeddings for multiple texts, padded to the
// longest sequence and processed together.
func (m *Model) EmbedBatch(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("goformer: empty input")
	}

	// Tokenise.
	tokenIDs, masks := m.tok.tokeniseBatch(texts)
	batch := len(texts)
	seqLen := len(tokenIDs[0])

	// Embedding lookup.
	eps := float32(m.config.LayerNormEps)
	hidden := m.config.HiddenSize

	x := embedLookup(tokenIDs, m.wordEmbed, m.posEmbed, m.typeEmbed, m.embLNGamma, m.embLNBeta, eps)

	// Flatten to [batch*seqLen, hidden] for transformer layers.
	x = x.reshape(batch*seqLen, hidden)

	// Run through transformer layers.
	for _, layer := range m.layers {
		x = layer.forward(x, masks, m.config.NumAttentionHeads, eps)
	}

	// Reshape back to [batch, seqLen, hidden].
	x = x.reshape(batch, seqLen, hidden)

	// Pool and normalise.
	pooled := meanPool(x, masks)
	l2Normalise(pooled)

	// Extract results.
	results := make([][]float32, batch)
	for i := 0; i < batch; i++ {
		results[i] = make([]float32, hidden)
		copy(results[i], pooled.data[i*hidden:(i+1)*hidden])
	}

	return results, nil
}

// Dims returns the embedding dimensionality.
func (m *Model) Dims() int {
	return m.config.HiddenSize
}

// MaxSeqLen returns the maximum sequence length the model supports.
func (m *Model) MaxSeqLen() int {
	return m.config.MaxPositionEmbed
}
