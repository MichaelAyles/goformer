# goformer Implementation Plan

## Phase 1: Foundation

### 1.1 Tensor Type (`tensor.go`)
- Define `Tensor` struct: `Data []float32`, `Shape []int`
- Constructors: `NewTensor(shape...)`, `TensorFrom(data, shape...)`
- Accessors: `Row(i)`, `At2D(i,j)`, `Reshape(shape...)`, `Clone()`
- Element-wise ops: `Add`, `Mul` (Hadamard), `Scale`
- Reductions: `SumAxis`, `MaxAxis`
- Activation functions: `Softmax(axis)`, `GELU`
- No broadcasting. Shapes must match exactly or panic.

### 1.2 Matrix Multiplication (`matmul.go`)
- `MatMul(a, b)` — standard A[M,K] @ B[K,N] -> C[M,N]
- `MatMulTransB(a, b)` — A[M,K] @ B^T[N,K] -> C[M,N] (avoids transpose in attention)
- `AddBias(t, bias)` — add 1D bias to every row, in-place
- Start with naive triple loop, get correctness first
- Then add loop tiling (benchmark tile sizes 8/16/32/64)

### 1.3 Safetensors Parser (`safetensors.go`)
- Parse 8-byte LE uint64 header length
- Parse JSON header into tensor metadata map (name -> dtype, shape, offsets)
- Read tensor data by name, return `*Tensor` with correct shape
- Support F32 dtype (F16 is stretch goal)
- Read entire file into memory at load time
- `WeightMap` type that resolves HuggingFace naming conventions:
  - `encoder.layer.{N}.attention.self.{query,key,value}.{weight,bias}`
  - `encoder.layer.{N}.attention.output.dense.{weight,bias}`
  - `encoder.layer.{N}.intermediate.dense.{weight,bias}`
  - `encoder.layer.{N}.output.dense.{weight,bias}`
  - `embeddings.{word,position,token_type}_embeddings.weight`
  - `embeddings.LayerNorm.{weight,bias}`

## Phase 2: Tokeniser

### 2.1 WordPiece Tokeniser (`tokeniser.go`)
- Parse HuggingFace `tokenizer.json` to extract vocab and special tokens
- Pre-tokenisation: lowercase + strip accents (configurable for uncased models)
- Split on whitespace and punctuation
- WordPiece: greedy longest-prefix matching with `##` suffix tokens
- Prepend `[CLS]`, append `[SEP]`
- Truncate to max sequence length
- Generate attention mask (1=real, 0=padding)
- Batch tokenisation: pad all sequences to longest in batch, return 2D token IDs + attention mask

### 2.2 Tokeniser Validation
- Compare token IDs against HuggingFace Python tokeniser output for all test cases
- Edge cases: empty string, single char, long text (truncation), unicode, punctuation

## Phase 3: Model Components

### 3.1 Embedding Lookup (`embedding.go`)
- `token_embed[token_id] + position_embed[position] + type_embed[0]`
- LayerNorm the sum
- Batch: output `[batch, seqLen, hiddenSize]`

### 3.2 Layer Normalisation (`layernorm.go`)
- `(x - mean) / sqrt(var + eps) * gamma + beta`
- eps = 1e-12 (BERT default), read from config
- Operates on last dimension (normalise each row independently)

### 3.3 Multi-Head Self-Attention (`attention.go`)
- Project Q, K, V: `X @ W + b` for each
- Split into heads: reshape to `[heads, seqLen, headDim]`
- Per-head: `softmax(Q @ K^T / sqrt(headDim) + mask) @ V`
- Mask: -10000.0 for padding, 0.0 for real tokens
- Concatenate heads, output projection
- Return attention output (caller handles residual)

### 3.4 Feed-Forward Network (`feedforward.go`)
- `GELU(X @ W1 + b1) @ W2 + b2`
- GELU tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`

### 3.5 Transformer Layer (`transformer.go`)
- `X = LayerNorm(X + attention(X))`
- `X = LayerNorm(X + ffn(X))`
- Stack N layers (read from config)

### 3.6 Pooling (`pooling.go`)
- Mean pooling: average hidden states over non-padding tokens per dimension
- CLS pooling: take hidden state at position 0
- L2 normalisation: `x / sqrt(sum(x^2))`
- Default to whichever matches Python reference output

## Phase 4: Integration

### 4.1 Model Config
- Parse `config.json` (HuggingFace format)
- Extract: vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, max_position_embeddings, layer_norm_eps
- Derive: `headDim = hiddenSize / numHeads` (assert divides evenly)

### 4.2 Public API (`goformer.go`)
- `Load(path string) (*Model, error)` — read config, weights, tokeniser
- `Model.Embed(text string) ([]float32, error)` — tokenise, forward pass, pool, normalise
- `Model.EmbedBatch(texts []string) ([][]float32, error)` — batch version
- `Model.Dims() int`, `Model.MaxSeqLen() int`

### 4.3 Reference Data Generation (`testdata/generate_references.py`)
- Load BGE-small-en-v1.5 via HuggingFace transformers
- For each test input, dump:
  - Token IDs and attention mask
  - Post-embedding tensor
  - Post-attention and post-FFN tensors per layer
  - Final hidden states
  - Pooled + normalised output
- Save as JSON in `testdata/`

### 4.4 End-to-End Tests (`goformer_test.go`)
- Token ID exact match vs Python
- Intermediate tensor match (tolerance 1e-5)
- Final embedding match (tolerance 1e-4)
- Cosine similarity Go vs Python > 0.9999
- Semantic sanity: similar texts → high similarity, dissimilar texts → low similarity
- Test inputs: short text, paragraph, truncation-length, empty-ish, unicode, punctuation

### 4.5 Benchmarks (`bench_test.go`)
- Per-component: MatMul (384x384, 384x1536), Tokenise, EmbedLookup, Attention, FFN, LayerNorm
- End-to-end: FullInference, Batch_1, Batch_8, Batch_32

## Phase 5: Optimisation

### 5.1 MatMul Tiling
- Profile to confirm matmul is the bottleneck
- Benchmark tile sizes (8, 16, 32, 64)
- Apply best tiling to both MatMul and MatMulTransB

### 5.2 Memory Reuse
- Pre-allocate workspace tensors for batch inference
- Reuse intermediate buffers across layers instead of allocating per layer

### 5.3 Performance Validation
- Single embed < 50ms (short text, ~20 tokens)
- Batch of 32 < 500ms
- Model load < 2s
- Memory < 200MB with model loaded

## Implementation Order

```
tensor.go          ← start here, everything depends on it
matmul.go          ← hot path, get it right early
safetensors.go     ← need weights to test anything real
tokeniser.go       ← independent of the math path
layernorm.go       ← simple, needed by embedding and transformer
embedding.go       ← first real model component
feedforward.go     ← needs matmul + GELU
attention.go       ← most complex single component
transformer.go     ← composes the above
pooling.go         ← straightforward
goformer.go        ← ties it all together
generate_references.py ← can be written anytime, run before Go tests
goformer_test.go   ← validate against references
bench_test.go      ← measure, then optimise
```

## Stretch Goals (post-v1)
- F16 safetensors loading (convert to F32 at load time)
- INT8 quantised inference
- Sharded safetensors support (multiple files)
- Go assembly SIMD matmul kernels for amd64/arm64
