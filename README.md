<p align="center">
  <img src="assets/goformer-logo.png" alt="goformer logo" width="400">
</p>

# goformer

Pure Go BERT-family transformer inference. No CGO. No ONNX. No native dependencies.

```go
import "github.com/MichaelAyles/goformer"

model, err := goformer.Load("./bge-small-en-v1.5")
if err != nil {
    log.Fatal(err)
}

embedding, err := model.Embed("DMA channel configuration")
// embedding is a []float32 of length model.Dims()
```

## What This Is

A Go library that loads BERT-family model weights directly from HuggingFace safetensors format and runs inference to produce embeddings. Point it at a model directory downloaded from HuggingFace, call `Embed()`, get a float32 slice. No Python export step, no ONNX conversion, no native libraries.

Reference model: [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (384-dim, 6 layers, 33M params). Any BERT-family model in safetensors format with a compatible tokeniser should work.

## Why

Every existing pure Go option for running transformer models requires ONNX format. To get ONNX, you need a Python environment with `transformers`, `optimum`, `torch`, and `onnx` — a non-trivial dependency chain just to produce the artifact your Go binary needs. If the point of writing in Go is to escape Python in production, requiring Python in your build pipeline undermines the argument.

goformer loads directly from the canonical safetensors weights published by model authors on HuggingFace. Same files, same format, same config. No intermediate conversion.

## API

```go
// Load reads model weights from a HuggingFace model directory
// (config.json, tokenizer.json, model.safetensors).
func Load(path string) (*Model, error)

// Embed produces a normalised embedding vector for the input text.
func (m *Model) Embed(text string) ([]float32, error)

// EmbedBatch produces embeddings for multiple texts, padded to the
// longest sequence and processed together.
func (m *Model) EmbedBatch(texts []string) ([][]float32, error)

// Dims returns the embedding dimensionality (e.g. 384 for BGE-small).
func (m *Model) Dims() int

// MaxSeqLen returns the maximum sequence length the model supports.
func (m *Model) MaxSeqLen() int
```

That is the entire public surface.

## Usage

1. Download a model from HuggingFace:
   ```bash
   # Using git (requires git-lfs)
   git clone https://huggingface.co/BAAI/bge-small-en-v1.5

   # Or download files manually: config.json, tokenizer.json, model.safetensors
   ```

2. Load and embed:
   ```go
   model, err := goformer.Load("./bge-small-en-v1.5")
   if err != nil {
       log.Fatal(err)
   }

   // Single embedding
   vec, err := model.Embed("What is a DMA controller?")

   // Batch embedding
   vecs, err := model.EmbedBatch([]string{
       "What is a DMA controller?",
       "How does SPI communication work?",
       "Configure the UART baud rate",
   })
   ```

3. Compute similarity:
   ```go
   func cosineSimilarity(a, b []float32) float32 {
       var dot float32
       for i := range a {
           dot += a[i] * b[i]
       }
       return dot // vectors are already L2-normalised
   }
   ```

## Performance

Benchmarked with BGE-small-en-v1.5 on Apple M1:

| Operation | Time | Allocs |
|---|---|---|
| Model load | 91ms | 261MB |
| Embed (short text, ~5 tokens) | 154ms | 1.6MB |
| Embed (medium text, ~11 tokens) | 287ms | 3.0MB |
| Embed (long text, ~40 tokens) | 1.1s | 12.7MB |
| Batch of 8 | 2.4s | 24.6MB |
| MatMul 384×384 | 31ms | 0.6MB |
| MatMul 384×1536 | 158ms | 2.3MB |
| Tokenise | 1.9µs | 1KB |

This is not competing with ONNX Runtime on throughput. It is fast enough for applications where embedding latency is not the bottleneck (search indexing, RAG pipelines, document processing). MatMul is the clear bottleneck — there is headroom for tiling optimisation and SIMD.

## How It Works

1. **Safetensors parser** reads the binary weight file directly — 8-byte header length, JSON metadata, raw float32 data at specified offsets.
2. **WordPiece tokeniser** parses HuggingFace `tokenizer.json` and produces token IDs + attention masks.
3. **BERT forward pass**: embedding lookup → N transformer layers (self-attention + FFN + layer norm with residual connections) → mean pooling → L2 normalisation.
4. All math is pure Go float32 operations. Matrix multiplication uses loop tiling for cache locality.

## Correctness

All outputs validated against the HuggingFace Python `transformers` library:

| Test case | Cosine similarity | Max element-wise diff |
|---|---|---|
| `DMA channel configuration` | 1.000000 | 0.000292 |
| `The quick brown fox jumps over the lazy dog` | 0.999999 | 0.000389 |
| `Hello` | 0.999999 | 0.000213 |
| Long paragraph (40 tokens) | 0.999999 | 0.000241 |
| `café résumé naïve` (unicode) | 0.999999 | 0.000211 |
| `Hello, world! How's it going?` | 1.000000 | 0.000199 |

- Token IDs: exact match against Python for all test cases
- Batch embeddings match single embeddings

## Limitations

- **CPU only.** No GPU acceleration.
- **Inference only.** No training or fine-tuning.
- **BERT-family only.** Encoder models with safetensors weights. Not GPT, T5, or other architectures.
- **F32 and F16 weights.** Float16 weights are converted to float32 at load time.

## License

MIT
