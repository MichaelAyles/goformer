# goformer

Pure Go BERT-family transformer inference library. Loads HuggingFace safetensors directly. No CGO, no ONNX, no native dependencies.

## Project Structure

```
goformer.go         # Public API: Load(), Model.Embed(), Model.EmbedBatch()
tensor.go           # Tensor type, shape handling, element-wise ops, softmax, GELU
matmul.go           # MatMul, MatMulTransB, AddBias (hot path)
safetensors.go      # Safetensors parser and weight loader
tokeniser.go        # WordPiece tokeniser (HuggingFace tokenizer.json)
embedding.go        # Token + position + type embedding lookup + LayerNorm
attention.go        # Multi-head self-attention
layernorm.go        # Layer normalisation
feedforward.go      # FFN (linear → GELU → linear)
transformer.go      # Single transformer layer (attention + LN + FFN + LN)
pooling.go          # Mean pooling + L2 normalisation
goformer_test.go    # End-to-end correctness tests against Python reference
bench_test.go       # Benchmarks at every stage
testdata/           # Python reference outputs and generation script
```

## Build & Test

```bash
go build ./...
go test ./...
go test -bench . -benchmem
```

No build tags. No CGO. No external dependencies beyond the Go standard library.

## Architecture Decisions

- **Tensor type**: Dense row-major float32. No broadcasting — shapes must match exactly.
- **MatMul**: Naive loops first, then tiled for cache locality. No SIMD intrinsics.
- **Safetensors**: Read entire file into memory (models are <150MB). Parse JSON header, extract tensors by offset.
- **Tokeniser**: Parses HuggingFace `tokenizer.json`. WordPiece with `##` suffixes.
- **Attention mask**: -10000.0 for padding positions, 0.0 for real tokens.
- **GELU**: Tanh approximation (`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`).
- **LayerNorm eps**: 1e-12 (BERT default), read from model config.
- **Pooling**: Mean pooling over non-padding tokens + L2 normalisation (verify against Python reference).
- **Residual connections**: Handled in transformer.go, not inside attention/FFN.

## Weight Name Mapping

HuggingFace safetensors names follow this pattern:
```
encoder.layer.{N}.attention.self.{query,key,value}.{weight,bias}
encoder.layer.{N}.attention.output.dense.{weight,bias}
encoder.layer.{N}.intermediate.dense.{weight,bias}
encoder.layer.{N}.output.dense.{weight,bias}
embeddings.{word,position,token_type}_embeddings.weight
embeddings.LayerNorm.{weight,bias}
```

## Reference Model

BGE-small-en-v1.5: 384-dim, 6 layers, 12 heads, 33M params, 512 max seq len.
Architecture is not hardcoded — any BERT-family safetensors model should work.

## Correctness

All Go outputs validated against Python HuggingFace transformers reference:
- Token IDs: exact match
- Intermediate tensors: tolerance 1e-5
- Final embeddings: tolerance 1e-4
- Cosine similarity Go vs Python: > 0.9999

## Conventions

- Keep the public API to exactly: `Load`, `Embed`, `EmbedBatch`, `Dims`, `MaxSeqLen`
- Everything else is unexported
- No unnecessary abstractions — this is a library, not a framework
- Profile before optimising. Benchmark at every level.
- Test inputs include: short text, paragraph, truncation-length, empty, unicode, punctuation
