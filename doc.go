// Package goformer provides pure Go BERT-family transformer inference.
//
// It loads model weights directly from HuggingFace safetensors format and
// runs inference to produce embeddings. No CGO, no ONNX, no native dependencies.
//
// # Quick Start
//
// Point it at a HuggingFace model directory containing config.json,
// tokenizer.json, and model.safetensors:
//
//	model, err := goformer.Load("./bge-small-en-v1.5")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	embedding, err := model.Embed("DMA channel configuration")
//	// embedding is a []float32 of length model.Dims()
//
// # Supported Models
//
// Any BERT-family encoder model published in safetensors format on HuggingFace
// should work. The reference model is BGE-small-en-v1.5 (384-dim, 6 layers, 33M params).
// Both F32 and F16 safetensors weights are supported (F16 is converted to F32 at load time).
//
// # Embeddings
//
// Embed and EmbedBatch produce L2-normalised embeddings using mean pooling over
// non-padding tokens. The output vectors can be compared directly using dot product
// (equivalent to cosine similarity for unit vectors).
package goformer
