package goformer

// feedForward computes GELU(X @ W1^T + b1) @ W2^T + b2.
// x is [tokens, hidden], w1 is [intermediate, hidden], w2 is [hidden, intermediate].
// Weights are stored as [out, in] (PyTorch convention), so we use transposed matmul.
func feedForward(x, w1, b1, w2, b2 *tensor) *tensor {
	// Linear 1: [tokens, hidden] @ [intermediate, hidden]^T -> [tokens, intermediate]
	h := matMulTransB(x, w1)
	addBias(h, b1)

	// GELU activation.
	h.gelu()

	// Linear 2: [tokens, intermediate] @ [hidden, intermediate]^T -> [tokens, hidden]
	out := matMulTransB(h, w2)
	addBias(out, b2)

	return out
}
