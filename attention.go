package goformer

import "math"

// attentionWeights holds the weight tensors for a single attention layer.
type attentionWeights struct {
	qW, qB   *tensor // query projection
	kW, kB   *tensor // key projection
	vW, vB   *tensor // value projection
	outW, outB *tensor // output projection
}

// selfAttention computes multi-head self-attention.
// x is [batch*seqLen, hidden], mask is [batch, seqLen] (1=real, 0=pad).
func selfAttention(x *tensor, mask [][]int, numHeads int, w *attentionWeights) *tensor {
	totalTokens := x.shape[0]
	hidden := x.shape[1]
	headDim := hidden / numHeads

	// Project Q, K, V: [totalTokens, hidden] @ [hidden, hidden]^T -> [totalTokens, hidden]
	// Weights are stored as [out, in] (PyTorch convention), so we use transposed matmul.
	q := matMulTransB(x, w.qW)
	addBias(q, w.qB)
	k := matMulTransB(x, w.kW)
	addBias(k, w.kB)
	v := matMulTransB(x, w.vW)
	addBias(v, w.vB)

	batch := len(mask)
	seqLen := totalTokens / batch
	scaleFactor := float32(1.0 / math.Sqrt(float64(headDim)))

	// Output buffer.
	out := newTensor(totalTokens, hidden)

	// Process per batch item, per head.
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			// Extract Q, K, V slices for this head.
			// Q[b,s,h,:] is at q.data[(b*seqLen+s)*hidden + h*headDim]
			qh := newTensor(seqLen, headDim)
			kh := newTensor(seqLen, headDim)
			vh := newTensor(seqLen, headDim)

			for s := 0; s < seqLen; s++ {
				srcOff := (b*seqLen+s)*hidden + h*headDim
				dstOff := s * headDim
				copy(qh.data[dstOff:dstOff+headDim], q.data[srcOff:srcOff+headDim])
				copy(kh.data[dstOff:dstOff+headDim], k.data[srcOff:srcOff+headDim])
				copy(vh.data[dstOff:dstOff+headDim], v.data[srcOff:srcOff+headDim])
			}

			// Attention scores: Q @ K^T -> [seqLen, seqLen]
			scores := matMulTransB(qh, kh)
			scores.scale(scaleFactor)

			// Apply mask: -10000 for padding positions.
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					if mask[b][j] == 0 {
						scores.data[i*seqLen+j] += -10000.0
					}
				}
			}

			scores.softmax()

			// Weighted sum: scores @ V -> [seqLen, headDim]
			attnOut := matMul(scores, vh)

			// Copy back into output.
			for s := 0; s < seqLen; s++ {
				srcOff := s * headDim
				dstOff := (b*seqLen+s)*hidden + h*headDim
				copy(out.data[dstOff:dstOff+headDim], attnOut.data[srcOff:srcOff+headDim])
			}
		}
	}

	// Output projection: [totalTokens, hidden] @ [hidden, hidden]^T -> [totalTokens, hidden]
	result := matMulTransB(out, w.outW)
	addBias(result, w.outB)

	return result
}
