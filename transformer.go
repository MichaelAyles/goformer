package goformer

// transformerLayer holds the weights for a single transformer encoder layer.
type transformerLayer struct {
	attn       *attentionWeights
	attnLNGamma, attnLNBeta *tensor // LayerNorm after attention
	ffnW1, ffnB1            *tensor // intermediate dense
	ffnW2, ffnB2            *tensor // output dense
	ffnLNGamma, ffnLNBeta   *tensor // LayerNorm after FFN
}

// forward runs one transformer layer: attention + residual + LN + FFN + residual + LN.
// x is [batch*seqLen, hidden], mask is [batch, seqLen].
func (layer *transformerLayer) forward(x *tensor, mask [][]int, numHeads int, eps float32) *tensor {
	// Self-attention.
	attnOut := selfAttention(x, mask, numHeads, layer.attn)

	// Residual connection + LayerNorm.
	attnOut.add(x)
	layerNorm(attnOut, layer.attnLNGamma, layer.attnLNBeta, eps)

	// Feed-forward network.
	ffnOut := feedForward(
		attnOut.reshape(attnOut.shape[0], attnOut.shape[1]),
		layer.ffnW1, layer.ffnB1,
		layer.ffnW2, layer.ffnB2,
	)

	// Residual connection + LayerNorm.
	ffnOut.add(attnOut)
	layerNorm(ffnOut, layer.ffnLNGamma, layer.ffnLNBeta, eps)

	return ffnOut
}
