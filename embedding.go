package goformer

// embedLookup computes token + position + type embeddings and applies LayerNorm.
// tokenIDs is [batch, seqLen], mask is [batch, seqLen].
func embedLookup(tokenIDs [][]int, wordEmbed, posEmbed, typeEmbed *tensor, lnGamma, lnBeta *tensor, eps float32) *tensor {
	batch := len(tokenIDs)
	seqLen := len(tokenIDs[0])
	hidden := wordEmbed.shape[1]

	out := newTensor(batch*seqLen, hidden)

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			tid := tokenIDs[b][s]
			outOff := (b*seqLen + s) * hidden
			wOff := tid * hidden
			pOff := s * hidden
			// type_id is always 0 for single-segment models.
			tOff := 0

			for h := 0; h < hidden; h++ {
				out.data[outOff+h] = wordEmbed.data[wOff+h] + posEmbed.data[pOff+h] + typeEmbed.data[tOff+h]
			}
		}
	}

	layerNorm(out, lnGamma, lnBeta, eps)
	return out.reshape(batch, seqLen, hidden)
}
