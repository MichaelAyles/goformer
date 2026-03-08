package goformer

import "math"

// meanPool computes the mean of hidden states over non-padding tokens per batch item.
// x is [batch, seqLen, hidden], mask is [batch, seqLen] (1=real, 0=pad).
// Returns [batch, hidden].
func meanPool(x *tensor, mask [][]int) *tensor {
	batch := x.shape[0]
	seqLen := x.shape[1]
	hidden := x.shape[2]

	out := newTensor(batch, hidden)

	for b := 0; b < batch; b++ {
		var count float32
		for s := 0; s < seqLen; s++ {
			if mask[b][s] == 0 {
				continue
			}
			count++
			off := (b*seqLen+s)*hidden
			outOff := b * hidden
			for h := 0; h < hidden; h++ {
				out.data[outOff+h] += x.data[off+h]
			}
		}
		if count > 0 {
			inv := 1.0 / count
			outOff := b * hidden
			for h := 0; h < hidden; h++ {
				out.data[outOff+h] *= inv
			}
		}
	}

	return out
}

// l2Normalise normalises each row to unit L2 norm.
// x is [batch, hidden].
func l2Normalise(x *tensor) {
	cols := x.shape[1]
	rows := x.shape[0]
	for r := 0; r < rows; r++ {
		row := x.data[r*cols : r*cols+cols]
		var sum float64
		for _, v := range row {
			sum += float64(v) * float64(v)
		}
		norm := float32(math.Sqrt(sum))
		if norm > 0 {
			inv := 1.0 / norm
			for i := range row {
				row[i] *= inv
			}
		}
	}
}
