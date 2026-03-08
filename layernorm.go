package goformer

import "math"

// layerNorm applies layer normalisation over the last dimension.
// gamma and beta are the learned scale and bias parameters.
func layerNorm(t, gamma, beta *tensor, eps float32) {
	cols := t.shape[len(t.shape)-1]
	rows := len(t.data) / cols

	for r := 0; r < rows; r++ {
		row := t.data[r*cols : r*cols+cols]

		// Compute mean.
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(cols)

		// Compute variance.
		var variance float64
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(cols)

		// Normalise, scale, and shift.
		inv := float32(1.0 / math.Sqrt(variance+float64(eps)))
		m := float32(mean)
		g := gamma.data
		b := beta.data
		for i := range row {
			row[i] = (row[i]-m)*inv*g[i] + b[i]
		}
	}
}
