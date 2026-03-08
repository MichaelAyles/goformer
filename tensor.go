package goformer

import "math"

// Tensor is a dense row-major float32 tensor.
type tensor struct {
	data  []float32
	shape []int
}

func newTensor(shape ...int) *tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	return &tensor{data: make([]float32, size), shape: shape}
}

func tensorFrom(data []float32, shape ...int) *tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	if len(data) != size {
		panic("tensor: data length does not match shape")
	}
	return &tensor{data: data, shape: shape}
}

func (t *tensor) reshape(shape ...int) *tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	if size != len(t.data) {
		panic("tensor: reshape size mismatch")
	}
	s := make([]int, len(shape))
	copy(s, shape)
	return &tensor{data: t.data, shape: s}
}

// row returns the i-th row of a 2D tensor as a slice into the underlying data.
func (t *tensor) row(i int) []float32 {
	cols := t.shape[len(t.shape)-1]
	start := i * cols
	return t.data[start : start+cols]
}

// add performs element-wise addition: t += other.
func (t *tensor) add(other *tensor) {
	for i := range t.data {
		t.data[i] += other.data[i]
	}
}

// scale multiplies every element by a scalar.
func (t *tensor) scale(s float32) {
	for i := range t.data {
		t.data[i] *= s
	}
}

// softmax applies softmax along the last axis (each row independently).
func (t *tensor) softmax() {
	cols := t.shape[len(t.shape)-1]
	rows := len(t.data) / cols
	for r := 0; r < rows; r++ {
		off := r * cols
		row := t.data[off : off+cols]
		// Find max for numerical stability.
		max := row[0]
		for _, v := range row[1:] {
			if v > max {
				max = v
			}
		}
		// Exponentiate and sum.
		var sum float32
		for i, v := range row {
			e := float32(math.Exp(float64(v - max)))
			row[i] = e
			sum += e
		}
		// Normalise.
		inv := 1.0 / sum
		for i := range row {
			row[i] *= inv
		}
	}
}

// gelu applies the GELU activation (tanh approximation) in-place.
func (t *tensor) gelu() {
	c := float32(math.Sqrt(2.0 / math.Pi))
	for i, x := range t.data {
		t.data[i] = 0.5 * x * (1.0 + float32(math.Tanh(float64(c*(x+0.044715*x*x*x)))))
	}
}
