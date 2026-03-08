package goformer

import (
	"math"
	"testing"
)

func TestTensorNewAndReshape(t *testing.T) {
	tn := newTensor(2, 3)
	if len(tn.data) != 6 {
		t.Fatalf("expected 6 elements, got %d", len(tn.data))
	}
	r := tn.reshape(3, 2)
	if r.shape[0] != 3 || r.shape[1] != 2 {
		t.Fatalf("expected shape [3,2], got %v", r.shape)
	}
}

func TestTensorFrom(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	tn := tensorFrom(data, 2, 3)
	if tn.shape[0] != 2 || tn.shape[1] != 3 {
		t.Fatalf("expected shape [2,3], got %v", tn.shape)
	}
	row := tn.row(1)
	if row[0] != 4 || row[1] != 5 || row[2] != 6 {
		t.Fatalf("expected row [4,5,6], got %v", row)
	}
}

func TestTensorAdd(t *testing.T) {
	a := tensorFrom([]float32{1, 2, 3}, 3)
	b := tensorFrom([]float32{4, 5, 6}, 3)
	a.add(b)
	expected := []float32{5, 7, 9}
	for i, v := range a.data {
		if v != expected[i] {
			t.Fatalf("expected %v, got %v", expected, a.data)
		}
	}
}

func TestTensorSoftmax(t *testing.T) {
	tn := tensorFrom([]float32{1, 2, 3, 1, 2, 3}, 2, 3)
	tn.softmax()
	// Each row should sum to 1.
	for r := 0; r < 2; r++ {
		var sum float32
		for c := 0; c < 3; c++ {
			sum += tn.data[r*3+c]
		}
		if math.Abs(float64(sum-1.0)) > 1e-6 {
			t.Fatalf("row %d sum = %f, expected 1.0", r, sum)
		}
	}
}

func TestTensorGELU(t *testing.T) {
	tn := tensorFrom([]float32{0, 1, -1}, 3)
	tn.gelu()
	// GELU(0) ≈ 0, GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1588
	if math.Abs(float64(tn.data[0])) > 1e-5 {
		t.Errorf("GELU(0) = %f, expected ~0", tn.data[0])
	}
	if math.Abs(float64(tn.data[1])-0.8412) > 1e-3 {
		t.Errorf("GELU(1) = %f, expected ~0.8412", tn.data[1])
	}
	if math.Abs(float64(tn.data[2])+0.1588) > 1e-3 {
		t.Errorf("GELU(-1) = %f, expected ~-0.1588", tn.data[2])
	}
}

func TestMatMul(t *testing.T) {
	// [2,3] @ [3,2] -> [2,2]
	a := tensorFrom([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := tensorFrom([]float32{7, 8, 9, 10, 11, 12}, 3, 2)
	c := matMul(a, b)
	// Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
	// Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
	expected := []float32{58, 64, 139, 154}
	for i, v := range c.data {
		if v != expected[i] {
			t.Fatalf("expected %v, got %v", expected, c.data)
		}
	}
}

func TestMatMulTransB(t *testing.T) {
	// A[2,3] @ B^T where B is [2,3] -> [2,2]
	a := tensorFrom([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := tensorFrom([]float32{7, 8, 9, 10, 11, 12}, 2, 3)
	c := matMulTransB(a, b)
	// Row 0: [1*7+2*8+3*9, 1*10+2*11+3*12] = [50, 68]
	// Row 1: [4*7+5*8+6*9, 4*10+5*11+6*12] = [122, 167]
	expected := []float32{50, 68, 122, 167}
	for i, v := range c.data {
		if v != expected[i] {
			t.Fatalf("expected %v, got %v", expected, c.data)
		}
	}
}

func TestAddBias(t *testing.T) {
	x := tensorFrom([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := tensorFrom([]float32{10, 20, 30}, 3)
	addBias(x, b)
	expected := []float32{11, 22, 33, 14, 25, 36}
	for i, v := range x.data {
		if v != expected[i] {
			t.Fatalf("expected %v, got %v", expected, x.data)
		}
	}
}

func TestLayerNorm(t *testing.T) {
	x := tensorFrom([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	gamma := tensorFrom([]float32{1, 1, 1}, 3)
	beta := tensorFrom([]float32{0, 0, 0}, 3)
	layerNorm(x, gamma, beta, 1e-12)
	// Each row should be normalised to mean≈0 and variance≈1.
	for r := 0; r < 2; r++ {
		var mean float64
		for c := 0; c < 3; c++ {
			mean += float64(x.data[r*3+c])
		}
		mean /= 3
		if math.Abs(mean) > 1e-5 {
			t.Errorf("row %d mean = %f, expected ~0", r, mean)
		}
	}
}

func TestMeanPool(t *testing.T) {
	// [1, 3, 2] tensor, mask = [1, 1, 0] (third token is padding).
	x := tensorFrom([]float32{1, 2, 3, 4, 0, 0}, 1, 3, 2)
	mask := [][]int{{1, 1, 0}}
	pooled := meanPool(x, mask)
	// Mean of [1,2] and [3,4] = [2, 3]
	if pooled.data[0] != 2 || pooled.data[1] != 3 {
		t.Fatalf("expected [2, 3], got %v", pooled.data)
	}
}

func TestL2Normalise(t *testing.T) {
	x := tensorFrom([]float32{3, 4}, 1, 2)
	l2Normalise(x)
	// ||[3,4]|| = 5, so normalised = [0.6, 0.8]
	if math.Abs(float64(x.data[0])-0.6) > 1e-6 || math.Abs(float64(x.data[1])-0.8) > 1e-6 {
		t.Fatalf("expected [0.6, 0.8], got %v", x.data)
	}
}

func TestFloat16ToFloat32(t *testing.T) {
	// 1.0 in F16 = 0x3C00
	got := float16ToFloat32(0x3C00)
	if got != 1.0 {
		t.Errorf("F16(0x3C00) = %f, expected 1.0", got)
	}
	// -2.0 in F16 = 0xC000
	got = float16ToFloat32(0xC000)
	if got != -2.0 {
		t.Errorf("F16(0xC000) = %f, expected -2.0", got)
	}
	// 0.0
	got = float16ToFloat32(0x0000)
	if got != 0.0 {
		t.Errorf("F16(0x0000) = %f, expected 0.0", got)
	}
}
