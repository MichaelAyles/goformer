package goformer

// matMul computes C = A @ B where A is [M,K] and B is [K,N].
func matMul(a, b *tensor) *tensor {
	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[1]
	c := newTensor(m, n)

	const tile = 32
	for i0 := 0; i0 < m; i0 += tile {
		iEnd := i0 + tile
		if iEnd > m {
			iEnd = m
		}
		for j0 := 0; j0 < n; j0 += tile {
			jEnd := j0 + tile
			if jEnd > n {
				jEnd = n
			}
			for p0 := 0; p0 < k; p0 += tile {
				pEnd := p0 + tile
				if pEnd > k {
					pEnd = k
				}
				for i := i0; i < iEnd; i++ {
					aRow := a.data[i*k : i*k+k]
					cRow := c.data[i*n : i*n+n]
					for p := p0; p < pEnd; p++ {
						ap := aRow[p]
						bRow := b.data[p*n : p*n+n]
						for j := j0; j < jEnd; j++ {
							cRow[j] += ap * bRow[j]
						}
					}
				}
			}
		}
	}
	return c
}

// matMulTransB computes C = A @ B^T where A is [M,K] and B is [N,K].
// Result is [M,N]. Avoids explicit transpose.
func matMulTransB(a, b *tensor) *tensor {
	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[0]
	c := newTensor(m, n)
	for i := 0; i < m; i++ {
		aRow := a.data[i*k : i*k+k]
		cRow := c.data[i*n : i*n+n]
		for j := 0; j < n; j++ {
			bRow := b.data[j*k : j*k+k]
			var sum float32
			for p := 0; p < k; p++ {
				sum += aRow[p] * bRow[p]
			}
			cRow[j] = sum
		}
	}
	return c
}

// addBias adds a 1D bias vector to every row of a 2D tensor in-place.
func addBias(t *tensor, bias *tensor) {
	cols := t.shape[len(t.shape)-1]
	rows := len(t.data) / cols
	b := bias.data
	for i := 0; i < rows; i++ {
		row := t.data[i*cols : i*cols+cols]
		for j := range row {
			row[j] += b[j]
		}
	}
}
