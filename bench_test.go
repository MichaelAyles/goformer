package goformer

import (
	"os"
	"testing"
)

func benchModel(b *testing.B) *Model {
	b.Helper()
	if _, err := os.Stat("models/bge-small-en-v1.5"); err != nil {
		b.Skip("model not found (download bge-small-en-v1.5)")
	}
	m, err := Load("models/bge-small-en-v1.5")
	if err != nil {
		b.Fatal(err)
	}
	return m
}

func BenchmarkModelLoad(b *testing.B) {
	if _, err := os.Stat("models/bge-small-en-v1.5"); err != nil {
		b.Skip("model not found")
	}
	for i := 0; i < b.N; i++ {
		_, err := Load("models/bge-small-en-v1.5")
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEmbed_Short(b *testing.B) {
	m := benchModel(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := m.Embed("DMA channel configuration")
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEmbed_Medium(b *testing.B) {
	m := benchModel(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := m.Embed("The quick brown fox jumps over the lazy dog")
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEmbed_Long(b *testing.B) {
	m := benchModel(b)
	text := "This is a longer paragraph that contains multiple sentences. It should test the model's ability to handle longer sequences with various punctuation marks, including commas, periods, and exclamation points!"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := m.Embed(text)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEmbedBatch_8(b *testing.B) {
	m := benchModel(b)
	texts := []string{
		"DMA channel configuration",
		"The quick brown fox jumps over the lazy dog",
		"Hello",
		"Configure the UART baud rate",
		"How does SPI communication work",
		"Memory mapped I/O registers",
		"Interrupt service routine",
		"Clock tree configuration",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := m.EmbedBatch(texts)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMul_384x384(b *testing.B) {
	a := newTensor(384, 384)
	bm := newTensor(384, 384)
	for i := range a.data {
		a.data[i] = 0.01
	}
	for i := range bm.data {
		bm.data[i] = 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matMul(a, bm)
	}
}

func BenchmarkMatMul_384x1536(b *testing.B) {
	a := newTensor(384, 384)
	bm := newTensor(384, 1536)
	for i := range a.data {
		a.data[i] = 0.01
	}
	for i := range bm.data {
		bm.data[i] = 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matMul(a, bm)
	}
}

func BenchmarkLayerNorm(b *testing.B) {
	x := newTensor(128, 384)
	gamma := newTensor(384)
	beta := newTensor(384)
	for i := range x.data {
		x.data[i] = 0.1
	}
	for i := range gamma.data {
		gamma.data[i] = 1.0
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layerNorm(x, gamma, beta, 1e-12)
	}
}

func BenchmarkSoftmax(b *testing.B) {
	x := newTensor(12, 128, 128)
	for i := range x.data {
		x.data[i] = 0.01 * float32(i%100)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.softmax()
	}
}

func BenchmarkGELU(b *testing.B) {
	x := newTensor(128, 1536)
	for i := range x.data {
		x.data[i] = 0.01 * float32(i%100)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.gelu()
	}
}

func BenchmarkTokenise(b *testing.B) {
	m := benchModel(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.tok.tokenise("The quick brown fox jumps over the lazy dog")
	}
}
