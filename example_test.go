package goformer_test

import (
	"fmt"
	"log"

	"github.com/MichaelAyles/goformer"
)

func Example() {
	model, err := goformer.Load("./bge-small-en-v1.5")
	if err != nil {
		log.Fatal(err)
	}

	embedding, err := model.Embed("DMA channel configuration")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("dims: %d\n", len(embedding))
}

func Example_batch() {
	model, err := goformer.Load("./bge-small-en-v1.5")
	if err != nil {
		log.Fatal(err)
	}

	embeddings, err := model.EmbedBatch([]string{
		"What is a DMA controller?",
		"How does SPI communication work?",
		"Configure the UART baud rate",
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("batch size: %d, dims: %d\n", len(embeddings), len(embeddings[0]))
}

func Example_similarity() {
	model, err := goformer.Load("./bge-small-en-v1.5")
	if err != nil {
		log.Fatal(err)
	}

	a, _ := model.Embed("What is a DMA controller?")
	b, _ := model.Embed("Direct memory access configuration")

	// Dot product of L2-normalised vectors equals cosine similarity.
	var similarity float32
	for i := range a {
		similarity += a[i] * b[i]
	}

	fmt.Printf("similarity: %.4f\n", similarity)
}
