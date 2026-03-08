"""Benchmark comparison: PyTorch vs ONNX Runtime vs goformer (reported separately)."""

import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "models/bge-small-en-v1.5"

TEST_INPUTS = {
    "short": "DMA channel configuration",
    "medium": "The quick brown fox jumps over the lazy dog",
    "long": "This is a longer paragraph that contains multiple sentences. It should test the model's ability to handle longer sequences with various punctuation marks, including commas, periods, and exclamation points!",
}

BATCH_8 = [
    "DMA channel configuration",
    "The quick brown fox jumps over the lazy dog",
    "Hello",
    "Configure the UART baud rate",
    "How does SPI communication work",
    "Memory mapped I/O registers",
    "Interrupt service routine",
    "Clock tree configuration",
]

WARMUP = 5
ITERATIONS = 50


def bench_pytorch():
    print("=== PyTorch (CPU) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()

    for label, text in TEST_INPUTS.items():
        # Warmup.
        for _ in range(WARMUP):
            encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**encoded)

        # Timed.
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
                torch.nn.functional.normalize(pooled, p=2, dim=1)
        elapsed = (time.perf_counter() - start) / ITERATIONS
        print(f"  {label:8s}: {elapsed*1000:.1f}ms")

    # Batch of 8.
    for _ in range(WARMUP):
        encoded = tokenizer(BATCH_8, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            model(**encoded)

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        encoded = tokenizer(BATCH_8, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            torch.nn.functional.normalize(pooled, p=2, dim=1)
    elapsed = (time.perf_counter() - start) / ITERATIONS
    print(f"  {'batch_8':8s}: {elapsed*1000:.1f}ms")


def bench_onnx():
    try:
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForFeatureExtraction
    except ImportError:
        print("\n=== ONNX Runtime ===")
        print("  (skipped — optimum[onnxruntime] not installed)")
        return

    print("\n=== ONNX Runtime (CPU) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Export to ONNX if not already done.
    model = ORTModelForFeatureExtraction.from_pretrained(MODEL_PATH, export=True)

    for label, text in TEST_INPUTS.items():
        # Warmup.
        for _ in range(WARMUP):
            encoded = tokenizer(text, return_tensors="np", padding=True, truncation=True)
            outputs = model(**{k: torch.from_numpy(v) for k, v in encoded.items()})

        # Timed.
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            encoded = tokenizer(text, return_tensors="np", padding=True, truncation=True)
            outputs = model(**{k: torch.from_numpy(v) for k, v in encoded.items()})
        elapsed = (time.perf_counter() - start) / ITERATIONS
        print(f"  {label:8s}: {elapsed*1000:.1f}ms")

    # Batch of 8.
    for _ in range(WARMUP):
        encoded = tokenizer(BATCH_8, return_tensors="np", padding=True, truncation=True)
        model(**{k: torch.from_numpy(v) for k, v in encoded.items()})

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        encoded = tokenizer(BATCH_8, return_tensors="np", padding=True, truncation=True)
        model(**{k: torch.from_numpy(v) for k, v in encoded.items()})
    elapsed = (time.perf_counter() - start) / ITERATIONS
    print(f"  {'batch_8':8s}: {elapsed*1000:.1f}ms")


if __name__ == "__main__":
    bench_pytorch()
    bench_onnx()
