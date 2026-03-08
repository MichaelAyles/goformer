"""Generate reference outputs from HuggingFace transformers for Go test validation."""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "models/bge-small-en-v1.5"

TEST_INPUTS = [
    "DMA channel configuration",
    "The quick brown fox jumps over the lazy dog",
    "Hello",
    "This is a longer paragraph that contains multiple sentences. It should test the model's ability to handle longer sequences with various punctuation marks, including commas, periods, and exclamation points!",
    "café résumé naïve",
    "Hello, world! How's it going?",
]

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()

    references = []

    for text in TEST_INPUTS:
        encoded = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
        token_ids = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()

        with torch.no_grad():
            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]

        # Mean pooling over non-padding tokens.
        mask_expanded = encoded["attention_mask"].unsqueeze(-1).float()
        sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1)
        mean_pooled = sum_hidden / count

        # L2 normalise.
        embedding = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

        references.append({
            "text": text,
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "embedding": embedding[0].tolist(),
        })

    with open("testdata/references.json", "w") as f:
        json.dump(references, f, indent=2)

    print(f"Generated {len(references)} reference outputs")

    # Quick sanity: cosine similarity between first two inputs.
    e0 = torch.tensor(references[0]["embedding"])
    e1 = torch.tensor(references[1]["embedding"])
    cos = torch.nn.functional.cosine_similarity(e0.unsqueeze(0), e1.unsqueeze(0))
    print(f"Cosine similarity (first two inputs): {cos.item():.6f}")

if __name__ == "__main__":
    main()
