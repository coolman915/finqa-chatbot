#!/usr/bin/env python3
"""Pre-compute embeddings to .npy for faster retrieval."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finqa_chatbot.config import get_settings
from finqa_chatbot.retrieval.facts import get_all_facts
from finqa_chatbot.retrieval.embeddings import openai_embed


def main():
    parser = argparse.ArgumentParser(description="Build embeddings cache")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    settings = get_settings()
    data_path = settings.dataset_dir / f"{args.split}.json"
    with open(data_path) as f:
        data = json.load(f)

    print(f"Processing {len(data)} entries from {args.split}...")

    all_texts: list[str] = []
    text_to_idx: dict[str, int] = {}

    for entry in data:
        question = entry["qa"]["question"]
        if question not in text_to_idx:
            text_to_idx[question] = len(all_texts)
            all_texts.append(question)
        facts = get_all_facts(entry)
        for _, text in facts:
            if text not in text_to_idx:
                text_to_idx[text] = len(all_texts)
                all_texts.append(text)

    print(f"Total unique texts: {len(all_texts)}")

    # Embed in batches
    embeddings: list[list[float]] = []
    for i in range(0, len(all_texts), args.batch_size):
        batch = all_texts[i:i + args.batch_size]
        batch_embs = openai_embed(batch)
        embeddings.extend(batch_embs)
        print(f"  Embedded {min(i + args.batch_size, len(all_texts))}/{len(all_texts)}")

    # Save
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / f"embeddings_{args.split}.npy"
    texts_path = output_dir / f"embedding_texts_{args.split}.json"

    np.save(str(emb_path), np.array(embeddings))
    with open(texts_path, "w") as f:
        json.dump(all_texts, f)

    print(f"Saved embeddings to {emb_path}")
    print(f"Saved text index to {texts_path}")


if __name__ == "__main__":
    main()
