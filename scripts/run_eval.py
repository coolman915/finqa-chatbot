#!/usr/bin/env python3
"""Batch evaluation CLI — run the DeALOG pipeline on dev/test splits."""

import argparse
import json
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finqa_chatbot.config import get_settings
from finqa_chatbot.pipeline import run_batch, load_dataset
from finqa_chatbot.evaluation.official import evaluate_result
from finqa_chatbot.evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="FinQA DeALOG batch evaluation")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for predictions JSON")
    args = parser.parse_args()

    settings = get_settings()
    print(f"Model: {settings.model_name}")
    print(f"Max rounds: {settings.max_rounds}")
    print(f"Candidates: {settings.num_candidates}")

    # Run pipeline
    predictions = run_batch(
        split=args.split,
        max_examples=args.max_examples,
        workers=args.workers,
    )

    # Save predictions
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output or str(output_dir / f"predictions_{args.split}.json")
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {out_file}")

    # Evaluate
    gold_data = load_dataset(args.split)

    print("\n" + "=" * 60)
    print("OFFICIAL EVALUATION")
    print("=" * 60)
    official = evaluate_result(predictions, gold_data)
    for k, v in official.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("EXTENDED METRICS")
    print("=" * 60)
    extended = compute_metrics(predictions, gold_data)
    for k, v in extended.items():
        print(f"  {k}: {v}")

    # Save results
    results_file = output_dir / f"results_{args.split}.json"
    with open(results_file, "w") as f:
        json.dump({"official": official, "extended": extended}, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
