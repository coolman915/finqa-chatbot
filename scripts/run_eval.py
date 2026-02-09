#!/usr/bin/env python3
"""Batch evaluation CLI — run the DeALOG pipeline on dev/test splits.

Usage:
    python scripts/run_eval.py --split dev --max_examples 20 --workers 8
    python scripts/run_eval.py --split dev --max_examples 100 --workers 12
    python scripts/run_eval.py --split dev --workers 16           # all 883
    python scripts/run_eval.py --split dev --resume               # resume interrupted run
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finqa_chatbot.config import get_settings
from finqa_chatbot.pipeline import run_batch, load_dataset
from finqa_chatbot.evaluation.official import evaluate_result
from finqa_chatbot.evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="FinQA DeALOG batch evaluation")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for predictions JSON")
    args = parser.parse_args()

    settings = get_settings()
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_tag = f"_{args.max_examples}" if args.max_examples else ""
    out_file = args.output or str(output_dir / f"predictions_{args.split}{n_tag}.json")

    print(f"Model:      {settings.model_name}")
    print(f"Max rounds: {settings.max_rounds}")
    print(f"Candidates: {settings.num_candidates}")
    print(f"Workers:    {args.workers}")
    print(f"Output:     {out_file}")
    print()

    # If not resuming, remove existing file
    if not args.resume and Path(out_file).exists():
        Path(out_file).unlink()

    start = time.time()

    predictions = run_batch(
        split=args.split,
        max_examples=args.max_examples,
        workers=args.workers,
        save_path=out_file,
        save_every=10,
    )

    total_time = time.time() - start
    print(f"\nCompleted in {total_time:.0f}s ({total_time/len(predictions):.1f}s avg)")

    # Save final predictions
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"Predictions saved to {out_file}")

    # Evaluate
    gold_data = load_dataset(args.split)
    if args.max_examples:
        gold_data = gold_data[:args.max_examples]

    print("\n" + "=" * 60)
    print("OFFICIAL EVALUATION")
    print("=" * 60)
    official = evaluate_result(predictions, gold_data)
    for k, v in official.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("EXTENDED METRICS")
    print("=" * 60)
    extended = compute_metrics(predictions, gold_data)
    for k, v in extended.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save results
    results_file = output_dir / f"results_{args.split}{n_tag}.json"
    with open(results_file, "w") as f:
        json.dump({
            "official": official,
            "extended": extended,
            "config": {
                "model": settings.model_name,
                "max_rounds": settings.max_rounds,
                "num_candidates": settings.num_candidates,
                "workers": args.workers,
                "total_time": round(total_time, 1),
                "avg_time": round(total_time / len(predictions), 1),
            },
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
