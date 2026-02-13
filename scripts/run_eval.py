#!/usr/bin/env python3
"""Batch evaluation CLI — run the DeALOG pipeline on dev/test splits.

Usage:
    python scripts/run_eval.py --split dev --max_examples 20 --workers 8
    python scripts/run_eval.py --split dev --max_examples 100 --workers 12
    python scripts/run_eval.py --split dev --workers 16           # all 883
    python scripts/run_eval.py --split dev --resume               # resume interrupted run
    python scripts/run_eval.py --split dev --max_examples 20 --llm-eval  # with LLM failure evaluation

LLM evaluation (--llm-eval):
    When enabled, failed predictions are evaluated by gpt-4o which:
    1. Judges whether the prediction is actually correct (alternate valid approach)
    2. Classifies the failure reason (wrong_number, wrong_computation, sign_error, etc.)
    Results are stored as llm_correct, failure_reason, llm_explanation in MongoDB.
"""

import argparse
import json
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

from finqa_chatbot.config import get_settings
from finqa_chatbot.pipeline import run_batch, load_dataset
from finqa_chatbot.evaluation.official import evaluate_result


def main():
    parser = argparse.ArgumentParser(description="FinQA DeALOG batch evaluation")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for predictions JSON")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Use LLM judge for prog_acc when exe passes but structural match fails")
    parser.add_argument("--llm-eval", action="store_true",
                        help="Run LLM evaluation on failed predictions (classifies failure reasons)")
    parser.add_argument("--start", type=int, default=None,
                        help="Start index (inclusive) for dataset slice")
    parser.add_argument("--end", type=int, default=None,
                        help="End index (exclusive) for dataset slice")
    args = parser.parse_args()

    settings = get_settings()
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename tag
    if args.start is not None or args.end is not None:
        n_tag = f"_{args.start or 0}_{args.end or 'end'}"
    elif args.max_examples:
        n_tag = f"_{args.max_examples}"
    else:
        n_tag = ""
    out_file = args.output or str(output_dir / f"predictions_{args.split}{n_tag}.json")

    # Load and slice dataset
    gold_data = load_dataset(args.split)
    if args.start is not None or args.end is not None:
        gold_data = gold_data[args.start:args.end]
    elif args.max_examples:
        gold_data = gold_data[:args.max_examples]

    print(f"Model:      {settings.model_name}")
    print(f"Max rounds: {settings.max_rounds}")
    print(f"Candidates: {settings.num_candidates}")
    print(f"Workers:    {args.workers}")
    print(f"Examples:   {len(gold_data)}")
    print(f"Output:     {out_file}")
    print()

    # If not resuming, remove existing file
    if not args.resume and Path(out_file).exists():
        Path(out_file).unlink()

    start = time.time()

    predictions, run_id = run_batch(
        split=args.split,
        workers=args.workers,
        save_path=out_file,
        save_every=10,
        data=gold_data,
        llm_eval=args.llm_eval,
    )

    total_time = time.time() - start
    print(f"\nCompleted in {total_time:.0f}s ({total_time/len(predictions):.1f}s avg)")
    if run_id:
        print(f"MongoDB run: {run_id}")

    # Save final predictions
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"Predictions saved to {out_file}")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    official = evaluate_result(predictions, gold_data, use_llm_judge=args.llm_judge)
    for k, v in official.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # LLM evaluation summary
    if args.llm_eval:
        llm_evaluated = [p for p in predictions if p.get("llm_correct") is not None]
        llm_correct = [p for p in llm_evaluated if p.get("llm_correct")]
        total = len(predictions)

        # Count failure reasons
        reason_counts: dict[str, int] = {}
        for p in llm_evaluated:
            reason = p.get("failure_reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        print("\n" + "=" * 60)
        print("LLM EVALUATION (gpt-4o)")
        print("=" * 60)
        print(f"  evaluated:        {len(llm_evaluated)}")
        print(f"  llm_correct:      {len(llm_correct)}")
        adj_exe = official["exe_correct"] + len(llm_correct)
        print(f"  adjusted_exe_acc: {adj_exe / total:.4f} ({adj_exe}/{total})")
        print(f"  failure breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

        # Add to official results for saving
        official["llm_evaluated"] = len(llm_evaluated)
        official["llm_correct"] = len(llm_correct)
        official["adjusted_exe_acc"] = round(adj_exe / total, 4)
        official["failure_reasons"] = reason_counts

    # Save results to MongoDB
    if run_id:
        from finqa_chatbot.storage import get_mongo_store
        store = get_mongo_store()
        if store:
            store.finish_run(run_id, official, extended, total_time)
            print(f"\nMongoDB run finalized: {run_id}")
            store.close()

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
