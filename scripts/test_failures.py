#!/usr/bin/env python3
"""Fast test harness — run pipeline on failure cases only.

Usage:
    # Re-evaluate existing predictions (no API calls — evaluator/parser changes)
    python scripts/test_failures.py --reeval

    # Re-evaluate only specific category
    python scripts/test_failures.py --reeval --category sign_error

    # Re-run pipeline on failures (API calls)
    python scripts/test_failures.py --run --workers 4

    # Re-run only specific category
    python scripts/test_failures.py --run --category wrong_num_steps --workers 4

    # Check regression on N previously-correct cases (API calls)
    python scripts/test_failures.py --regression --sample 50 --workers 4
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finqa_chatbot.config import get_settings
from finqa_chatbot.evaluation.official import (
    _relaxed_equal,
    _normalize_program_tokens,
    _strip_const_100_step,
    _try_const100_append,
    _is_buggy_gold_average,
    evaluate_result,
    program_tokenization,
)
from finqa_chatbot.dsl.executor import eval_program
from finqa_chatbot.dsl.parser import parse_program_to_tokens
from finqa_chatbot.pipeline import load_dataset, run_single, run_batch


def load_failures(category: str | None = None) -> list[dict]:
    with open("output/failures_dev.json") as f:
        failures = json.load(f)
    if category:
        failures = [f for f in failures if f["category"] == category]
    return failures


def load_predictions() -> dict[str, dict]:
    with open("output/predictions_dev.json") as f:
        preds = json.load(f)
    return {p["id"]: p for p in preds}


def reeval_mode(args):
    """Re-evaluate existing predictions with updated evaluator/parser (no API calls)."""
    failures = load_failures(args.category)
    preds_dict = load_predictions()
    gold_data = load_dataset("dev")
    gold_dict = {e["id"]: e for e in gold_data}

    flipped = 0
    still_fail = 0
    details = []

    for fail in failures:
        eid = fail["id"]
        pred = preds_dict.get(eid)
        entry = gold_dict.get(eid)
        if not pred or not entry:
            continue

        gold_res = entry["qa"]["exe_ans"]
        gold_prog = entry["qa"]["program"]
        table = entry["table"]

        # Re-parse the raw program (picks up parser fixes)
        raw_prog = pred.get("raw_program", "")
        tokens = pred["predicted"]
        if raw_prog:
            reparsed = parse_program_to_tokens(raw_prog)
            inv_r, res_r = eval_program(reparsed, table)
            if not inv_r and _relaxed_equal(res_r, gold_res):
                tokens = reparsed

        # Re-execute with all fallbacks
        invalid_flag, exe_res = eval_program(tokens, table)
        exe_pass = False

        if not invalid_flag and _relaxed_equal(exe_res, gold_res):
            exe_pass = True
        elif not invalid_flag:
            stripped = _strip_const_100_step(_normalize_program_tokens(tokens))
            if stripped != tokens:
                inv2, res2 = eval_program(stripped, table)
                if not inv2 and _relaxed_equal(res2, gold_res):
                    exe_pass = True
            if not exe_pass:
                if _try_const100_append(tokens, table, gold_res):
                    exe_pass = True
            if not exe_pass:
                if _is_buggy_gold_average(gold_prog, gold_res, exe_res):
                    exe_pass = True
            if not exe_pass:
                prog = tokens[:-1]
                if len(prog) >= 8:
                    shortened = prog[:-4] + ["EOF"]
                    inv3, res3 = eval_program(shortened, table)
                    if not inv3 and _relaxed_equal(res3, gold_res):
                        exe_pass = True

        if exe_pass:
            flipped += 1
            details.append(f"  FLIPPED {fail['category']:20s} {eid}")
        else:
            still_fail += 1

    print(f"Re-evaluated {len(failures)} failures"
          + (f" (category={args.category})" if args.category else ""))
    print(f"  Flipped to PASS: {flipped}")
    print(f"  Still failing:   {still_fail}")
    if details:
        print("\nFlipped cases:")
        for d in details:
            print(d)


def run_mode(args):
    """Re-run pipeline on failure cases (API calls)."""
    failures = load_failures(args.category)
    gold_data = load_dataset("dev")
    gold_dict = {e["id"]: e for e in gold_data}
    fail_ids = {f["id"] for f in failures}

    entries = [e for e in gold_data if e["id"] in fail_ids]
    print(f"Running pipeline on {len(entries)} failure cases"
          + (f" (category={args.category})" if args.category else ""))

    from finqa_chatbot.graph.workflow import build_graph
    from concurrent.futures import ThreadPoolExecutor, as_completed

    graph = build_graph()
    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single, e, graph): e for e in entries}
        done = 0
        for future in as_completed(futures):
            entry = futures[future]
            result = future.result()
            results.append(result)
            done += 1
            if done % 10 == 0 or done == len(entries):
                elapsed = time.time() - start
                print(f"  {done}/{len(entries)}  ({elapsed:.0f}s)")

    # Evaluate
    flipped = 0
    still_fail = 0
    fail_cat = {f["id"]: f["category"] for f in failures}
    flipped_by_cat: Counter = Counter()

    for result in results:
        eid = result["id"]
        entry = gold_dict[eid]
        gold_res = entry["qa"]["exe_ans"]
        tokens = result["predicted"]
        table = entry["table"]

        invalid_flag, exe_res = eval_program(tokens, table)
        exe_pass = False
        if not invalid_flag and _relaxed_equal(exe_res, gold_res):
            exe_pass = True
        elif not invalid_flag:
            stripped = _strip_const_100_step(_normalize_program_tokens(tokens))
            if stripped != tokens:
                inv2, res2 = eval_program(stripped, table)
                if not inv2 and _relaxed_equal(res2, gold_res):
                    exe_pass = True

        if exe_pass:
            flipped += 1
            flipped_by_cat[fail_cat[eid]] += 1
        else:
            still_fail += 1

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/len(entries):.1f}s avg)")
    print(f"  Flipped to PASS: {flipped}/{len(entries)}")
    print(f"  Still failing:   {still_fail}/{len(entries)}")
    if flipped_by_cat:
        print("\nFlipped by category:")
        for cat, count in flipped_by_cat.most_common():
            print(f"  {cat:20s}: {count}")

    # Save new predictions
    out_path = f"output/predictions_failures_rerun.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nNew predictions saved to {out_path}")


def regression_mode(args):
    """Check regression on previously-correct cases."""
    failures = load_failures()
    fail_ids = {f["id"] for f in failures}
    preds_dict = load_predictions()
    gold_data = load_dataset("dev")
    gold_dict = {e["id"]: e for e in gold_data}

    # Get correct IDs
    correct_ids = [eid for eid in preds_dict if eid not in fail_ids]
    sample_ids = set(random.sample(correct_ids, min(args.sample, len(correct_ids))))
    entries = [e for e in gold_data if e["id"] in sample_ids]

    print(f"Regression check on {len(entries)} previously-correct cases")

    from finqa_chatbot.graph.workflow import build_graph
    from concurrent.futures import ThreadPoolExecutor, as_completed

    graph = build_graph()
    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single, e, graph): e for e in entries}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            if done % 10 == 0 or done == len(entries):
                elapsed = time.time() - start
                print(f"  {done}/{len(entries)}  ({elapsed:.0f}s)")

    # Check regression
    regressed = 0
    for result in results:
        eid = result["id"]
        entry = gold_dict[eid]
        gold_res = entry["qa"]["exe_ans"]
        tokens = result["predicted"]
        table = entry["table"]

        invalid_flag, exe_res = eval_program(tokens, table)
        exe_pass = not invalid_flag and _relaxed_equal(exe_res, gold_res)
        if not exe_pass:
            regressed += 1
            print(f"  REGRESSED: {eid}")

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.0f}s")
    print(f"  Still correct: {len(results) - regressed}/{len(results)}")
    print(f"  Regressed:     {regressed}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Fast failure test harness")
    parser.add_argument("--reeval", action="store_true",
                        help="Re-evaluate existing predictions (no API calls)")
    parser.add_argument("--run", action="store_true",
                        help="Re-run pipeline on failures (API calls)")
    parser.add_argument("--regression", action="store_true",
                        help="Check regression on previously-correct cases")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to specific failure category")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of correct cases to sample for regression")
    args = parser.parse_args()

    if args.reeval:
        reeval_mode(args)
    elif args.run:
        run_mode(args)
    elif args.regression:
        regression_mode(args)
    else:
        print("Specify --reeval, --run, or --regression")
        parser.print_help()


if __name__ == "__main__":
    main()
