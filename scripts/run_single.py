#!/usr/bin/env python3
"""Interactive single-example demo CLI."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finqa_chatbot.config import get_settings
from finqa_chatbot.pipeline import load_dataset, run_single
from finqa_chatbot.schema import EntryType


def main():
    parser = argparse.ArgumentParser(description="FinQA DeALOG single example demo")
    parser.add_argument("--entry_id", type=str, default=None,
                        help="Specific entry ID to run")
    parser.add_argument("--index", type=int, default=0,
                        help="Index in dataset (if no entry_id)")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    data = load_dataset(args.split)
    settings = get_settings()

    # Find the entry
    if args.entry_id:
        entry = next((e for e in data if e["id"] == args.entry_id), None)
        if not entry:
            print(f"Entry {args.entry_id} not found in {args.split} split")
            sys.exit(1)
    else:
        entry = data[args.index]

    print(f"Entry ID: {entry['id']}")
    print(f"Question: {entry['qa']['question']}")
    print(f"Gold program: {entry['qa']['program']}")
    print(f"Gold answer: {entry['qa']['exe_ans']}")
    print(f"Model: {settings.model_name}")
    print("-" * 60)

    result = run_single(entry)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Predicted program: {result['raw_program']}")
    print(f"Execution result:  {result.get('exe_result', 'n/a')}")
    print(f"Rounds used:       {result.get('rounds_used', '?')}")
    print(f"Verification:      {result.get('verification_status', '?')}")
    print(f"Final answer:      {result.get('final_answer', 'n/a')}")

    gold_ans = entry["qa"]["exe_ans"]
    correct = result.get("exe_result") == gold_ans
    print(f"\nCorrect: {'YES' if correct else 'NO'}")
    if not correct:
        print(f"  Expected: {gold_ans}")
        print(f"  Got:      {result.get('exe_result', 'n/a')}")


if __name__ == "__main__":
    main()
