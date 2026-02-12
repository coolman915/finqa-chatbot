#!/usr/bin/env python3
"""Interactive single-example demo CLI."""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure correct .env is loaded before any other imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

from finqa_chatbot.config import get_settings
from finqa_chatbot.pipeline import load_dataset, run_single
from finqa_chatbot.evaluation.official import _relaxed_equal, relaxed_equal_program
from finqa_chatbot.dsl.parser import parse_program_to_tokens
from finqa_chatbot.schema import EntryType
from finqa_chatbot.storage import get_mongo_store


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
    print(f"Max rounds: {settings.max_rounds}")
    print(f"API key: ...{settings.openai_api_key[-10:]}")
    print("-" * 60)

    # MongoDB tracing
    store = get_mongo_store()
    run_id = None
    if store:
        from datetime import datetime, timezone
        run_id = f"debug_{entry['id'].replace('/', '_')}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
        store.create_run(run_id=run_id, split=args.split, num_examples=1, config={
            "model": settings.model_name, "max_rounds": settings.max_rounds,
        })
        print(f"Tracing to MongoDB: {run_id}")

    result = run_single(entry, run_id=run_id, store=store)

    if store and run_id:
        print(f"\nView trace:  python scripts/query_results.py trace {run_id} \"{entry['id']}\"")
        store.close()

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Predicted program: {result['raw_program']}")
    print(f"Execution result:  {result.get('exe_result', 'n/a')}")
    print(f"Rounds used:       {result.get('rounds_used', '?')}")
    print(f"Verification:      {result.get('verification_status', '?')}")
    print(f"Final answer:      {result.get('final_answer', 'n/a')}")

    gold_ans = entry["qa"]["exe_ans"]
    text_answer = entry["qa"].get("answer", "")
    exe_result = result.get("exe_result", "n/a")
    exact = str(exe_result) == str(gold_ans)
    relaxed = _relaxed_equal(exe_result, gold_ans, answer=text_answer)

    # Program accuracy
    pred_program = result.get("raw_program", "")
    gold_program = entry["qa"]["program"]
    pred_tokens = parse_program_to_tokens(pred_program) if pred_program else ["EOF"]
    gold_tokens = parse_program_to_tokens(gold_program) if gold_program else ["EOF"]
    prog_match = relaxed_equal_program(gold_tokens, pred_tokens)

    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    # exe_acc
    if exact:
        print(f"exe_acc:  YES")
    elif relaxed:
        print(f"exe_acc:  YES (relaxed — scale/sign tolerance)")
        print(f"  Gold: {gold_ans}  |  Got: {exe_result}")
    else:
        print(f"exe_acc:  NO")
        print(f"  Expected: {gold_ans}")
        print(f"  Got:      {exe_result}")
    # prog_acc
    print(f"prog_acc: {'YES' if prog_match else 'NO'}")
    if not prog_match:
        print(f"  Gold prog: {gold_program}")
        print(f"  Pred prog: {pred_program}")


if __name__ == "__main__":
    main()
