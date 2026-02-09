#!/usr/bin/env python3
"""Run 20 diverse dev examples through the full live pipeline."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

# Disable LangSmith tracing to avoid warnings
# LangSmith tracing controlled by .env (LANGCHAIN_TRACING_V2)

from finqa_chatbot.config import get_settings
from finqa_chatbot.pipeline import run_single, load_dataset
from finqa_chatbot.dsl.parser import parse_program_to_tokens
from finqa_chatbot.dsl.executor import eval_program
from finqa_chatbot.evaluation.official import program_tokenization

settings = get_settings()
print(f"Model: {settings.model_name}")
print(f"Candidates: {settings.num_candidates}")
print(f"Max rounds: {settings.max_rounds}")
print()

data = load_dataset("dev")

# Pick 20 diverse examples: spread across different op types and complexities
INDICES = [0, 3, 5, 7, 10, 14, 18, 20, 24, 28, 30, 32, 36, 38, 40, 43, 45, 47, 48, 49]

entries = [data[i] for i in INDICES]
print(f"Running {len(entries)} examples through the full DeALOG pipeline...\n")
print(f"{'#':>3}  {'ID':<45} {'Gold Program':<55} {'Gold Ans':>10}")
print("-" * 118)
for i, e in zip(INDICES, entries):
    print(f"{i:3d}  {e['id'][:44]:<45} {e['qa']['program'][:54]:<55} {str(e['qa']['exe_ans']):>10}")
print()

# Run each example
results = []
total_start = time.time()

for idx, (dev_idx, entry) in enumerate(zip(INDICES, entries)):
    start = time.time()
    print(f"[{idx+1:2d}/20] {entry['id'][:50]}...", end=" ", flush=True)

    result = run_single(entry)
    elapsed = time.time() - start

    # Evaluate
    gold_ans = entry["qa"]["exe_ans"]
    exe_res = result.get("exe_result", "n/a")
    correct = exe_res == gold_ans

    # Also check program accuracy
    gold_tokens = program_tokenization(entry["qa"]["program"])
    pred_tokens = result.get("predicted", ["EOF"])
    prog_match = "".join(pred_tokens) == "".join(gold_tokens)

    result["dev_index"] = dev_idx
    result["gold_answer"] = gold_ans
    result["gold_program"] = entry["qa"]["program"]
    result["correct"] = correct
    result["prog_match"] = prog_match
    result["elapsed"] = round(elapsed, 1)
    results.append(result)

    status = "OK" if correct else "WRONG"
    print(f"{elapsed:5.1f}s  {status:5s}  prog={result.get('raw_program', '')[:60]}")

# Summary
total_elapsed = time.time() - total_start
exe_correct = sum(1 for r in results if r["correct"])
prog_correct = sum(1 for r in results if r["prog_match"])
invalid = sum(1 for r in results if r.get("exe_result") == "n/a" or r.get("predicted") == ["EOF"])
errors = [r for r in results if r.get("error")]
rounds_used = [r.get("rounds_used", 1) for r in results]

print()
print("=" * 70)
print("RESULTS SUMMARY — 20 Dev Examples")
print("=" * 70)
print(f"  Execution accuracy:  {exe_correct}/20 ({exe_correct/20:.1%})")
print(f"  Program accuracy:    {prog_correct}/20 ({prog_correct/20:.1%})")
print(f"  Invalid/errors:      {invalid + len(errors)}/20")
print(f"  Avg rounds used:     {sum(rounds_used)/len(rounds_used):.1f}")
print(f"  Total time:          {total_elapsed:.1f}s ({total_elapsed/20:.1f}s avg)")
print()

# Per-example detail
print(f"{'#':>3}  {'Correct':>7}  {'Rounds':>6}  {'Time':>5}  {'Gold Ans':>12}  {'Pred Ans':>12}  Program")
print("-" * 110)
for r in results:
    mark = "  OK " if r["correct"] else "WRONG"
    pred = str(r.get("exe_result", "n/a"))[:12]
    gold = str(r["gold_answer"])[:12]
    prog = r.get("raw_program", "")[:50]
    print(f"{r['dev_index']:3d}  {mark:>7}  {r.get('rounds_used',0):6d}  {r['elapsed']:5.1f}  {gold:>12}  {pred:>12}  {prog}")

# Save
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
with open(output_dir / "results_20_examples.json", "w") as f:
    # Convert non-serializable items
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items()}
        # predicted tokens list is fine
        serializable.append(sr)
    json.dump(serializable, f, indent=2, default=str)
print(f"\nDetailed results saved to {output_dir / 'results_20_examples.json'}")
