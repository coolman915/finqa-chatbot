#!/usr/bin/env python3
"""Run 5 diverse dev examples through the full LIVE DeALOG pipeline."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))
# LangSmith tracing controlled by .env (LANGCHAIN_TRACING_V2)

from finqa_chatbot.config import get_settings
from finqa_chatbot.pipeline import load_dataset, run_single
from finqa_chatbot.dsl.parser import parse_program_to_tokens
from finqa_chatbot.dsl.executor import eval_program
from finqa_chatbot.evaluation.official import program_tokenization

settings = get_settings()
data = load_dataset("dev")

# 5 diverse examples: simple divide, subtract, multi-step, percent change, table_average
INDICES = [0, 5, 14, 18, 30]
LABELS = ["simple divide", "simple subtract", "multi-step add", "percent change", "table_average"]

print(f"Model: {settings.model_name}")
print(f"Candidates: {settings.num_candidates}  |  Max rounds: {settings.max_rounds}")
print(f"Running 5 diverse examples through the full live pipeline...\n")

results = []
total_start = time.time()

for i, (idx, label) in enumerate(zip(INDICES, LABELS)):
    entry = data[idx]
    gold_prog = entry["qa"]["program"]
    gold_ans = entry["qa"]["exe_ans"]

    print(f"{'='*75}")
    print(f"[{i+1}/5] {label.upper()} — {entry['id']}")
    print(f"{'='*75}")
    print(f"  Question:  {entry['qa']['question']}")
    print(f"  Gold:      {gold_prog}  →  {gold_ans}")
    print(f"  Table:     {len(entry['table'])} rows  |  Text: {len([s for s in entry.get('pre_text',[]) + entry.get('post_text',[]) if s.strip() and s.strip() != '.'])} passages")
    print(f"  Running...", end=" ", flush=True)

    t0 = time.time()
    result = run_single(entry)
    elapsed = time.time() - t0

    pred_prog = result.get("raw_program", "")
    pred_ans = result.get("exe_result", "n/a")
    correct = pred_ans == gold_ans
    rounds = result.get("rounds_used", 0)
    status = result.get("verification_status", "?")

    print(f"done in {elapsed:.1f}s")
    print(f"  Predicted: {pred_prog}  →  {pred_ans}")
    print(f"  Rounds:    {rounds}  |  Verification: {status}")
    print(f"  Result:    {'CORRECT' if correct else 'WRONG'}")
    if not correct:
        print(f"  Expected {gold_ans}, got {pred_ans}")
    print()

    results.append({
        "index": idx,
        "label": label,
        "id": entry["id"],
        "question": entry["qa"]["question"],
        "gold_program": gold_prog,
        "gold_answer": gold_ans,
        "pred_program": pred_prog,
        "pred_answer": pred_ans,
        "correct": correct,
        "rounds": rounds,
        "verification": status,
        "time": round(elapsed, 1),
    })

total_time = time.time() - total_start
correct_count = sum(1 for r in results if r["correct"])

print(f"{'='*75}")
print(f"SUMMARY")
print(f"{'='*75}")
print(f"  Correct: {correct_count}/5 ({correct_count/5:.0%})")
print(f"  Total time: {total_time:.1f}s  ({total_time/5:.1f}s avg)")
print()
print(f"  {'#':>3}  {'Label':<20} {'Correct':>7}  {'Rounds':>6}  {'Time':>6}  {'Verification':>12}  Program")
print(f"  {'-'*95}")
for r in results:
    mark = "  OK " if r["correct"] else "WRONG"
    print(f"  {r['index']:3d}  {r['label']:<20} {mark:>7}  {r['rounds']:>6}  {r['time']:>5.1f}s  {r['verification']:>12}  {r['pred_program'][:40]}")

# Save
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
with open(output_dir / "results_5_live.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {output_dir / 'results_5_live.json'}")
