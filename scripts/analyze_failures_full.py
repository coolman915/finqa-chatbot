"""Comprehensive failure analysis for the full dev set."""
import sys, os
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

import json
from collections import Counter, defaultdict
from finqa_chatbot.evaluation.official import (
    _relaxed_equal, program_tokenization, relaxed_equal_program,
    _strip_const_100_step, _normalize_program_tokens,
)
from finqa_chatbot.dsl.executor import eval_program

with open("output/predictions_dev.json") as f:
    preds = json.load(f)
with open("../FinQA_paper/dataset/dev.json") as f:
    gold = json.load(f)

data_dict = {e["id"]: e for e in gold}

failures = []
for pred in preds:
    entry = data_dict[pred["id"]]
    gold_res = entry["qa"]["exe_ans"]
    gold_prog = entry["qa"]["program"]
    gold_tokens = program_tokenization(gold_prog)
    pred_tokens = pred["predicted"]

    text_answer = entry["qa"].get("answer", "")
    invalid_flag, exe_res = eval_program(pred_tokens, entry["table"])

    # Check exe with relaxed + stripped
    exe_ok = False
    if not invalid_flag and _relaxed_equal(exe_res, gold_res, answer=text_answer):
        exe_ok = True
    elif not invalid_flag:
        stripped = _strip_const_100_step(_normalize_program_tokens(pred_tokens))
        if stripped != pred_tokens:
            inv2, res2 = eval_program(stripped, entry["table"])
            if not inv2 and _relaxed_equal(res2, gold_res, answer=text_answer):
                exe_ok = True

    prog_ok = relaxed_equal_program(gold_tokens, pred_tokens)

    if not exe_ok:
        # Categorize the failure
        category = "unknown"
        ratio = None
        try:
            p, g = float(exe_res if not invalid_flag else 0), float(gold_res)
            if g != 0:
                ratio = p / g
        except:
            pass

        if invalid_flag:
            category = "invalid_program"
        elif ratio is not None:
            abs_ratio = abs(ratio)
            if abs(ratio - 1) < 0.05:
                category = "near_miss_5pct"
            elif abs(ratio + 1) < 0.05:
                category = "sign_error"
            elif abs(abs_ratio - 10) < 1 or abs(abs_ratio - 0.1) < 0.01:
                category = "scale_10x"
            elif abs(abs_ratio - 100) < 5 or abs(abs_ratio - 0.01) < 0.005:
                category = "scale_100x"
            elif abs(abs_ratio - 1000) < 50 or abs(abs_ratio - 0.001) < 0.0005:
                category = "scale_1000x"
            else:
                # Check if it's a simple wrong value vs wrong approach
                pred_raw = pred.get("raw_program", "")
                gold_ops = [op for op in gold_prog.replace("(", " ").replace(")", " ").replace(",", " ").split()
                           if op in ("add", "subtract", "multiply", "divide", "exp", "greater",
                                    "table_sum", "table_average", "table_max", "table_min")]
                pred_ops = [op for op in pred_raw.replace("(", " ").replace(")", " ").replace(",", " ").split()
                           if op in ("add", "subtract", "multiply", "divide", "exp", "greater",
                                    "table_sum", "table_average", "table_max", "table_min")]
                if gold_ops == pred_ops:
                    category = "wrong_values"
                elif len(gold_ops) != len(pred_ops):
                    category = "wrong_num_steps"
                else:
                    category = "wrong_approach"
        else:
            category = "wrong_approach"

        # Count gold steps
        gold_steps = gold_prog.count("),") + 1

        failures.append({
            "id": pred["id"],
            "question": entry["qa"]["question"],
            "gold_prog": gold_prog,
            "pred_prog": pred.get("raw_program", ""),
            "gold_ans": gold_res,
            "pred_ans": exe_res if not invalid_flag else "INVALID",
            "ratio": ratio,
            "category": category,
            "gold_steps": gold_steps,
            "prog_match": prog_ok,
            "rounds_used": pred.get("rounds_used", 1),
        })

# === Summary ===
print(f"Total examples: {len(preds)}")
print(f"Exe correct: {len(preds) - len(failures)}")
print(f"Exe failures: {len(failures)}")
print(f"Exe accuracy: {(len(preds) - len(failures)) / len(preds):.4f}")
print()

# Category breakdown
cat_counts = Counter(f["category"] for f in failures)
print("=== FAILURE CATEGORIES ===")
for cat, count in cat_counts.most_common():
    pct = count / len(failures) * 100
    print(f"  {cat:20s}: {count:4d} ({pct:.1f}%)")
print()

# By number of gold steps
step_counts = Counter(f["gold_steps"] for f in failures)
total_by_steps = Counter()
for pred in preds:
    entry = data_dict[pred["id"]]
    steps = entry["qa"]["program"].count("),") + 1
    total_by_steps[steps] += 1

print("=== ACCURACY BY GOLD PROGRAM COMPLEXITY ===")
for steps in sorted(total_by_steps.keys()):
    total = total_by_steps[steps]
    failed = step_counts.get(steps, 0)
    correct = total - failed
    acc = correct / total if total > 0 else 0
    print(f"  {steps}-step programs: {correct}/{total} = {acc:.1%}")
print()

# Examples by category
print("=== SAMPLE FAILURES BY CATEGORY ===")
for cat in cat_counts:
    cat_failures = [f for f in failures if f["category"] == cat][:3]
    print(f"\n--- {cat} ({cat_counts[cat]} total) ---")
    for f in cat_failures:
        ratio_str = f"  ratio={f['ratio']:.4f}" if f['ratio'] is not None else ""
        print(f"  ID: {f['id']}")
        print(f"  Q: {f['question'][:100]}")
        print(f"  Gold: {f['gold_prog'][:80]}  => {f['gold_ans']}")
        print(f"  Pred: {f['pred_prog'][:80]}  => {f['pred_ans']}{ratio_str}")
        print(f"  Rounds: {f['rounds_used']}")
        print()

# Verification round analysis
print("=== ROUNDS USED (FAILURES vs ALL) ===")
all_rounds = Counter(p.get("rounds_used", 1) for p in preds)
fail_rounds = Counter(f["rounds_used"] for f in failures)
for r in sorted(all_rounds.keys()):
    total = all_rounds[r]
    failed = fail_rounds.get(r, 0)
    correct = total - failed
    acc = correct / total if total > 0 else 0
    print(f"  Round {r}: {correct}/{total} correct ({acc:.1%})")
print()

# Save detailed failures to JSON for further analysis
with open("output/failures_dev.json", "w") as f:
    json.dump(failures, f, indent=2, default=str)
print(f"Detailed failures saved to output/failures_dev.json ({len(failures)} entries)")
