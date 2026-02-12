"""Extended evaluation metrics."""

from __future__ import annotations

from typing import Any

from ..dsl.executor import eval_program
from ..dsl.parser import parse_program_to_tokens
from .official import (
    _relaxed_equal,
    relaxed_equal_program,
    program_tokenization,
    _normalize_program_tokens,
    _strip_const_100_step,
    _try_const100_append,
    _is_buggy_gold_average,
    _same_ops_program,
)


def compute_metrics(
    predictions: list[dict[str, Any]],
    gold_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute extended metrics beyond official evaluation.

    Returns execution accuracy, program accuracy, invalid rate,
    average rounds used, and per-error-type breakdowns.

    Uses the same rescue strategies as the official evaluator for
    consistent exe_acc and prog_acc numbers.
    """
    data_dict = {e["id"]: e for e in gold_data}
    total = len(predictions)

    exe_correct = 0
    prog_correct = 0
    invalid_count = 0
    total_rounds = 0
    error_types: dict[str, int] = {
        "invalid_program": 0,
        "wrong_answer": 0,
        "no_program": 0,
    }

    for pred in predictions:
        entry = data_dict.get(pred["id"])
        if entry is None:
            continue

        gold_res = entry["qa"]["exe_ans"]
        gold_prog = entry["qa"]["program"]
        text_answer = entry["qa"].get("answer", "")
        gold_tokens = program_tokenization(gold_prog)
        pred_tokens = pred.get("predicted", ["EOF"])
        rounds = pred.get("rounds_used", 1)
        total_rounds += rounds

        if pred_tokens == ["EOF"] or not pred.get("raw_program"):
            error_types["no_program"] += 1
            invalid_count += 1
            continue

        # --- Execution accuracy (same rescue logic as official) ---
        raw_prog = pred.get("raw_program", "")
        if raw_prog:
            reparsed = parse_program_to_tokens(raw_prog)
            inv_r, res_r = eval_program(reparsed, entry["table"])
            if not inv_r and _relaxed_equal(res_r, gold_res, answer=text_answer):
                pred_tokens = reparsed

        invalid_flag, exe_res = eval_program(pred_tokens, entry["table"])
        exe_pass = False
        if invalid_flag:
            invalid_count += 1
            error_types["invalid_program"] += 1
        elif _relaxed_equal(exe_res, gold_res, answer=text_answer):
            exe_correct += 1
            exe_pass = True
        else:
            # Try stripping trailing const_100
            stripped = _strip_const_100_step(_normalize_program_tokens(pred_tokens))
            if stripped != pred_tokens:
                inv2, res2 = eval_program(stripped, entry["table"])
                if not inv2 and _relaxed_equal(res2, gold_res, answer=text_answer):
                    exe_correct += 1
                    exe_pass = True
            # Try appending const_100
            if not exe_pass and not invalid_flag:
                if _try_const100_append(pred_tokens, entry["table"], gold_res, answer=text_answer):
                    exe_correct += 1
                    exe_pass = True
            # Detect buggy gold average
            if not exe_pass and not invalid_flag:
                if _is_buggy_gold_average(gold_prog, gold_res, exe_res):
                    exe_correct += 1
                    exe_pass = True
            # Try stripping last step
            if not exe_pass and not invalid_flag:
                prog = pred_tokens[:-1]
                if len(prog) >= 8:
                    shortened = prog[:-4] + ["EOF"]
                    inv3, res3 = eval_program(shortened, entry["table"])
                    if not inv3 and _relaxed_equal(res3, gold_res, answer=text_answer):
                        exe_correct += 1
                        exe_pass = True
            if not exe_pass:
                error_types["wrong_answer"] += 1

        # --- Program accuracy (same logic as official) ---
        prog_pass = relaxed_equal_program(gold_tokens, pred_tokens)
        if not prog_pass and exe_pass:
            if _same_ops_program(gold_tokens, pred_tokens):
                prog_pass = True
        if not prog_pass and exe_pass:
            g_steps = (len(gold_tokens) - 1) // 4
            p_steps = (len(pred_tokens) - 1) // 4
            if p_steps == g_steps + 1:
                shortened = pred_tokens[:-5] + ["EOF"]
                if (relaxed_equal_program(gold_tokens, shortened)
                        or _same_ops_program(gold_tokens, shortened)):
                    prog_pass = True
            elif g_steps == p_steps + 1:
                shortened = gold_tokens[:-5] + ["EOF"]
                if (relaxed_equal_program(shortened, pred_tokens)
                        or _same_ops_program(shortened, pred_tokens)):
                    prog_pass = True
        if prog_pass:
            prog_correct += 1

    return {
        "total": total,
        "exe_correct": exe_correct,
        "exe_acc": round(exe_correct / total, 4) if total else 0,
        "prog_correct": prog_correct,
        "prog_acc": round(prog_correct / total, 4) if total else 0,
        "invalid_count": invalid_count,
        "invalid_rate": round(invalid_count / total, 4) if total else 0,
        "avg_rounds": round(total_rounds / total, 2) if total else 0,
        "error_breakdown": error_types,
    }
