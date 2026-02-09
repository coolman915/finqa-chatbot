"""Extended evaluation metrics."""

from __future__ import annotations

from typing import Any

from ..dsl.executor import eval_program
from ..dsl.parser import parse_program_to_tokens


def compute_metrics(
    predictions: list[dict[str, Any]],
    gold_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute extended metrics beyond official evaluation.

    Returns execution accuracy, program accuracy, invalid rate,
    average rounds used, and per-error-type breakdowns.
    """
    data_dict = {e["id"]: e for e in gold_data}
    total = len(predictions)

    exe_correct = 0
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
        pred_tokens = pred.get("predicted", ["EOF"])
        rounds = pred.get("rounds_used", 1)
        total_rounds += rounds

        if pred_tokens == ["EOF"] or not pred.get("raw_program"):
            error_types["no_program"] += 1
            invalid_count += 1
            continue

        invalid_flag, exe_res = eval_program(pred_tokens, entry["table"])
        if invalid_flag:
            invalid_count += 1
            error_types["invalid_program"] += 1
        elif exe_res == gold_res:
            exe_correct += 1
        else:
            error_types["wrong_answer"] += 1

    return {
        "total": total,
        "exe_correct": exe_correct,
        "exe_acc": round(exe_correct / total, 4) if total else 0,
        "invalid_count": invalid_count,
        "invalid_rate": round(invalid_count / total, 4) if total else 0,
        "avg_rounds": round(total_rounds / total, 2) if total else 0,
        "error_breakdown": error_types,
    }
