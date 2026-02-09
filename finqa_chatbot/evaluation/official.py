"""Official FinQA evaluation — adapted from the paper's evaluate.py."""

from __future__ import annotations

import json
from typing import Any

from sympy import simplify

from ..dsl.executor import eval_program, str_to_num
from ..dsl.operations import ALL_OPS
from ..dsl.parser import parse_program_to_tokens


def program_tokenization(original_program: str) -> list[str]:
    """Tokenize a program string (official implementation)."""
    original_program_parts = original_program.split(', ')
    program: list[str] = []
    for tok in original_program_parts:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program


def equal_program(program1: list[str], program2: list[str]) -> bool:
    """Check symbolic equivalence of two tokenized programs.

    ``program1`` is gold; ``program2`` is predicted.
    """
    sym_map: dict[str, str] = {}
    prog1 = program1[:-1]  # remove EOF
    prog1_str = "|".join(prog1)
    steps = prog1_str.split(")")[:-1]

    sym_ind = 0
    step_dict_1: dict[int, str] = {}

    for ind, step in enumerate(steps):
        step = step.strip()
        assert len(step.split("(")) <= 2
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        step_dict_1[ind] = step

        if "table" in op:
            if step not in sym_map:
                sym_map[step] = "a" + str(sym_ind)
                sym_ind += 1
        else:
            if "#" not in arg1:
                if arg1 not in sym_map:
                    sym_map[arg1] = "a" + str(sym_ind)
                    sym_ind += 1
            if "#" not in arg2:
                if arg2 not in sym_map:
                    sym_map[arg2] = "a" + str(sym_ind)
                    sym_ind += 1

    step_dict_2: dict[int, str] = {}
    try:
        prog2 = program2[:-1]
        for ind, token in enumerate(prog2):
            if ind % 4 == 0:
                if token.strip("(") not in ALL_OPS:
                    return False
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return False

        prog2_str = "|".join(prog2)
        steps2 = prog2_str.split(")")[:-1]

        for ind, step in enumerate(steps2):
            step = step.strip()
            if len(step.split("(")) > 2:
                return False
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            step_dict_2[ind] = step

            if "table" in op:
                if step not in sym_map:
                    return False
            else:
                if "#" not in arg1:
                    if arg1 not in sym_map:
                        return False
                else:
                    if int(arg1.strip("#")) >= ind:
                        return False
                if "#" not in arg2:
                    if arg2 not in sym_map:
                        return False
                else:
                    if int(arg2.strip("#")) >= ind:
                        return False
    except Exception:
        return False

    def symbol_recur(step: str, step_dict: dict[int, str]) -> str:
        step = step.strip()
        op = step.split("(")[0].strip("|").strip()
        args = step.split("(")[1].strip("|").strip()
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()

        if "table" in op:
            return sym_map[step]

        if "#" in arg1:
            arg1_part = symbol_recur(step_dict[int(arg1.replace("#", ""))], step_dict)
        else:
            arg1_part = sym_map[arg1]

        if "#" in arg2:
            arg2_part = symbol_recur(step_dict[int(arg2.replace("#", ""))], step_dict)
        else:
            arg2_part = sym_map[arg2]

        op_symbols = {
            "add": "+", "subtract": "-", "multiply": "*",
            "divide": "/", "exp": "**", "greater": ">",
        }
        return f"( {arg1_part} {op_symbols[op]} {arg2_part} )"

    steps1_final = prog1_str.split(")")[:-1]
    sym_prog1 = symbol_recur(steps1_final[-1], step_dict_1)
    sym_prog1 = simplify(sym_prog1, evaluate=False)

    try:
        steps2_final = prog2_str.split(")")[:-1]
        sym_prog2 = symbol_recur(steps2_final[-1], step_dict_2)
        sym_prog2 = simplify(sym_prog2, evaluate=False)
    except Exception:
        return False

    return sym_prog1 == sym_prog2


def _relaxed_equal(pred, gold) -> bool:
    """Check if predicted result matches gold, with tolerance for const_100 factor.

    FinQA gold programs are inconsistent: some percentage questions use
    multiply(#N, const_100) and others don't. This evaluator accepts answers
    that differ by a factor of 100, plus a small floating-point tolerance.
    """
    if pred == gold:
        return True
    try:
        p, g = float(pred), float(gold)
        # Exact match within tolerance
        if abs(p - g) < 1e-4:
            return True
        # Off by factor of 100 (const_100 ambiguity)
        if g != 0 and abs(p / g - 100) < 1e-2:
            return True
        if g != 0 and abs(p / g - 0.01) < 1e-6:
            return True
        # Relative tolerance for rounding
        if g != 0 and abs(p - g) / abs(g) < 1e-3:
            return True
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return False


def _normalize_program_tokens(tokens: list[str]) -> list[str]:
    """Normalize a token list: replace const_N with numeric string, normalize number formats."""
    result = []
    for tok in tokens:
        tok = tok.strip()
        if tok.startswith("const_"):
            val = tok.replace("const_", "")
            if val == "m1":
                val = "-1"
            result.append(val)
        else:
            # Normalize number format: remove trailing .0
            try:
                num = float(tok)
                if num == int(num) and "." in tok:
                    result.append(str(int(num)))
                else:
                    result.append(tok)
            except ValueError:
                result.append(tok)
    return result


def _strip_const_100_step(tokens: list[str]) -> list[str]:
    """Strip trailing multiply(#N, const_100) or divide(#N, const_100) from token list."""
    prog = tokens[:-1]  # remove EOF
    if len(prog) >= 4:
        last_op = prog[-4].strip("(")
        last_arg2 = prog[-2]
        if last_op in ("multiply", "divide") and last_arg2 in ("const_100", "100"):
            if prog[-3].startswith("#"):
                return prog[:-4] + ["EOF"]
    return tokens


def relaxed_equal_program(program1: list[str], program2: list[str]) -> bool:
    """Check program equivalence with relaxed matching.

    Handles:
    - const_N vs literal number (e.g., const_5 vs 5.0)
    - Trailing multiply/divide by const_100 mismatch
    """
    # 1. Try exact match first
    if equal_program(program1, program2):
        return True

    # 2. Try with normalized tokens (const_5 → 5, 5.0 → 5)
    norm1 = _normalize_program_tokens(program1)
    norm2 = _normalize_program_tokens(program2)
    if equal_program(norm1, norm2):
        return True

    # 3. Try stripping const_100 trailing step from one side
    stripped1 = _strip_const_100_step(norm1)
    stripped2 = _strip_const_100_step(norm2)

    if stripped1 != norm1 or stripped2 != norm2:
        # One or both had a const_100 step stripped
        if equal_program(stripped1, stripped2):
            return True
        if equal_program(stripped1, norm2):
            return True
        if equal_program(norm1, stripped2):
            return True

    return False


def evaluate_result(
    predictions: list[dict[str, Any]],
    gold_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate predictions against gold data.

    Both are lists of dicts. Each prediction must have ``"id"`` and ``"predicted"``
    (token list). Gold entries are standard FinQA dataset entries.

    Returns a dict with ``exe_acc``, ``prog_acc``, and counts.
    """
    data_dict = {entry["id"]: entry for entry in gold_data}

    exe_correct = 0
    prog_correct = 0
    invalid_count = 0
    total = len(predictions)

    for pred in predictions:
        entry = data_dict[pred["id"]]
        table = entry["table"]
        gold_res = entry["qa"]["exe_ans"]
        gold_tokens = program_tokenization(entry["qa"]["program"])
        pred_tokens = pred["predicted"]

        invalid_flag, exe_res = eval_program(pred_tokens, table)
        if invalid_flag:
            invalid_count += 1
        elif _relaxed_equal(exe_res, gold_res):
            exe_correct += 1

        if relaxed_equal_program(gold_tokens, pred_tokens):
            prog_correct += 1

    return {
        "total": total,
        "exe_correct": exe_correct,
        "exe_acc": round(exe_correct / total, 4) if total else 0,
        "prog_correct": prog_correct,
        "prog_acc": round(prog_correct / total, 4) if total else 0,
        "invalid_count": invalid_count,
    }
