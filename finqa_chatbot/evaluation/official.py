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
        elif exe_res == gold_res:
            exe_correct += 1

        if equal_program(gold_tokens, pred_tokens):
            prog_correct += 1

    return {
        "total": total,
        "exe_correct": exe_correct,
        "exe_acc": round(exe_correct / total, 4) if total else 0,
        "prog_correct": prog_correct,
        "prog_acc": round(prog_correct / total, 4) if total else 0,
        "invalid_count": invalid_count,
    }
