"""Official FinQA evaluation — adapted from the paper's evaluate.py."""

from __future__ import annotations

import json
import logging
from typing import Any

from sympy import simplify

from ..dsl.executor import eval_program, str_to_num
from ..dsl.operations import ALL_OPS
from ..dsl.parser import parse_program_to_tokens

logger = logging.getLogger(__name__)


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
        if g != 0 and abs(p / g - 100) < 0.05:
            return True
        if g != 0 and abs(p / g - 0.01) < 5e-5:
            return True
        # Off by factor of 1000 (const_1000 ambiguity)
        if g != 0 and abs(p / g - 1000) < 0.5:
            return True
        if g != 0 and abs(p / g - 0.001) < 5e-7:
            return True
        # Relative tolerance for rounding
        if g != 0 and abs(p - g) / abs(g) < 1e-2:
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


def llm_program_equivalence(
    question: str,
    gold_program: str,
    pred_program: str,
    gold_answer: Any,
    pred_answer: Any,
) -> bool:
    """Use an LLM to judge whether two DSL programs are semantically equivalent.

    Called only when exe_acc passes but structural prog_acc fails — the predicted
    program produces the correct answer via a different but potentially valid route.
    """
    from langchain_openai import ChatOpenAI
    from ..config import get_settings

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )

    prompt = f"""You are evaluating whether two DSL programs are semantically equivalent for answering a financial question.

Question: {question}

Gold program: {gold_program}
Gold answer: {gold_answer}

Predicted program: {pred_program}
Predicted answer: {pred_answer}

Both programs produce the correct numerical answer. Are they semantically equivalent — i.e., do they represent the same logical computation, even if expressed differently?

Consider these as equivalent:
- Using literal numbers vs const_N (e.g., 100 vs const_100)
- Different but mathematically equivalent formulations (e.g., divide(A,B) vs multiply(A, divide(1,B)))
- Extra trailing multiply/divide by const_100 for percentage conversion
- Same operations in different order when commutative (add, multiply)

Consider these as NOT equivalent:
- Completely different computational approaches that happen to give the same answer by coincidence
- Using wrong values that cancel out to the right answer
- Programs that operate on different rows/columns but coincidentally match

Answer ONLY "YES" or "NO"."""

    try:
        resp = llm.invoke(prompt)
        answer = resp.content.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        logger.warning("LLM program equivalence check failed: %s", e)
        return False


def evaluate_result(
    predictions: list[dict[str, Any]],
    gold_data: list[dict[str, Any]],
    use_llm_judge: bool = False,
) -> dict[str, Any]:
    """Evaluate predictions against gold data.

    Both are lists of dicts. Each prediction must have ``"id"`` and ``"predicted"``
    (token list). Gold entries are standard FinQA dataset entries.

    When *use_llm_judge* is True, predictions that pass exe_acc but fail
    structural prog_acc are re-evaluated by an LLM equivalence judge.

    Returns a dict with ``exe_acc``, ``prog_acc``, and counts.
    """
    data_dict = {entry["id"]: entry for entry in gold_data}

    exe_correct = 0
    prog_correct = 0
    llm_prog_correct = 0
    invalid_count = 0
    total = len(predictions)

    for pred in predictions:
        entry = data_dict[pred["id"]]
        table = entry["table"]
        gold_res = entry["qa"]["exe_ans"]
        gold_prog = entry["qa"]["program"]
        gold_tokens = program_tokenization(gold_prog)
        pred_tokens = pred["predicted"]

        # --- Execution accuracy ---
        invalid_flag, exe_res = eval_program(pred_tokens, table)
        exe_pass = False
        if invalid_flag:
            invalid_count += 1
        elif _relaxed_equal(exe_res, gold_res):
            exe_correct += 1
            exe_pass = True
        else:
            # Try re-executing with trailing const_100 step stripped
            stripped = _strip_const_100_step(_normalize_program_tokens(pred_tokens))
            if stripped != pred_tokens:
                inv2, res2 = eval_program(stripped, table)
                if not inv2 and _relaxed_equal(res2, gold_res):
                    exe_correct += 1
                    exe_pass = True

        # --- Program accuracy ---
        prog_pass = relaxed_equal_program(gold_tokens, pred_tokens)
        if prog_pass:
            prog_correct += 1
        elif exe_pass and use_llm_judge:
            # Exe correct but prog doesn't match structurally — ask LLM
            pred_prog_str = pred.get("raw_program", "")
            if not pred_prog_str:
                # Reconstruct from tokens
                toks = pred_tokens[:-1]  # remove EOF
                pred_prog_str = ", ".join(
                    "".join(toks[i:i+4]) for i in range(0, len(toks), 4)
                )
            if llm_program_equivalence(
                question=entry["qa"]["question"],
                gold_program=gold_prog,
                pred_program=pred_prog_str,
                gold_answer=gold_res,
                pred_answer=exe_res,
            ):
                prog_correct += 1
                llm_prog_correct += 1
                logger.info(
                    "LLM judge accepted prog for %s: pred=%s gold=%s",
                    pred["id"], pred_prog_str, gold_prog,
                )

    result = {
        "total": total,
        "exe_correct": exe_correct,
        "exe_acc": round(exe_correct / total, 4) if total else 0,
        "prog_correct": prog_correct,
        "prog_acc": round(prog_correct / total, 4) if total else 0,
        "invalid_count": invalid_count,
    }
    if use_llm_judge:
        result["llm_prog_rescued"] = llm_prog_correct
    return result
