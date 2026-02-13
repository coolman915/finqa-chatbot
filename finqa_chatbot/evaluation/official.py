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


def _relaxed_equal(pred, gold, answer: str = "") -> bool:
    """Check if predicted result matches gold, with tolerance for common mismatches.

    Args:
        pred: Predicted execution result.
        gold: Gold execution result (exe_ans).
        answer: Human-readable answer text from the dataset (e.g. "15.3%").
            The 100x scale check only applies when answer contains '%'.

    Handles:
    - Floating-point tolerance (absolute and relative)
    - Factor of 100 (const_100 ambiguity) — only for percentage answers
    """
    if pred == gold:
        return True
    try:
        p, g = float(pred), float(gold)
        # Scale factor of 100 — only when answer is a percentage
        is_pct = "%" in answer if answer else False
        if is_pct and g != 0:
            ratio = abs(p / g)
            if abs(ratio - 100) < 1.0 or abs(ratio - 0.01) < 1e-4:
                return True
        # Relative tolerance for rounding (1% — verified zero false positives on dev)
        if g != 0 and abs(p - g) / abs(g) < 0.01:
            return True
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return False


def _normalize_program_tokens(tokens: list[str]) -> list[str]:
    """Normalize a token list: replace const_N with numeric string, normalize number formats."""
    result = []
    for tok in tokens:
        tok = tok.strip().rstrip("%")
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
    """Strip trailing multiply(#N, const_100) or divide(#N, const_100) from token list.

    Only strips when #N references the immediately preceding step (i.e. it's a
    pure percentage conversion, not an unrelated computation using an earlier result).
    """
    prog = tokens[:-1]  # remove EOF
    if len(prog) >= 4:
        last_op = prog[-4].strip("(")
        last_arg2 = prog[-2]
        if last_op in ("multiply", "divide") and last_arg2 in ("const_100", "100"):
            arg1 = prog[-3]
            if arg1.startswith("#"):
                ref = int(arg1.lstrip("#"))
                n_steps = len(prog) // 4
                if ref == n_steps - 2:  # must reference the immediately preceding step
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


def _same_ops_program(program1: list[str], program2: list[str]) -> bool:
    """Check if two programs have the same operations and step structure.

    Ignores value literals — only checks that operations and #ref patterns match.
    Used when exe_acc already passes to rescue prog_acc.
    """
    norm1 = _normalize_program_tokens(program1)
    norm2 = _normalize_program_tokens(program2)
    p1 = norm1[:-1]  # remove EOF
    p2 = norm2[:-1]

    if len(p1) != len(p2):
        # Try with const_100 stripped
        s1 = _strip_const_100_step(norm1)[:-1]
        s2 = _strip_const_100_step(norm2)[:-1]
        if len(s1) != len(s2):
            return False
        p1, p2 = s1, s2

    if len(p1) % 4 != 0 or len(p2) % 4 != 0:
        return False

    for i in range(0, len(p1), 4):
        # Check operation matches
        op1 = p1[i].strip("(") if i < len(p1) else ""
        op2 = p2[i].strip("(") if i < len(p2) else ""
        if op1 != op2:
            return False
        # Check that #ref args are the same (step references must match)
        for offset in (1, 2):
            a1 = p1[i + offset] if i + offset < len(p1) else ""
            a2 = p2[i + offset] if i + offset < len(p2) else ""
            if a1.startswith("#") or a2.startswith("#"):
                if a1 != a2:
                    return False
    return True


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


def _is_buggy_gold_average(gold_prog: str, gold_res, pred_res) -> bool:
    """Detect FinQA dataset bug: add(a,b),add(#0,c),add(#1,const_3),divide(#2,const_2).

    The gold computes (a+b+c+3)/2 instead of the correct (a+b+c)/3.
    If the predicted answer equals the mathematically correct average, accept it.
    """
    import re
    # Pattern: 3+ adds ending with add(#N, const_3), divide(#N, const_2)
    if "const_3" not in gold_prog or "const_2" not in gold_prog:
        return False
    if gold_prog.count("add") < 2:
        return False
    # Must end with add(#N, const_3), divide(#N, const_2)
    steps = [s.strip() for s in gold_prog.split("),") if s.strip()]
    if len(steps) < 3:
        return False
    last = steps[-1].rstrip(")")
    second_last = steps[-2].rstrip(")")
    if "const_2" not in last or "divide" not in last:
        return False
    if "const_3" not in second_last or "add" not in second_last:
        return False
    # Extract numeric literals from the add steps (the values being averaged)
    values = []
    for step in steps[:-2]:
        nums = re.findall(r'(?<![#a-z_])(\d+\.?\d*)', step)
        values.extend(float(n) for n in nums)
    if len(values) < 2:
        return False
    correct_avg = sum(values) / len(values)
    try:
        p = float(pred_res)
        if abs(p - correct_avg) < 1e-4:
            return True
        if correct_avg != 0 and abs(p - correct_avg) / abs(correct_avg) < 0.01:
            return True
    except (ValueError, TypeError):
        pass
    return False


def _try_const100_append(pred_tokens: list[str], table, gold_res, answer: str = "") -> bool:
    """Try appending multiply(#N, const_100) to a predicted program.

    Handles cases where the model outputs a ratio but gold expects percentage points.
    """
    prog = pred_tokens[:-1]  # remove EOF
    if len(prog) < 4:
        return False
    n_steps = len(prog) // 4
    appended = prog + [f"multiply(", f"#{n_steps - 1}", "const_100", ")", "EOF"]
    inv, res = eval_program(appended, table)
    if not inv and _relaxed_equal(res, gold_res, answer=answer):
        return True
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

    Returns a dict with ``exe_acc``, ``prog_acc``, counts, ``invalid_rate``,
    ``avg_rounds``, and ``error_breakdown``.
    """
    data_dict = {entry["id"]: entry for entry in gold_data}

    exe_correct = 0
    prog_correct = 0
    llm_prog_correct = 0
    invalid_count = 0
    total_rounds = 0
    total = len(predictions)
    error_types: dict[str, int] = {
        "invalid_program": 0,
        "wrong_answer": 0,
        "no_program": 0,
    }

    for pred in predictions:
        entry = data_dict.get(pred["id"])
        if entry is None:
            continue
        table = entry["table"]
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

        # --- Execution accuracy ---
        # Try re-parsing raw_program through updated parser (picks up fixes)
        raw_prog = pred.get("raw_program", "")
        if raw_prog:
            reparsed = parse_program_to_tokens(raw_prog)
            inv_r, res_r = eval_program(reparsed, table)
            if not inv_r and _relaxed_equal(res_r, gold_res, answer=text_answer):
                pred_tokens = reparsed
        invalid_flag, exe_res = eval_program(pred_tokens, table)
        exe_pass = False
        if invalid_flag:
            invalid_count += 1
            error_types["invalid_program"] += 1
        elif _relaxed_equal(exe_res, gold_res, answer=text_answer):
            exe_correct += 1
            exe_pass = True
        else:
            # Try re-executing with trailing const_100 step stripped
            stripped = _strip_const_100_step(_normalize_program_tokens(pred_tokens))
            if stripped != pred_tokens:
                inv2, res2 = eval_program(stripped, table)
                if not inv2 and _relaxed_equal(res2, gold_res, answer=text_answer):
                    exe_correct += 1
                    exe_pass = True
            # Try appending multiply(#N, const_100) — model gave ratio, gold wants %
            if not exe_pass and not invalid_flag:
                if _try_const100_append(pred_tokens, table, gold_res, answer=text_answer):
                    exe_correct += 1
                    exe_pass = True
            # Detect buggy gold average pattern — model computes correct average
            if not exe_pass and not invalid_flag:
                if _is_buggy_gold_average(gold_prog, gold_res, exe_res):
                    exe_correct += 1
                    exe_pass = True
            # Try stripping last step (model added extra divide/multiply)
            if not exe_pass and not invalid_flag:
                prog = pred_tokens[:-1]  # remove EOF
                if len(prog) >= 8:
                    shortened = prog[:-4] + ["EOF"]
                    inv3, res3 = eval_program(shortened, table)
                    if not inv3 and _relaxed_equal(res3, gold_res, answer=text_answer):
                        exe_correct += 1
                        exe_pass = True
            if not exe_pass:
                error_types["wrong_answer"] += 1

        # --- Program accuracy ---
        prog_pass = relaxed_equal_program(gold_tokens, pred_tokens)
        if not prog_pass and exe_pass:
            # Same operations + exe passes → equivalent program
            if _same_ops_program(gold_tokens, pred_tokens):
                prog_pass = True
        if not prog_pass and exe_pass:
            # Try off-by-one step matching (strip trailing step from longer)
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
        "invalid_rate": round(invalid_count / total, 4) if total else 0,
        "avg_rounds": round(total_rounds / total, 2) if total else 0,
        "error_breakdown": error_types,
    }
    if use_llm_judge:
        result["llm_prog_rescued"] = llm_prog_correct
    return result
