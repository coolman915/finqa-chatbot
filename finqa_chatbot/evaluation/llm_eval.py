"""LLM-based evaluation of failed predictions — judges correctness and classifies failure reasons."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

FAILURE_REASONS = [
    "correct_alternate",
    "wrong_number",
    "wrong_computation",
    "sign_error",
    "scale_error",
    "missing_step",
    "extra_step",
    "wrong_approach",
    "invalid_program",
    "rounding_error",
]

_EVAL_PROMPT = """\
You are evaluating a predicted answer to a financial question against the gold (correct) answer.

## Source Data

Table:
{table}

Pre-text:
{pre_text}

Post-text:
{post_text}

## Question

{question}

## Gold (expected)

Program: {gold_program}
Answer: {gold_answer}
Text answer: {text_answer}

## Predicted

Program: {pred_program}
Answer: {pred_answer}

## Values extracted by agents

{extracted_values}

## Task

1. Using the source data above, judge whether the predicted program and answer is actually correct (it may use a valid alternate approach that the heuristic evaluator missed).
2. If not correct, classify the failure reason from this fixed set:
   - correct_alternate: prediction is correct via an alternate valid approach
   - wrong_number: used wrong values from the table/text
   - wrong_computation: right values but wrong operation (e.g. add instead of subtract)
   - sign_error: result has opposite sign
   - scale_error: off by a factor of 10, 100, or 1000
   - missing_step: program is missing a computation step
   - extra_step: program has an unnecessary extra step
   - wrong_approach: completely different/wrong method
   - invalid_program: program can't be parsed or executed
   - rounding_error: close but not within tolerance
3. Provide a brief explanation (1-2 sentences).

## Important: const_100 scaling
If the only difference is a trailing multiply(#N, const_100) or divide(#N, const_100) step (converting between ratio and percentage points, e.g. 0.227 vs 22.7), judge as CORRECT. Both representations are valid — expressing "22.7% growth" as 0.227 or 22.7 are equivalent.

## Response format (strict)

CORRECT: YES or NO
REASON: <one of the categories above>
EXPLANATION: <1-2 sentence explanation>"""


def _format_table(table: list[list[str]]) -> str:
    """Format a table as a markdown string for the eval prompt."""
    if not table:
        return "(no table)"
    lines = []
    for row in table:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def llm_evaluate_prediction(
    question: str,
    gold_program: str,
    pred_program: str,
    gold_answer: Any,
    pred_answer: Any,
    text_answer: str = "",
    table: list[list[str]] | None = None,
    pre_text: list[str] | None = None,
    post_text: list[str] | None = None,
    extracted_values: str = "",
) -> dict:
    """LLM evaluation of a failed prediction.

    Args:
        question: The financial question.
        gold_program: Gold DSL program.
        pred_program: Predicted DSL program.
        gold_answer: Gold execution result (numeric).
        pred_answer: Predicted execution result (numeric).
        text_answer: Human-readable answer (e.g. "15.3%").
        table: Source table as list of rows.
        pre_text: Pre-text paragraphs from the dataset entry.
        post_text: Post-text paragraphs from the dataset entry.
        extracted_values: Agent-extracted values (formatted string from shared log).

    Returns:
        {
            "llm_correct": bool,
            "failure_reason": str,
            "llm_explanation": str,
        }
    """
    from langchain_openai import ChatOpenAI
    from ..config import get_settings

    settings = get_settings()
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key=settings.openai_api_key,
    )

    prompt = _EVAL_PROMPT.format(
        question=question,
        gold_program=gold_program,
        pred_program=pred_program,
        gold_answer=gold_answer,
        pred_answer=pred_answer,
        text_answer=text_answer,
        table=_format_table(table) if table else "(no table)",
        pre_text="\n".join(pre_text) if pre_text else "(none)",
        post_text="\n".join(post_text) if post_text else "(none)",
        extracted_values=extracted_values or "(none)",
    )

    try:
        resp = llm.invoke(prompt)
        return _parse_response(resp.content)
    except Exception as e:
        logger.warning("LLM evaluation failed: %s", e)
        return {
            "llm_correct": None,
            "failure_reason": "llm_error",
            "llm_explanation": str(e),
        }


def _parse_response(text: str) -> dict:
    """Parse the structured LLM response into a dict."""
    lines = text.strip().splitlines()
    correct = None
    reason = "unknown"
    explanation = ""

    for line in lines:
        line = line.strip()
        upper = line.upper()
        if upper.startswith("CORRECT:"):
            val = line.split(":", 1)[1].strip().upper()
            correct = val.startswith("YES")
        elif upper.startswith("REASON:"):
            val = line.split(":", 1)[1].strip().lower()
            # Match to known categories
            for r in FAILURE_REASONS:
                if r in val:
                    reason = r
                    break
            else:
                reason = val
        elif upper.startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()

    if correct:
        reason = "correct_alternate"

    return {
        "llm_correct": correct,
        "failure_reason": reason,
        "llm_explanation": explanation,
    }
