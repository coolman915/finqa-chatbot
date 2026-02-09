"""LangSmith dataset upload and evaluator definitions."""

from __future__ import annotations

import json
from typing import Any

from langsmith import Client
from langsmith.schemas import Example, Run

from ..config import get_settings
from ..dsl.executor import eval_program
from ..dsl.parser import parse_program_to_tokens
from ..evaluation.official import equal_program, program_tokenization


def upload_dataset(
    data_path: str,
    dataset_name: str = "finqa-dev",
    description: str = "FinQA development set",
) -> str:
    """Upload a FinQA JSON file as a LangSmith dataset.

    Returns the dataset ID.
    """
    settings = get_settings()
    client = Client(api_key=settings.langchain_api_key)

    with open(data_path) as f:
        data = json.load(f)

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description,
    )

    for entry in data:
        client.create_example(
            inputs={
                "entry": entry,
                "question": entry["qa"]["question"],
            },
            outputs={
                "gold_program": entry["qa"]["program"],
                "gold_answer": entry["qa"]["exe_ans"],
            },
            dataset_id=dataset.id,
        )

    return str(dataset.id)


# ── Custom LangSmith evaluators ────────────────────────────────────────

def execution_accuracy(run: Run, example: Example) -> dict:
    """LangSmith evaluator: checks if predicted program produces correct answer."""
    outputs = run.outputs or {}
    pred_program = outputs.get("selected_program", "")
    gold_answer = example.outputs.get("gold_answer")
    table = example.inputs.get("entry", {}).get("table", [])

    if not pred_program:
        return {"key": "execution_accuracy", "score": 0}

    tokens = parse_program_to_tokens(pred_program)
    invalid, result = eval_program(tokens, table)

    if invalid:
        return {"key": "execution_accuracy", "score": 0}

    score = 1 if result == gold_answer else 0
    return {"key": "execution_accuracy", "score": score}


def program_accuracy(run: Run, example: Example) -> dict:
    """LangSmith evaluator: symbolic program equivalence."""
    outputs = run.outputs or {}
    pred_program = outputs.get("selected_program", "")
    gold_program = example.outputs.get("gold_program", "")

    if not pred_program or not gold_program:
        return {"key": "program_accuracy", "score": 0}

    pred_tokens = parse_program_to_tokens(pred_program)
    gold_tokens = program_tokenization(gold_program)

    score = 1 if equal_program(gold_tokens, pred_tokens) else 0
    return {"key": "program_accuracy", "score": score}


def rounds_used(run: Run, example: Example) -> dict:
    """LangSmith evaluator: how many DeALOG rounds were used."""
    outputs = run.outputs or {}
    rounds = outputs.get("round_number", 1)
    return {"key": "rounds_used", "score": rounds}
