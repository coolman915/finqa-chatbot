"""Main pipeline — load data, run graph, collect results."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import get_settings
from .graph.workflow import build_graph
from .graph.callbacks import FinQATracingCallback
from .dsl.parser import parse_program_to_tokens


def load_dataset(split: str = "dev") -> list[dict]:
    """Load a FinQA dataset split."""
    settings = get_settings()
    path = settings.dataset_dir / f"{split}.json"
    with open(path) as f:
        return json.load(f)


def run_single(entry: dict, graph=None) -> dict[str, Any]:
    """Run the full DeALOG pipeline on a single FinQA entry.

    Returns a prediction dict with ``id``, ``predicted`` (token list),
    ``raw_program``, ``exe_result``, and ``rounds_used``.
    """
    if graph is None:
        graph = build_graph()

    settings = get_settings()
    entry_id = entry.get("id", "unknown")

    initial_state = {
        "entry": entry,
        "question": "",
        "table": [],
        "pre_text": [],
        "post_text": [],
        "log": [],
        "round_number": 0,
        "active_agents": [],
        "max_rounds": settings.max_rounds,
        "candidate_programs": [],
        "selected_program": "",
        "program_tokens": [],
        "exe_result": None,
        "exe_invalid": False,
        "verification_status": "",
        "flag_targets": [],
        "final_answer": None,
    }

    callback = FinQATracingCallback(entry_id=entry_id)
    config = {
        "callbacks": [callback],
        "metadata": {"entry_id": entry_id},
    }

    try:
        result = graph.invoke(initial_state, config=config)
        program = result.get("selected_program", "")
        tokens = parse_program_to_tokens(program) if program else ["EOF"]
        return {
            "id": entry_id,
            "predicted": tokens,
            "raw_program": program,
            "exe_result": result.get("exe_result", "n/a"),
            "rounds_used": result.get("round_number", 1),
            "verification_status": result.get("verification_status", ""),
            "final_answer": result.get("final_answer", "n/a"),
        }
    except Exception as e:
        return {
            "id": entry_id,
            "predicted": ["EOF"],
            "raw_program": "",
            "exe_result": "n/a",
            "rounds_used": 0,
            "error": str(e),
        }


def run_batch(
    split: str = "dev",
    max_examples: int | None = None,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Run the pipeline on a full dataset split."""
    data = load_dataset(split)
    if max_examples:
        data = data[:max_examples]

    graph = build_graph()
    predictions: list[dict] = []
    start = time.time()

    print(f"Running on {len(data)} examples from {split} split...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_single, entry, graph): entry["id"]
            for entry in data
        }
        done = 0
        for future in as_completed(futures):
            result = future.result()
            predictions.append(result)
            done += 1
            if done % 20 == 0 or done == len(data):
                elapsed = time.time() - start
                print(f"  Progress: {done}/{len(data)} ({elapsed:.1f}s)")

    # Sort by original order
    id_order = {e["id"]: i for i, e in enumerate(data)}
    predictions.sort(key=lambda x: id_order.get(x["id"], 0))

    return predictions
