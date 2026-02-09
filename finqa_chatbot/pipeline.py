"""Main pipeline — load data, run graph, collect results."""

from __future__ import annotations

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import get_settings
from .graph.workflow import build_graph
from .graph.callbacks import FinQATracingCallback
from .dsl.parser import parse_program_to_tokens
from .evaluation.official import _relaxed_equal


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
        "best_program": "",
        "best_exe_result": None,
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
    save_path: str | None = None,
    save_every: int = 10,
    data: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Run the pipeline on a full dataset split with incremental saves.

    Args:
        split: Dataset split name.
        max_examples: Limit number of examples (None = all).
        workers: Number of parallel workers.
        save_path: Path to save incremental results (JSON). If None, no
            incremental saving.
        save_every: Save results every N completed examples.
        data: Pre-loaded/sliced dataset (overrides split + max_examples).
    """
    if data is None:
        data = load_dataset(split)
        if max_examples:
            data = data[:max_examples]

    graph = build_graph()
    predictions: list[dict] = []
    lock = threading.Lock()
    start = time.time()
    correct_count = 0
    error_count = 0

    # Load existing results to resume
    completed_ids: set[str] = set()
    if save_path and Path(save_path).exists():
        with open(save_path) as f:
            existing = json.load(f)
        predictions = existing
        completed_ids = {p["id"] for p in existing}
        print(f"  Resuming: {len(completed_ids)} already done")

    remaining = [e for e in data if e["id"] not in completed_ids]
    total = len(data)

    print(f"Running {len(remaining)} examples ({total} total) from {split} split with {workers} workers...")

    def _save():
        if not save_path:
            return
        id_order = {e["id"]: i for i, e in enumerate(data)}
        sorted_preds = sorted(predictions, key=lambda x: id_order.get(x["id"], 0))
        with open(save_path, "w") as f:
            json.dump(sorted_preds, f, indent=2, default=str)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_single, entry, graph): entry
            for entry in remaining
        }
        done = len(completed_ids)
        for future in as_completed(futures):
            entry = futures[future]
            result = future.result()

            # Quick accuracy check (relaxed for const_100 ambiguity)
            gold_ans = entry["qa"]["exe_ans"]
            is_correct = _relaxed_equal(result.get("exe_result"), gold_ans)
            has_error = "error" in result

            with lock:
                predictions.append(result)
                done += 1
                if is_correct:
                    correct_count += 1
                if has_error:
                    error_count += 1

                elapsed = time.time() - start
                rate = (done - len(completed_ids)) / elapsed if elapsed > 0 else 0
                remaining_time = (total - done) / rate if rate > 0 else 0

                if done % save_every == 0 or done == total:
                    acc = correct_count / done if done > 0 else 0
                    print(
                        f"  {done}/{total}  "
                        f"acc={acc:.1%}  "
                        f"err={error_count}  "
                        f"{elapsed:.0f}s elapsed  "
                        f"~{remaining_time:.0f}s remaining  "
                        f"({rate:.1f} ex/s)"
                    )
                    _save()

    # Final save
    _save()

    # Sort by original order
    id_order = {e["id"]: i for i, e in enumerate(data)}
    predictions.sort(key=lambda x: id_order.get(x["id"], 0))

    return predictions
