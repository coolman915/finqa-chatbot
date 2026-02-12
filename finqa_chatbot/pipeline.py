"""Main pipeline — load data, run graph, collect results."""

from __future__ import annotations

import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_settings
from .graph.workflow import build_graph
from .graph.callbacks import FinQATracingCallback
from .dsl.parser import parse_program_to_tokens
from .evaluation.official import _relaxed_equal, relaxed_equal_program
from .storage import get_mongo_store

logger = logging.getLogger(__name__)


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
    llm_eval: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """Run the pipeline on a full dataset split with incremental saves.

    Args:
        split: Dataset split name.
        max_examples: Limit number of examples (None = all).
        workers: Number of parallel workers.
        save_path: Path to save incremental results (JSON). If None, no
            incremental saving.
        save_every: Save results every N completed examples.
        data: Pre-loaded/sliced dataset (overrides split + max_examples).

    Returns:
        Tuple of (predictions, run_id). run_id is None if MongoDB is not available.
    """
    if data is None:
        data = load_dataset(split)
        if max_examples:
            data = data[:max_examples]

    settings = get_settings()

    # MongoDB tracing (opt-in)
    store = get_mongo_store()
    run_id: str | None = None
    if store:
        now = datetime.now(timezone.utc)
        run_id = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{split}_{len(data)}"
        store.create_run(
            run_id=run_id,
            split=split,
            num_examples=len(data),
            config={
                "model": settings.model_name,
                "max_rounds": settings.max_rounds,
                "num_candidates": settings.num_candidates,
                "temperature": settings.temperature,
                "workers": workers,
            },
        )
        logger.info("MongoDB run created: %s", run_id)

    data_dict = {e["id"]: e for e in data}
    id_to_idx = {e["id"]: i for i, e in enumerate(data)}
    graph = build_graph()
    predictions: list[dict] = []
    lock = threading.Lock()
    start = time.time()
    correct_count = 0
    prog_correct_count = 0
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
            text_answer = entry["qa"].get("answer", "")
            is_exe_correct = _relaxed_equal(result.get("exe_result"), gold_ans, answer=text_answer)
            has_error = "error" in result

            # Program accuracy check
            pred_prog = result.get("raw_program", "")
            gold_prog = entry["qa"].get("program", "")
            pred_tokens = parse_program_to_tokens(pred_prog) if pred_prog else ["EOF"]
            gold_tokens = parse_program_to_tokens(gold_prog) if gold_prog else ["EOF"]
            is_prog_correct = relaxed_equal_program(gold_tokens, pred_tokens)

            # LLM evaluation of failures
            llm_eval_result = None
            if llm_eval and (not is_exe_correct or not is_prog_correct):
                from .evaluation.llm_eval import llm_evaluate_prediction
                gold_entry = data_dict.get(result["id"])
                if gold_entry:
                    llm_eval_result = llm_evaluate_prediction(
                        question=gold_entry["qa"]["question"],
                        gold_program=gold_prog,
                        pred_program=pred_prog,
                        gold_answer=gold_ans,
                        pred_answer=result.get("exe_result"),
                        text_answer=text_answer,
                        table=gold_entry.get("table"),
                        pre_text=gold_entry.get("pre_text"),
                        post_text=gold_entry.get("post_text"),
                    )

            # Attach LLM eval results to prediction dict
            if llm_eval_result:
                result["llm_correct"] = llm_eval_result.get("llm_correct")
                result["failure_reason"] = llm_eval_result.get("failure_reason")
                result["llm_explanation"] = llm_eval_result.get("llm_explanation")

            # Insert prediction into MongoDB
            if store and run_id:
                gold_entry = data_dict.get(result["id"])
                if gold_entry:
                    store.insert_prediction(
                        run_id, result, gold_entry,
                        split=split,
                        list_number=id_to_idx.get(result["id"]),
                        llm_eval=llm_eval_result,
                    )

            with lock:
                predictions.append(result)
                done += 1
                if is_exe_correct:
                    correct_count += 1
                if is_prog_correct:
                    prog_correct_count += 1
                if has_error:
                    error_count += 1

                elapsed = time.time() - start
                rate = (done - len(completed_ids)) / elapsed if elapsed > 0 else 0
                remaining_time = (total - done) / rate if rate > 0 else 0

                if done % save_every == 0 or done == total:
                    exe_acc = correct_count / done if done > 0 else 0
                    prog_acc = prog_correct_count / done if done > 0 else 0
                    print(
                        f"  {done}/{total}  "
                        f"exe_acc={exe_acc:.1%}  "
                        f"prog_acc={prog_acc:.1%}  "
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

    return predictions, run_id
