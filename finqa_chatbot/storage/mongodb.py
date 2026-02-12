"""MongoDB storage backend for FinQA dataset, evaluation runs, and predictions."""

from __future__ import annotations

import logging
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from ..config import get_settings
from ..dsl.parser import parse_program_to_tokens
from ..evaluation.official import _relaxed_equal, relaxed_equal_program

logger = logging.getLogger(__name__)


class MongoStore:
    """Thin wrapper around pymongo for FinQA collections."""

    def __init__(self, uri: str, database: str = "finqa"):
        self._client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self._db = self._client[database]
        self.dataset = self._db["dataset"]
        self.runs = self._db["runs"]
        self.predictions = self._db["predictions"]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the server is reachable."""
        try:
            self._client.admin.command("ping")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError):
            return False

    def ensure_indexes(self) -> None:
        """Create indexes (idempotent)."""
        self.dataset.create_index("split")
        self.dataset.create_index("filename")

        self.runs.create_index("run_id", unique=True)
        self.runs.create_index([("started_at", DESCENDING)])
        self.runs.create_index([("results_official.exe_acc", DESCENDING)])

        self.predictions.create_index(
            [("run_id", ASCENDING), ("entry_id", ASCENDING)], unique=True
        )
        self.predictions.create_index("entry_id")
        self.predictions.create_index(
            [("run_id", ASCENDING), ("exe_correct", ASCENDING)]
        )

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def insert_dataset_entries(self, entries: list[dict], split: str) -> int:
        """Bulk upsert dataset entries. Returns number of upserted/updated docs."""
        ops = []
        for idx, entry in enumerate(entries):
            qa = entry.get("qa", {})
            doc = {
                "_id": entry["id"],
                "split": split,
                "list_number": idx,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "program": qa.get("program", ""),
                "exe_ans": qa.get("exe_ans"),
                "steps": qa.get("steps", []),
                "table": entry.get("table", []),
                "pre_text": entry.get("pre_text", []),
                "post_text": entry.get("post_text", []),
                "filename": entry.get("filename", ""),
            }
            ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))
        if ops:
            result = self.dataset.bulk_write(ops, ordered=False)
            return result.upserted_count + result.modified_count
        return 0

    def get_dataset(self, split: str) -> list[dict]:
        """Reconstruct original entry format from MongoDB docs, sorted by list_number."""
        entries = []
        for doc in self.dataset.find({"split": split}).sort("list_number", 1):
            entries.append({
                "id": doc["_id"],
                "list_number": doc.get("list_number", 0),
                "qa": {
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer", ""),
                    "program": doc.get("program", ""),
                    "exe_ans": doc.get("exe_ans"),
                    "steps": doc.get("steps", []),
                },
                "table": doc.get("table", []),
                "pre_text": doc.get("pre_text", []),
                "post_text": doc.get("post_text", []),
                "filename": doc.get("filename", ""),
            })
        return entries

    def drop_dataset(self, split: str | None = None) -> int:
        """Drop dataset entries. If split is None, drop all."""
        if split:
            result = self.dataset.delete_many({"split": split})
        else:
            result = self.dataset.delete_many({})
        return result.deleted_count

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        split: str,
        num_examples: int,
        config: dict[str, Any],
    ) -> str:
        """Insert a new run document. Returns the run_id."""
        git_hash = _get_git_hash()
        doc = {
            "run_id": run_id,
            "split": split,
            "num_examples": num_examples,
            "config": config,
            "started_at": datetime.now(timezone.utc),
            "finished_at": None,
            "duration_seconds": None,
            "results_official": None,
            "results_extended": None,
            "git_hash": git_hash,
            "hostname": platform.node(),
        }
        self.runs.insert_one(doc)
        return run_id

    def finish_run(
        self,
        run_id: str,
        official: dict[str, Any],
        extended: dict[str, Any],
        duration_seconds: float,
    ) -> None:
        """Update a run with final results."""
        self.runs.update_one(
            {"run_id": run_id},
            {"$set": {
                "finished_at": datetime.now(timezone.utc),
                "duration_seconds": round(duration_seconds, 1),
                "results_official": official,
                "results_extended": extended,
            }},
        )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def insert_prediction(
        self,
        run_id: str,
        prediction: dict[str, Any],
        gold_entry: dict[str, Any],
        split: str = "",
        list_number: int | None = None,
    ) -> None:
        """Insert a single prediction with inline correctness evaluation."""
        try:
            gold_ans = gold_entry["qa"]["exe_ans"]
            gold_prog = gold_entry["qa"].get("program", "")
            text_answer = gold_entry["qa"].get("answer", "")
            pred_prog = prediction.get("raw_program", "")
            pred_tokens = parse_program_to_tokens(pred_prog) if pred_prog else ["EOF"]
            gold_tokens = parse_program_to_tokens(gold_prog) if gold_prog else ["EOF"]

            exe_correct = _relaxed_equal(prediction.get("exe_result"), gold_ans, answer=text_answer)
            prog_correct = relaxed_equal_program(gold_tokens, pred_tokens)

            doc = {
                "run_id": run_id,
                "entry_id": prediction["id"],
                "split": split,
                "list_number": list_number,
                "raw_program": pred_prog,
                "exe_result": prediction.get("exe_result"),
                "rounds_used": prediction.get("rounds_used", 0),
                "final_answer": prediction.get("final_answer"),
                "verification_status": prediction.get("verification_status", ""),
                "error": prediction.get("error"),
                "exe_correct": exe_correct,
                "prog_correct": prog_correct,
                "gold_program": gold_prog,
                "gold_answer": gold_ans,
                "answer": text_answer,
            }
            self.predictions.update_one(
                {"run_id": run_id, "entry_id": prediction["id"]},
                {"$set": doc},
                upsert=True,
            )
        except Exception:
            logger.exception("Failed to insert prediction for %s", prediction.get("id"))

    def insert_predictions_batch(
        self,
        run_id: str,
        predictions: list[dict[str, Any]],
        gold_data: list[dict[str, Any]],
        split: str = "",
    ) -> int:
        """Batch insert predictions. Returns count of upserted docs."""
        gold_dict = {e["id"]: e for e in gold_data}
        # Build list_number lookup from original data order
        id_to_idx = {e["id"]: i for i, e in enumerate(gold_data)}
        ops = []
        for pred in predictions:
            entry = gold_dict.get(pred["id"])
            if entry is None:
                continue
            gold_ans = entry["qa"]["exe_ans"]
            gold_prog = entry["qa"].get("program", "")
            text_answer = entry["qa"].get("answer", "")
            pred_prog = pred.get("raw_program", "")
            pred_tokens = parse_program_to_tokens(pred_prog) if pred_prog else ["EOF"]
            gold_tokens = parse_program_to_tokens(gold_prog) if gold_prog else ["EOF"]

            doc = {
                "run_id": run_id,
                "entry_id": pred["id"],
                "split": split,
                "list_number": id_to_idx.get(pred["id"]),
                "raw_program": pred_prog,
                "exe_result": pred.get("exe_result"),
                "rounds_used": pred.get("rounds_used", 0),
                "final_answer": pred.get("final_answer"),
                "verification_status": pred.get("verification_status", ""),
                "error": pred.get("error"),
                "exe_correct": _relaxed_equal(pred.get("exe_result"), gold_ans, answer=text_answer),
                "prog_correct": relaxed_equal_program(gold_tokens, pred_tokens),
                "gold_program": gold_prog,
                "gold_answer": gold_ans,
                "answer": text_answer,
            }
            ops.append(UpdateOne(
                {"run_id": run_id, "entry_id": pred["id"]},
                {"$set": doc},
                upsert=True,
            ))
        if ops:
            result = self.predictions.bulk_write(ops, ordered=False)
            return result.upserted_count + result.modified_count
        return 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_recent_runs(self, limit: int = 20) -> list[dict]:
        """Return recent runs, newest first."""
        return list(
            self.runs.find({}, {"_id": 0})
            .sort("started_at", DESCENDING)
            .limit(limit)
        )

    def get_run(self, run_id: str) -> dict | None:
        """Return a single run by run_id."""
        return self.runs.find_one({"run_id": run_id}, {"_id": 0})

    def get_predictions(self, run_id: str) -> list[dict]:
        """Return all predictions for a run."""
        return list(
            self.predictions.find({"run_id": run_id}, {"_id": 0})
        )

    def get_failures(self, run_id: str) -> list[dict]:
        """Return predictions where exe_correct is False."""
        return list(
            self.predictions.find(
                {"run_id": run_id, "exe_correct": False}, {"_id": 0}
            )
        )

    def get_entry_history(self, entry_id: str) -> list[dict]:
        """Return all predictions for a given entry across runs."""
        return list(
            self.predictions.find({"entry_id": entry_id}, {"_id": 0})
            .sort("run_id", ASCENDING)
        )

    def get_best_run(self) -> dict | None:
        """Return the run with the highest exe_acc."""
        cursor = (
            self.runs.find(
                {"results_official": {"$ne": None}}, {"_id": 0}
            )
            .sort("results_official.exe_acc", DESCENDING)
            .limit(1)
        )
        runs = list(cursor)
        return runs[0] if runs else None


def _get_git_hash() -> str:
    """Get current git short hash, or empty string on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def get_mongo_store() -> MongoStore | None:
    """Return a connected MongoStore, or None if not configured / unreachable."""
    settings = get_settings()
    if not settings.mongodb_uri:
        return None
    try:
        store = MongoStore(settings.mongodb_uri, settings.mongodb_database)
        if not store.ping():
            logger.warning("MongoDB not reachable at %s", settings.mongodb_uri)
            return None
        store.ensure_indexes()
        return store
    except Exception:
        logger.warning("Failed to connect to MongoDB", exc_info=True)
        return None
