"""Step-level tracing callback for per-node observability (MongoDB traces)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


GRAPH_NODES = frozenset({
    "init", "scheduler", "table_agent", "context_agent",
    "summarizer", "executor", "verifier", "finalize",
})


class FinQATracingCallback(BaseCallbackHandler):
    """Capture per-node step traces during a single graph invocation.

    Each ``run_single()`` call creates its own callback instance, so all
    internal state is thread-local by construction.

    After ``graph.invoke()`` completes, call ``get_steps()`` to retrieve
    the collected step dicts ready for MongoDB insertion.
    """

    def __init__(
        self,
        entry_id: str = "",
        run_id: str = "",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.entry_id = entry_id
        self.run_id = run_id

        self._steps: list[dict[str, Any]] = []
        self._active_nodes: dict[UUID, dict[str, Any]] = {}  # run_id → step dict
        self._llm_to_node: dict[UUID, UUID] = {}  # llm run_id → parent node run_id
        self._current_round: int = 0
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Chain (node) events
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name", "")
        if name not in GRAPH_NODES:
            return

        # Track round number from scheduler outputs
        if name == "scheduler" and isinstance(inputs, dict):
            rn = inputs.get("round_number")
            if isinstance(rn, int):
                self._current_round = rn

        step = {
            "run_id": self.run_id,
            "entry_id": self.entry_id,
            "node_name": name,
            "round_number": self._current_round,
            "step_index": self._step_counter,
            "started_at": datetime.now(timezone.utc),
            "finished_at": None,
            "duration_ms": None,
            "outputs": {},
            "llm_usage": None,
            "error": None,
            "_chain_run_id": run_id,  # internal, stripped before insert
        }
        self._step_counter += 1
        self._active_nodes[run_id] = step

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._active_nodes.pop(run_id, None)
        if step is None:
            return

        now = datetime.now(timezone.utc)
        step["finished_at"] = now
        step["duration_ms"] = round(
            (now - step["started_at"]).total_seconds() * 1000, 1
        )
        step["outputs"] = self._extract_outputs(step["node_name"], outputs)

        # Update round from scheduler output
        if step["node_name"] == "scheduler" and isinstance(outputs, dict):
            rn = outputs.get("round_number")
            if isinstance(rn, int):
                self._current_round = rn
                step["round_number"] = rn

        self._steps.append(step)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._active_nodes.pop(run_id, None)
        if step is None:
            return

        now = datetime.now(timezone.utc)
        step["finished_at"] = now
        step["duration_ms"] = round(
            (now - step["started_at"]).total_seconds() * 1000, 1
        )
        step["error"] = {
            "type": type(error).__name__,
            "message": str(error)[:500],
        }
        self._steps.append(step)

    # ------------------------------------------------------------------
    # LLM events (attribute tokens to parent node)
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is not None:
            # Walk up: parent might be a sub-chain inside a graph node
            node_id = self._find_ancestor_node(parent_run_id)
            if node_id is not None:
                self._llm_to_node[run_id] = node_id

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        node_id = self._llm_to_node.pop(run_id, None)
        if node_id is None:
            return
        step = self._active_nodes.get(node_id)
        if step is None:
            return

        # Extract token usage from LLMResult
        usage = {}
        model = ""
        if response.llm_output:
            tu = response.llm_output.get("token_usage", {})
            usage = {
                "prompt_tokens": tu.get("prompt_tokens", 0),
                "completion_tokens": tu.get("completion_tokens", 0),
                "total_tokens": tu.get("total_tokens", 0),
            }
            model = response.llm_output.get("model_name", "")

        # Also check per-generation usage_metadata (langchain-openai style)
        if not usage.get("total_tokens"):
            for gen_list in response.generations:
                for gen in gen_list:
                    um = getattr(gen, "usage_metadata", None) or {}
                    if isinstance(um, dict) and um.get("total_tokens"):
                        usage = {
                            "prompt_tokens": um.get("input_tokens", 0),
                            "completion_tokens": um.get("output_tokens", 0),
                            "total_tokens": um.get("total_tokens", 0),
                        }
                        break

        # Accumulate onto the step's llm_usage
        if step["llm_usage"] is None:
            step["llm_usage"] = {
                "model": model,
                "num_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
            }
        lu = step["llm_usage"]
        lu["num_calls"] += 1
        lu["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
        lu["total_completion_tokens"] += usage.get("completion_tokens", 0)
        lu["total_tokens"] += usage.get("total_tokens", 0)
        if model and not lu["model"]:
            lu["model"] = model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_ancestor_node(self, run_id: UUID) -> UUID | None:
        """Find the closest ancestor that is an active graph node."""
        if run_id in self._active_nodes:
            return run_id
        # Check if this run_id is already mapped to a node
        mapped = self._llm_to_node.get(run_id)
        if mapped and mapped in self._active_nodes:
            return mapped
        return None

    def get_steps(self) -> list[dict[str, Any]]:
        """Return collected steps (call after graph.invoke completes)."""
        return list(self._steps)

    @staticmethod
    def _log_entries_content(log: Any, max_entries: int = 20) -> list[str]:
        """Extract content strings from LogEntry objects or dicts."""
        if not isinstance(log, list):
            return []
        items: list[str] = []
        for entry in log[:max_entries]:
            if hasattr(entry, "content"):
                items.append(entry.content)
            elif isinstance(entry, dict):
                items.append(entry.get("content", str(entry)))
            else:
                items.append(str(entry))
        return items

    @staticmethod
    def _log_entries_metadata(log: Any) -> list[dict]:
        """Extract metadata dicts from LogEntry objects."""
        if not isinstance(log, list):
            return []
        items: list[dict] = []
        for entry in log:
            if hasattr(entry, "metadata") and entry.metadata:
                items.append(entry.metadata)
            elif isinstance(entry, dict) and entry.get("metadata"):
                items.append(entry["metadata"])
        return items

    @classmethod
    def _extract_outputs(cls, node_name: str, outputs: Any) -> dict[str, Any]:
        """Extract debugging-friendly output from each node."""
        if not isinstance(outputs, dict):
            return {}

        log = outputs.get("log", [])

        if node_name == "init":
            q = outputs.get("question", "")
            return {"question": q[:300] if isinstance(q, str) else str(q)[:300]}

        if node_name == "scheduler":
            return {
                "round_number": outputs.get("round_number"),
                "active_agents": outputs.get("active_agents", []),
            }

        if node_name == "table_agent":
            entries = cls._log_entries_content(log, max_entries=30)
            return {
                "num_lookups": len(entries),
                "lookups": entries,
            }

        if node_name == "context_agent":
            entries = cls._log_entries_content(log, max_entries=10)
            metadata = cls._log_entries_metadata(log)
            numbers_found: list[str] = []
            for m in metadata:
                numbers_found.extend(m.get("numbers", []))
            return {
                "num_quotes": len(entries),
                "quotes": entries,
                "numbers_found": numbers_found,
            }

        if node_name == "summarizer":
            candidates = outputs.get("candidate_programs", [])
            log_content = cls._log_entries_content(log)
            return {
                "program": outputs.get("selected_program", ""),
                "num_candidates": len(candidates),
                "candidates": candidates,
                "reasoning": log_content,
            }

        if node_name == "executor":
            program = outputs.get("program_tokens", [])
            log_content = cls._log_entries_content(log)
            return {
                "result": outputs.get("exe_result"),
                "invalid": outputs.get("exe_invalid", False),
                "program_tokens": program,
                "detail": log_content[0] if log_content else "",
            }

        if node_name == "verifier":
            log_content = cls._log_entries_content(log)
            metadata = cls._log_entries_metadata(log)
            issues: list[str] = []
            for m in metadata:
                issues.extend(m.get("issues", []))
                if m.get("issue"):
                    issues.append(m["issue"])
            return {
                "status": outputs.get("verification_status", ""),
                "flag_targets": outputs.get("flag_targets", []),
                "issues": issues,
                "detail": log_content,
            }

        if node_name == "finalize":
            return {
                "final_answer": outputs.get("final_answer"),
                "selected_program": outputs.get("selected_program", ""),
                "used_fallback": "best_program" in outputs
                    and "selected_program" in outputs,
            }

        return {}
