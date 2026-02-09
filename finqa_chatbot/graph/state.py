"""LangGraph state definition with append-reducer for the shared log."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from ..schema import LogEntry


class GraphState(TypedDict):
    """Full state for the DeALOG multi-agent graph.

    The ``log`` field uses ``operator.add`` as a reducer so each node
    can return new log entries that get auto-appended.
    """

    # ── Input (set once at init) ────────────────────────────────────────
    entry: dict
    question: str
    table: list[list[str]]
    pre_text: list[str]
    post_text: list[str]

    # ── DeALOG shared log (append-only) ─────────────────────────────────
    log: Annotated[list[LogEntry], operator.add]

    # ── Scheduler state ─────────────────────────────────────────────────
    round_number: int
    active_agents: list[str]
    max_rounds: int

    # ── Summarizer output ───────────────────────────────────────────────
    candidate_programs: list[str]
    selected_program: str
    program_tokens: list[str]

    # ── Executor output ─────────────────────────────────────────────────
    exe_result: Any
    exe_invalid: bool

    # ── Verification ────────────────────────────────────────────────────
    verification_status: str  # "OK" | "FLAG" | ""
    flag_targets: list[str]

    # ── Final ───────────────────────────────────────────────────────────
    final_answer: Any
