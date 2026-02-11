"""Shared data types: LogEntry, KGTriplet, GraphState."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, TypedDict


# ── Log entry types (DeALOG shared-log protocol) ───────────────────────

class EntryType(str, Enum):
    LOOKUP = "LOOKUP"           # table data
    QUOTE = "QUOTE"             # text passages
    KG_TRIPLET = "KG_TRIPLET"   # knowledge graph
    SUMMARY = "SUMMARY"         # generated program
    ANSWER = "ANSWER"           # execution result
    FLAG = "FLAG"               # verification
    OK = "OK"                   # verification


@dataclass
class LogEntry:
    """A single entry in the DeALOG shared log."""

    agent: str
    entry_type: EntryType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Human-readable representation for prompt inclusion."""
        meta_str = ""
        if self.metadata:
            meta_parts = [f"{k}={v}" for k, v in self.metadata.items()]
            meta_str = f" [{', '.join(meta_parts)}]"
        return f"[{self.agent}|{self.entry_type.value}] {self.content}{meta_str}"


# ── KG triplet (Paper 2: Structure First, Reason Next) ────────────────

@dataclass
class KGTriplet:
    """A knowledge-graph triplet extracted from financial documents."""

    subject: str  # e.g. "Revenue:AppleInc" or "Revenue"
    relation: str  # e.g. "HAS_VALUE_IN_2017"
    object: str  # e.g. "5829 million USD"
    entity_type: str = ""  # e.g. "financial_metric"
    company: str = ""
    period: str = ""  # e.g. "2017"
    value: float | None = None
    unit: str = ""

    def to_text(self) -> str:
        return f"({self.subject}, {self.relation}, {self.object})"


# ── LangGraph state ───────────────────────────────────────────────────

class GraphState(TypedDict):
    """LangGraph state for the DeALOG multi-agent system.

    The `log` field uses an append-reducer: each node returns new entries
    that get auto-appended via ``operator.add``.
    """

    # Input (set once at init)
    entry: dict  # raw FinQA dataset entry
    question: str
    table: list[list[str]]
    pre_text: list[str]
    post_text: list[str]

    # DeALOG shared log (append-only)
    log: Annotated[list[LogEntry], operator.add]

    # Scheduler state
    round_number: int
    active_agents: list[str]  # which agents to run this round
    max_rounds: int

    # Summarizer output
    candidate_programs: list[str]  # raw program strings from self-consistency
    selected_program: str  # chosen program after voting
    program_tokens: list[str]  # tokenized for executor

    # Executor output
    exe_result: Any  # numerical result or "n/a"
    exe_invalid: bool

    # Verification
    verification_status: str  # "OK", "FLAG", or ""
    flag_targets: list[str]  # agents to re-engage on FLAG

    # Final output
    final_answer: Any
