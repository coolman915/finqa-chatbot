"""DeALOG Scheduler — round management and agent gating."""

from __future__ import annotations

from ..schema import LogEntry, EntryType
from .state import GraphState

ALL_RETRIEVAL_AGENTS = ["table_agent", "context_agent", "kg_agent"]


def init_node(state: GraphState) -> dict:
    """Initialize state from a raw FinQA entry."""
    entry = state["entry"]
    return {
        "question": entry["qa"]["question"],
        "table": entry["table"],
        "pre_text": entry.get("pre_text", []),
        "post_text": entry.get("post_text", []),
        "round_number": 0,
        "active_agents": [],
        "max_rounds": state.get("max_rounds", 3),
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
        "log": [],
    }


def scheduler_node(state: GraphState) -> dict:
    """Determine which agents to activate this round.

    Round 1: all retrieval agents.
    Re-engagement rounds: only agents targeted by the verifier's FLAG.
    """
    round_number = state.get("round_number", 0) + 1
    flag_targets = state.get("flag_targets", [])

    if round_number == 1:
        active = list(ALL_RETRIEVAL_AGENTS)
    elif flag_targets:
        active = list(flag_targets)
    else:
        active = list(ALL_RETRIEVAL_AGENTS)

    return {
        "round_number": round_number,
        "active_agents": active,
        "flag_targets": [],
    }


def should_run_agent(agent_name: str):
    """Return a gate function that checks if *agent_name* is in ``active_agents``."""
    def gate(state: GraphState) -> bool:
        return agent_name in state.get("active_agents", [])
    return gate


# ── Routing functions ───────────────────────────────────────────────────

def route_after_verification(state: GraphState) -> str:
    """Route after verification: END if OK, back to scheduler if FLAG (within limit)."""
    status = state.get("verification_status", "")
    round_number = state.get("round_number", 0)
    max_rounds = state.get("max_rounds", 3)

    if status == "OK" or round_number >= max_rounds:
        return "end"
    elif status == "FLAG":
        return "scheduler"
    else:
        return "end"


def route_retrieval_agents(state: GraphState) -> list[str]:
    """Return the list of agent node names to fan out to."""
    active = state.get("active_agents", [])
    nodes = []
    for agent in active:
        if agent in ALL_RETRIEVAL_AGENTS:
            nodes.append(agent)
    return nodes if nodes else ALL_RETRIEVAL_AGENTS
