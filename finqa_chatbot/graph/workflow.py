"""LangGraph StateGraph construction and compilation."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .state import GraphState
from .scheduler import (
    init_node,
    scheduler_node,
    route_after_verification,
    route_retrieval_agents,
)
from ..agents.table_agent import table_agent_node
from ..agents.context_agent import context_agent_node
from ..agents.kg_agent import kg_agent_node
from ..agents.summarizer_agent import summarizer_node
from ..agents.verification_agent import verification_node


def _executor_node(state: GraphState) -> dict:
    """Execute the selected DSL program deterministically."""
    from ..dsl.executor import eval_program
    from ..dsl.parser import parse_program_to_tokens
    from ..schema import LogEntry, EntryType

    program_str = state.get("selected_program", "")
    if not program_str:
        return {
            "exe_result": "n/a",
            "exe_invalid": True,
            "log": [LogEntry(
                agent="Executor",
                entry_type=EntryType.ANSWER,
                content="No program to execute",
            )],
        }

    tokens = parse_program_to_tokens(program_str)
    invalid_flag, result = eval_program(tokens, state["table"])

    updates = {
        "program_tokens": tokens,
        "exe_result": result,
        "exe_invalid": bool(invalid_flag),
        "log": [LogEntry(
            agent="Executor",
            entry_type=EntryType.ANSWER,
            content=f"Program: {program_str} → Result: {result}"
                    + (f" (INVALID)" if invalid_flag else ""),
            metadata={"result": result, "invalid": bool(invalid_flag)},
        )],
    }

    # Track best valid program across rounds (fallback on degradation)
    if not invalid_flag and result != "n/a":
        best = state.get("best_program", "")
        if not best:
            updates["best_program"] = program_str
            updates["best_exe_result"] = result

    return updates


def _finalize_node(state: GraphState) -> dict:
    """Set the final answer, falling back to best program if current is invalid."""
    exe_result = state.get("exe_result", "n/a")
    program = state.get("selected_program", "")
    exe_invalid = state.get("exe_invalid", False)

    # Fall back to best program if current round degraded
    if (not program or exe_invalid or exe_result == "n/a"):
        best_prog = state.get("best_program", "")
        best_result = state.get("best_exe_result")
        if best_prog and best_result is not None:
            return {
                "selected_program": best_prog,
                "exe_result": best_result,
                "exe_invalid": False,
                "final_answer": best_result,
            }

    return {"final_answer": exe_result}


def build_graph() -> StateGraph:
    """Construct and return the compiled DeALOG workflow graph."""

    graph = StateGraph(GraphState)

    # ── Add nodes ───────────────────────────────────────────────────────
    graph.add_node("init", init_node)
    graph.add_node("scheduler", scheduler_node)
    graph.add_node("table_agent", table_agent_node)
    graph.add_node("context_agent", context_agent_node)
    graph.add_node("kg_agent", kg_agent_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("executor", _executor_node)
    graph.add_node("verifier", verification_node)
    graph.add_node("finalize", _finalize_node)

    # ── Edges ───────────────────────────────────────────────────────────
    graph.set_entry_point("init")
    graph.add_edge("init", "scheduler")

    # Scheduler fans out to retrieval agents conditionally
    graph.add_conditional_edges(
        "scheduler",
        route_retrieval_agents,
        {
            "table_agent": "table_agent",
            "context_agent": "context_agent",
            "kg_agent": "kg_agent",
        },
    )

    # All retrieval agents converge to summarizer
    graph.add_edge("table_agent", "summarizer")
    graph.add_edge("context_agent", "summarizer")
    graph.add_edge("kg_agent", "summarizer")

    # Summarizer → Executor → Verifier
    graph.add_edge("summarizer", "executor")
    graph.add_edge("executor", "verifier")

    # Verifier routes: OK → finalize → END, FLAG → scheduler loop
    graph.add_conditional_edges(
        "verifier",
        route_after_verification,
        {
            "end": "finalize",
            "scheduler": "scheduler",
        },
    )
    graph.add_edge("finalize", END)

    return graph.compile()
