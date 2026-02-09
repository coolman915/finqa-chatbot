"""End-to-end graph tests (require API key — skipped if not available)."""

import os
import pytest

from finqa_chatbot.graph.workflow import build_graph
from finqa_chatbot.graph.scheduler import init_node, scheduler_node
from finqa_chatbot.schema import EntryType


def test_graph_compiles():
    """The graph should compile without errors."""
    graph = build_graph()
    assert graph is not None


def test_init_node(sample_entry):
    """Init node should populate state from entry."""
    state = {"entry": sample_entry, "max_rounds": 3, "log": []}
    result = init_node(state)
    assert result["question"] == sample_entry["qa"]["question"]
    assert result["table"] == sample_entry["table"]
    assert result["round_number"] == 0


def test_scheduler_round_1():
    """Scheduler should activate all retrieval agents on round 1."""
    state = {
        "round_number": 0,
        "flag_targets": [],
        "active_agents": [],
    }
    result = scheduler_node(state)
    assert result["round_number"] == 1
    assert "table_agent" in result["active_agents"]
    assert "context_agent" in result["active_agents"]
    assert "kg_agent" in result["active_agents"]


def test_scheduler_re_engagement():
    """Scheduler should only activate targeted agents after FLAG."""
    state = {
        "round_number": 1,
        "flag_targets": ["table_agent", "summarizer"],
        "active_agents": [],
    }
    result = scheduler_node(state)
    assert result["round_number"] == 2
    assert result["active_agents"] == ["table_agent", "summarizer"]


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live API test",
)
def test_full_graph_single_entry(sample_entry_with_text):
    """Full graph run on a single entry (requires API key)."""
    graph = build_graph()
    initial_state = {
        "entry": sample_entry_with_text,
        "question": "",
        "table": [],
        "pre_text": [],
        "post_text": [],
        "log": [],
        "round_number": 0,
        "active_agents": [],
        "max_rounds": 2,
        "candidate_programs": [],
        "selected_program": "",
        "program_tokens": [],
        "exe_result": None,
        "exe_invalid": False,
        "verification_status": "",
        "flag_targets": [],
        "final_answer": None,
    }

    result = graph.invoke(initial_state)
    assert result["final_answer"] is not None
    assert result["round_number"] >= 1
    # Check log has entries from multiple agents
    agent_names = {e.agent for e in result["log"]}
    assert "TableAgent" in agent_names
    assert "ContextAgent" in agent_names
