"""Unit tests for each agent's log entries."""

import pytest

from finqa_chatbot.schema import EntryType
from finqa_chatbot.agents.table_agent import table_agent_node
from finqa_chatbot.agents.context_agent import context_agent_node


def test_table_agent_produces_lookups(sample_entry):
    """TableAgent should produce LOOKUP entries from table data."""
    state = {
        "entry": sample_entry,
        "question": sample_entry["qa"]["question"],
        "table": sample_entry["table"],
        "pre_text": sample_entry.get("pre_text", []),
        "post_text": sample_entry.get("post_text", []),
    }
    result = table_agent_node(state)
    log = result["log"]
    assert len(log) > 0
    assert all(e.entry_type == EntryType.LOOKUP for e in log)
    assert all(e.agent == "TableAgent" for e in log)

    # Should have metadata with row_name and value
    for entry in log:
        assert "row_name" in entry.metadata
        assert "value" in entry.metadata


def test_table_agent_empty_table():
    """TableAgent should handle empty tables gracefully."""
    state = {"table": [], "question": "test", "entry": {}}
    result = table_agent_node(state)
    assert len(result["log"]) == 1


def test_context_agent_produces_quotes(sample_entry_with_text):
    """ContextAgent should produce QUOTE entries from text."""
    state = {
        "entry": sample_entry_with_text,
        "question": sample_entry_with_text["qa"]["question"],
        "table": sample_entry_with_text["table"],
        "pre_text": sample_entry_with_text.get("pre_text", []),
        "post_text": sample_entry_with_text.get("post_text", []),
    }
    result = context_agent_node(state)
    log = result["log"]
    assert len(log) > 0
    assert all(e.entry_type == EntryType.QUOTE for e in log)
    assert all(e.agent == "ContextAgent" for e in log)


def test_context_agent_no_text():
    """ContextAgent should handle entries with no text."""
    state = {
        "entry": {"pre_text": [], "post_text": []},
        "question": "test",
        "table": [],
        "pre_text": [],
        "post_text": [],
    }
    result = context_agent_node(state)
    assert len(result["log"]) >= 1
