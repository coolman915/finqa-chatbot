"""Tests for the verification agent's structural checks."""

import pytest

from finqa_chatbot.schema import LogEntry, EntryType
from finqa_chatbot.agents.verification_agent import (
    _extract_program_literals,
    _check_evidence_grounding,
    _check_unit_consistency,
    _check_temporal_consistency,
)


def test_extract_program_literals():
    lits = _extract_program_literals("subtract(11503, 10815)")
    assert "11503" in lits
    assert "10815" in lits


def test_extract_program_literals_with_refs():
    lits = _extract_program_literals("subtract(7.0, 6.3), divide(#0, 6.3)")
    assert "7.0" in lits
    assert "6.3" in lits
    # #0 is a ref, not a literal
    assert "#0" not in lits


def test_evidence_grounding_pass():
    log = [
        LogEntry(
            agent="TableAgent", entry_type=EntryType.LOOKUP,
            content="income | 2017 = 11503",
            metadata={"numeric": "11503"},
        ),
        LogEntry(
            agent="TableAgent", entry_type=EntryType.LOOKUP,
            content="income | 2016 = 10815",
            metadata={"numeric": "10815"},
        ),
    ]
    grounded, missing = _check_evidence_grounding("subtract(11503, 10815)", log)
    assert grounded
    assert missing == []


def test_evidence_grounding_fail():
    log = [
        LogEntry(
            agent="TableAgent", entry_type=EntryType.LOOKUP,
            content="income | 2017 = 11503",
            metadata={"numeric": "11503"},
        ),
    ]
    grounded, missing = _check_evidence_grounding("subtract(11503, 10815)", log)
    assert not grounded
    assert "10815" in missing


def test_unit_consistency_ok():
    ok, issue = _check_unit_consistency("subtract(27.5%, 27.3%)")
    assert ok


def test_unit_consistency_fail():
    ok, issue = _check_unit_consistency("subtract(100, 27.3%)")
    # Note: this is mixing absolute and %, but both are non-ref literals
    # The check should catch this
    # Actually our parser sees "100" and "27.3%" — one has % and one doesn't
    # This should fail
    assert not ok


def test_temporal_consistency_ok():
    log = [
        LogEntry(
            agent="KGAgent", entry_type=EntryType.KG_TRIPLET,
            content="(Rev, HAS_VALUE_IN_2017, 100)",
            metadata={"period": "2017"},
        ),
    ]
    ok, issue = _check_temporal_consistency("divide(100, 50)", log, "revenue in 2017?")
    assert ok


def test_temporal_consistency_no_kg():
    """No KG triplets should always pass."""
    ok, issue = _check_temporal_consistency("divide(100, 50)", [], "revenue in 2017?")
    assert ok
