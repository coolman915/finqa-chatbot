"""Shared test fixtures."""

import json
from pathlib import Path

import pytest


DATASET_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def dev_data():
    """Load the full dev dataset."""
    with open(DATASET_DIR / "dev.json") as f:
        return json.load(f)


@pytest.fixture
def sample_entry(dev_data):
    """Return the first dev entry (Visa payments volume)."""
    return dev_data[0]


@pytest.fixture
def sample_table():
    """A simple table for testing."""
    return [
        ["company", "payments volume ( billions )", "total transactions ( billions )"],
        ["visa", "$ 2457", "50.3"],
        ["mastercard", "1697", "27.0"],
        ["american express", "637", "5.0"],
    ]


@pytest.fixture
def sample_entry_with_text():
    """An entry with both text and table."""
    return {
        "id": "test-entry-1",
        "pre_text": [
            "total net revenue increased $ 693 million , or 11% ( 11 % ) , to $ 7.0 billion ."
        ],
        "post_text": [
            "the increase was driven by organic growth ."
        ],
        "table": [
            ["year", "net revenue"],
            ["2006", "7.0"],
            ["2005", "6.3"],
        ],
        "qa": {
            "question": "what is the percent change in total net revenue from 2005 to 2006?",
            "answer": "0.11111",
            "program": "subtract(7.0, 6.3), divide(#0, 6.3)",
            "exe_ans": 0.11111,
            "steps": [
                {"op": "subtract1-1", "arg1": "7.0", "arg2": "6.3", "res": "0.7"},
                {"op": "divide1-2", "arg1": "#0", "arg2": "6.3", "res": "0.11111"},
            ],
        },
    }
