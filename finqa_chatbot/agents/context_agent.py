"""ContextAgent — retrieves QUOTE entries from text passages."""

from __future__ import annotations

import re

from ..schema import LogEntry, EntryType
from ..graph.state import GraphState
from ..retrieval.facts import get_all_facts
from ..retrieval.tfidf import tfidf_retrieve


def _has_numeric(text: str) -> bool:
    """Check if text contains a number."""
    return bool(re.search(r'\d', text))


def context_agent_node(state: GraphState) -> dict:
    """Retrieve relevant text passages and emit QUOTE log entries.

    Uses TF-IDF retrieval to find the most relevant facts from
    pre_text and post_text, then creates QUOTE entries.
    """
    question = state.get("question", "")
    entry = state.get("entry", {})
    entries: list[LogEntry] = []

    # Gather all text facts (not table facts for this agent)
    all_text = entry.get("pre_text", []) + entry.get("post_text", [])
    text_facts: list[tuple[str, str]] = []
    for i, sent in enumerate(all_text):
        sent = sent.strip()
        if sent and sent != ".":
            text_facts.append((f"text_{i}", sent))

    if not text_facts:
        entries.append(LogEntry(
            agent="ContextAgent",
            entry_type=EntryType.QUOTE,
            content="No text passages available",
        ))
        return {"log": entries}

    # Retrieve top passages
    retrieved = tfidf_retrieve(question, text_facts, topn=5)

    for ind_key, text in retrieved:
        # Extract numbers mentioned in the passage
        numbers = re.findall(r'[\$]?[\d,]+\.?\d*\s*%?', text)
        numbers = [n.strip() for n in numbers if n.strip()]

        entries.append(LogEntry(
            agent="ContextAgent",
            entry_type=EntryType.QUOTE,
            content=text,
            metadata={
                "source_ind": ind_key,
                "span": text[:100],
                "numbers": numbers,
            },
        ))

    return {"log": entries}
