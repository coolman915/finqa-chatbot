"""Fact extraction from FinQA entries."""

from __future__ import annotations


def table_row_to_text(header: list[str], row: list[str]) -> str:
    """Convert a table row to natural-language text."""
    res = ""
    if header[0]:
        res += header[0] + " "
    for head, cell in zip(header[1:], row[1:]):
        res += "the " + row[0] + " of " + head + " is " + cell + " ; "
    return res.strip()


def get_all_facts(entry: dict) -> list[tuple[str, str]]:
    """Return ``(ind_key, text)`` for every sentence and table row."""
    facts: list[tuple[str, str]] = []
    all_text = entry["pre_text"] + entry["post_text"]
    for i, sent in enumerate(all_text):
        facts.append(("text_" + str(i), sent))
    header = entry["table"][0] if entry["table"] else []
    for i, row in enumerate(entry["table"]):
        row_text = table_row_to_text(header, row) if i > 0 else " ".join(row)
        facts.append(("table_" + str(i), row_text))
    return facts
