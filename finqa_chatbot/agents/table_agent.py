"""TableAgent — extracts LOOKUP entries from table rows."""

from __future__ import annotations

import re

from ..schema import LogEntry, EntryType
from ..graph.state import GraphState


def _extract_numeric(cell: str) -> str | None:
    """Extract a numeric value from a table cell, if present."""
    cell = cell.strip().replace("$", "").replace(",", "")
    # Match numbers like 1234, 12.34, 12.34%, (12.34)
    m = re.search(r'-?[\d]+\.?[\d]*%?', cell)
    return m.group(0) if m else None


def table_agent_node(state: GraphState) -> dict:
    """Parse the table and emit LOOKUP log entries for relevant cells.

    Scans every cell that contains a numeric value and creates a LOOKUP
    entry with row name, column header, value, and source index.
    """
    table = state.get("table", [])
    question = state.get("question", "").lower()
    entries: list[LogEntry] = []

    if not table or len(table) < 2:
        entries.append(LogEntry(
            agent="TableAgent",
            entry_type=EntryType.LOOKUP,
            content="No table data available",
        ))
        return {"log": entries}

    header = table[0]

    for row_idx, row in enumerate(table[1:], start=1):
        row_name = row[0] if row else ""
        for col_idx, cell in enumerate(row[1:], start=1):
            col_name = header[col_idx] if col_idx < len(header) else ""
            numeric = _extract_numeric(cell)
            if numeric is not None:
                entries.append(LogEntry(
                    agent="TableAgent",
                    entry_type=EntryType.LOOKUP,
                    content=f"{row_name} | {col_name} = {cell}",
                    metadata={
                        "row_name": row_name,
                        "column": col_name,
                        "value": cell.strip(),
                        "numeric": numeric,
                        "source_ind": f"table_{row_idx}",
                    },
                ))

    if not entries:
        entries.append(LogEntry(
            agent="TableAgent",
            entry_type=EntryType.LOOKUP,
            content="No numeric values found in table",
        ))

    return {"log": entries}
