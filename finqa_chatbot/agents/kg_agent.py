"""KGAgent — extracts KG triplets from financial documents (Paper 2)."""

from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ..config import get_settings
from ..schema import LogEntry, EntryType, KGTriplet
from ..graph.state import GraphState
from ..prompts.kg_extraction import KG_EXTRACTION_SYSTEM, KG_EXTRACTION_FEW_SHOT
from ..prompts.system import format_context


def _parse_triplets_response(text: str) -> list[dict]:
    """Best-effort parse of LLM JSON response to triplet dicts."""
    text = text.strip()
    # Try to find JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: try entire text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []


def _dict_to_triplet(d: dict) -> KGTriplet:
    """Convert a dict from the LLM to a KGTriplet dataclass."""
    return KGTriplet(
        subject=d.get("subject", ""),
        relation=d.get("relation", ""),
        object=d.get("object", ""),
        entity_type=d.get("entity_type", ""),
        company=d.get("company", ""),
        period=d.get("period", ""),
        value=d.get("value"),
        unit=d.get("unit", ""),
    )


def kg_agent_node(state: GraphState) -> dict:
    """Extract KG triplets via LLM and emit KG_TRIPLET log entries."""
    settings = get_settings()
    entry = state.get("entry", {})
    question = state.get("question", "")

    context = format_context(entry)

    # Build messages with few-shot
    messages = [SystemMessage(content=KG_EXTRACTION_SYSTEM)]
    for ex in KG_EXTRACTION_FEW_SHOT:
        messages.append(HumanMessage(
            content=f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}"
        ))
        messages.append(AIMessage(content=json.dumps(ex["triplets"], indent=2)))

    messages.append(HumanMessage(
        content=f"Context:\n{context}\n\nQuestion: {question}"
    ))

    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )

    try:
        response = llm.invoke(messages)
        raw_triplets = _parse_triplets_response(response.content)
    except Exception as e:
        return {"log": [LogEntry(
            agent="KGAgent",
            entry_type=EntryType.KG_TRIPLET,
            content=f"KG extraction failed: {e}",
        )]}

    entries: list[LogEntry] = []
    for td in raw_triplets:
        triplet = _dict_to_triplet(td)
        entries.append(LogEntry(
            agent="KGAgent",
            entry_type=EntryType.KG_TRIPLET,
            content=triplet.to_text(),
            metadata={
                "subject": triplet.subject,
                "relation": triplet.relation,
                "object": triplet.object,
                "period": triplet.period,
                "value": triplet.value,
                "unit": triplet.unit,
                "entity_type": triplet.entity_type,
            },
        ))

    if not entries:
        entries.append(LogEntry(
            agent="KGAgent",
            entry_type=EntryType.KG_TRIPLET,
            content="No triplets extracted",
        ))

    return {"log": entries}
