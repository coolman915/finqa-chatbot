"""SummarizingAgent — reads shared log, generates DSL program (temp=0, deterministic)."""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ..config import get_settings
from ..schema import LogEntry, EntryType
from ..graph.state import GraphState
from ..dsl.parser import parse_program_to_tokens
from ..dsl.executor import eval_program
from ..dsl.operations import ALL_OPS
from ..prompts.summarizer import SUMMARIZER_SYSTEM, SUMMARIZER_USER_TEMPLATE, SUMMARIZER_FEW_SHOT


def _format_table(table: list[list[str]]) -> str:
    """Format a table as a markdown string."""
    if not table:
        return "(no table)"
    lines = []
    for row in table:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_log(log: list[LogEntry]) -> str:
    """Format log entries as text for the LLM prompt."""
    if not log:
        return "(no evidence yet)"
    return "\n".join(e.to_text() for e in log)


def _extract_program(text: str) -> str:
    """Extract the DSL program from LLM output.

    Handles both single-line programs and multi-line outputs where each
    step is on a separate line.
    """
    text = text.strip().strip("`").strip()
    if text.startswith("Program:"):
        text = text[len("Program:"):].strip()

    # Collect all lines that contain a DSL operation
    op_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line and any(op + "(" in line for op in ALL_OPS):
            op_lines.append(line)

    if not op_lines:
        return ""

    # If the last op line already contains commas joining multiple steps,
    # it's a complete single-line program — use it directly
    last_line = op_lines[-1]
    op_count_last = sum(1 for op in ALL_OPS if op + "(" in last_line)
    if op_count_last >= len(op_lines) or len(op_lines) == 1:
        return last_line

    # Otherwise, join all operation lines into a single program
    return ", ".join(op_lines)


def summarizer_node(state: GraphState) -> dict:
    """Generate a DSL program using temp=0 deterministic reasoning."""
    settings = get_settings()
    question = state.get("question", "")
    table = state.get("table", [])
    log = state.get("log", [])

    table_str = _format_table(table)
    log_text = _format_log(log)

    user_content = SUMMARIZER_USER_TEMPLATE.format(
        question=question,
        table_str=table_str,
        log_text=log_text,
    )

    # On retry rounds, include previous attempt and error for self-correction
    round_number = state.get("round_number", 1)
    if round_number > 1:
        prev_program = state.get("selected_program", "")
        verification_status = state.get("verification_status", "")
        # Extract issues from the log (latest FLAG entry)
        prev_issues = ""
        for entry in reversed(log):
            if entry.entry_type == EntryType.FLAG:
                prev_issues = entry.content
                break
        if prev_program:
            user_content += (
                f"\n\nPREVIOUS ATTEMPT (round {round_number - 1}): {prev_program}"
                f"\nISSUE: {prev_issues or verification_status}"
                f"\nGenerate a DIFFERENT and CORRECT program."
            )

    # Build messages with few-shot (chain-of-thought reasoning examples)
    messages = [SystemMessage(content=SUMMARIZER_SYSTEM)]
    for ex in SUMMARIZER_FEW_SHOT:
        messages.append(HumanMessage(
            content=f"Question: {ex['question']}\n\nTable:\n{ex['table']}\n\nIdentify the relevant row(s) and values, then write the DSL program:"
        ))
        messages.append(AIMessage(content=ex["reasoning"]))
    messages.append(HumanMessage(content=user_content))

    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )

    program = ""
    try:
        resp = llm.invoke(messages)
        program = _extract_program(resp.content)
    except Exception:
        pass

    log_entries = [LogEntry(
        agent="SummarizingAgent",
        entry_type=EntryType.SUMMARY,
        content=f"Program: {program}" if program else "Failed to generate program",
        metadata={"selected": program},
    )]

    return {
        "candidate_programs": [program] if program else [],
        "selected_program": program,
        "log": log_entries,
    }
