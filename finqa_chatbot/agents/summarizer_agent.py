"""SummarizingAgent — reads shared log, generates DSL programs with self-consistency."""

from __future__ import annotations

import asyncio
from collections import Counter

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import get_settings
from ..schema import LogEntry, EntryType
from ..graph.state import GraphState
from ..dsl.parser import parse_program_to_tokens
from ..dsl.executor import eval_program
from ..dsl.operations import ALL_OPS
from ..prompts.summarizer import SUMMARIZER_SYSTEM, SUMMARIZER_USER_TEMPLATE
from ..prompts.system import FEW_SHOT_EXAMPLES


def _format_table(table: list[list[str]]) -> str:
    """Format a table as a markdown string."""
    if not table:
        return "(no table)"
    lines = []
    for i, row in enumerate(table):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_log(log: list[LogEntry]) -> str:
    """Format log entries as text for the LLM prompt."""
    if not log:
        return "(no evidence yet)"
    return "\n".join(e.to_text() for e in log)


def _count_steps(program_str: str) -> int:
    """Count the number of steps in a program string."""
    return len([p for p in program_str.split(",") if any(op + "(" in p for op in ALL_OPS)])


def _select_program(
    candidates: list[str],
    table: list[list[str]],
) -> str:
    """Select the best program via simplicity-weighted majority voting.

    Groups candidates by execution result, then picks the group with
    the most votes, breaking ties by preferring simpler programs.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    # Execute each candidate and group by result
    result_groups: dict[str, list[str]] = {}
    for prog in candidates:
        tokens = parse_program_to_tokens(prog)
        invalid, result = eval_program(tokens, table)
        if invalid:
            key = "INVALID"
        else:
            key = str(result)
        result_groups.setdefault(key, []).append(prog)

    # Remove INVALID group unless it's the only one
    valid_groups = {k: v for k, v in result_groups.items() if k != "INVALID"}
    if not valid_groups:
        valid_groups = result_groups

    # Pick group with most votes
    best_key = max(valid_groups, key=lambda k: len(valid_groups[k]))
    group = valid_groups[best_key]

    # Within the group, prefer the simplest program
    group.sort(key=_count_steps)
    return group[0]


def summarizer_node(state: GraphState) -> dict:
    """Generate DSL programs via self-consistency and select the best one."""
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

    # Build messages with few-shot
    messages = [SystemMessage(content=SUMMARIZER_SYSTEM)]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append(HumanMessage(
            content=f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}\n\nWrite the DSL program:"
        ))
        from langchain_core.messages import AIMessage
        messages.append(AIMessage(content=ex["program"]))

    messages.append(HumanMessage(content=user_content))

    # Generate N candidates with higher temperature for diversity
    llm_diverse = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.candidate_temperature,
        api_key=settings.openai_api_key,
    )

    candidates: list[str] = []
    for _ in range(settings.num_candidates):
        try:
            resp = llm_diverse.invoke(messages)
            prog = resp.content.strip()
            # Basic cleanup
            prog = prog.strip("`").strip()
            if prog.startswith("Program:"):
                prog = prog[len("Program:"):].strip()
            # Only keep lines containing operations
            for line in prog.split("\n"):
                line = line.strip()
                if line and any(op + "(" in line for op in ALL_OPS):
                    candidates.append(line)
                    break
            else:
                if prog:
                    candidates.append(prog)
        except Exception:
            continue

    # Select best via simplicity-weighted majority voting
    selected = _select_program(candidates, table)

    log_entries = [LogEntry(
        agent="SummarizingAgent",
        entry_type=EntryType.SUMMARY,
        content=f"Generated {len(candidates)} candidates, selected: {selected}",
        metadata={
            "candidates": candidates,
            "selected": selected,
            "num_candidates": len(candidates),
        },
    )]

    return {
        "candidate_programs": candidates,
        "selected_program": selected,
        "log": log_entries,
    }
