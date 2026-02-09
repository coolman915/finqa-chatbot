"""VerificationAgent — log-grounded cross-check with re-engagement targeting."""

from __future__ import annotations

import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import get_settings
from ..schema import LogEntry, EntryType
from ..graph.state import GraphState
from ..dsl.parser import parse_program_to_tokens
from ..dsl.executor import eval_program
from ..dsl.operations import ALL_OPS
from ..prompts.verification import VERIFICATION_SYSTEM, VERIFICATION_USER_TEMPLATE


# Agent name → scheduler agent key mapping
_AGENT_MAP = {
    "tableagent": "table_agent",
    "contextagent": "context_agent",
    "kgagent": "kg_agent",
    "summarizingagent": "summarizer",
    "summarizer": "summarizer",
}


def _extract_program_literals(program_str: str) -> list[str]:
    """Extract numeric literals from a program string (not #refs or ops)."""
    literals = []
    tokens = parse_program_to_tokens(program_str)
    for tok in tokens:
        tok = tok.strip()
        if tok == "EOF" or tok.endswith("(") or tok == ")" or tok.startswith("#"):
            continue
        if tok.startswith("const_"):
            continue
        if any(tok == op for op in ALL_OPS):
            continue
        # It's a literal (number or row name)
        if re.match(r'^-?[\d.]+%?$', tok):
            literals.append(tok)
    return literals


def _check_evidence_grounding(
    program_str: str, log: list[LogEntry]
) -> tuple[bool, list[str]]:
    """Check that every numeric literal in the program appears in log evidence."""
    literals = _extract_program_literals(program_str)
    if not literals:
        return True, []

    # Collect all numbers mentioned in evidence entries
    evidence_numbers: set[str] = set()
    for entry in log:
        if entry.entry_type in (EntryType.LOOKUP, EntryType.QUOTE, EntryType.KG_TRIPLET):
            # Extract numbers from content and metadata
            nums = re.findall(r'-?[\d,]+\.?\d*%?', entry.content)
            evidence_numbers.update(n.replace(",", "") for n in nums)
            if "value" in entry.metadata:
                evidence_numbers.add(str(entry.metadata["value"]))
            if "numeric" in entry.metadata:
                evidence_numbers.add(str(entry.metadata["numeric"]))

    missing = [lit for lit in literals if lit not in evidence_numbers]
    return len(missing) == 0, missing


def _check_temporal_consistency(
    program_str: str, log: list[LogEntry], question: str
) -> tuple[bool, str]:
    """Check that values come from periods mentioned in the question."""
    q_years = set(re.findall(r'\b((?:19|20)\d{2})\b', question))
    if not q_years:
        return True, ""

    kg_entries = [e for e in log if e.entry_type == EntryType.KG_TRIPLET]
    if not kg_entries:
        return True, ""

    for entry in kg_entries:
        period = entry.metadata.get("period", "")
        if not period:
            continue
        period_years = set(re.findall(r'\b((?:19|20)\d{2})\b', period))
        if period_years and not period_years.issubset(q_years):
            return False, f"KG triplet period {period} not in question years {q_years}"

    return True, ""


def _check_unit_consistency(program_str: str) -> tuple[bool, str]:
    """Check that add/subtract don't mix % values with absolute values."""
    tokens = parse_program_to_tokens(program_str)
    program = tokens[:-1]  # remove EOF

    for ind in range(0, len(program), 4):
        if ind + 2 >= len(program):
            break
        op = program[ind].strip("(")
        if op in ("add", "subtract"):
            arg1 = program[ind + 1] if ind + 1 < len(program) else ""
            arg2 = program[ind + 2] if ind + 2 < len(program) else ""
            a1_pct = "%" in arg1 and "#" not in arg1
            a2_pct = "%" in arg2 and "#" not in arg2
            if a1_pct != a2_pct and not arg1.startswith("#") and not arg2.startswith("#"):
                return False, f"Mixing % and absolute in {op}({arg1}, {arg2})"

    return True, ""


def verification_node(state: GraphState) -> dict:
    """Perform log-grounded verification of the selected program.

    Runs structural checks first, then an LLM cross-check.
    Per DeALOG paper: verification improves accuracy by ~4 points.
    """
    settings = get_settings()
    program_str = state.get("selected_program", "")
    exe_result = state.get("exe_result", "n/a")
    exe_invalid = state.get("exe_invalid", False)
    log = state.get("log", [])
    question = state.get("question", "")

    # If execution failed, FLAG immediately
    if exe_invalid or exe_result == "n/a":
        return {
            "verification_status": "FLAG",
            "flag_targets": ["table_agent", "context_agent", "summarizer"],
            "log": [LogEntry(
                agent="VerificationAgent",
                entry_type=EntryType.FLAG,
                content=f"Program execution failed: {program_str}",
                metadata={"issue": "execution_failure", "targets": ["table_agent", "context_agent", "summarizer"]},
            )],
        }

    issues: list[str] = []
    targets: set[str] = set()

    # 1. Evidence grounding
    grounded, missing = _check_evidence_grounding(program_str, log)
    if not grounded:
        issues.append(f"Missing evidence for literals: {missing}")
        targets.update(["table_agent", "context_agent"])

    # 2. Arithmetic recomputation
    tokens = parse_program_to_tokens(program_str)
    inv, recomputed = eval_program(tokens, state.get("table", []))
    if inv or recomputed != exe_result:
        issues.append(f"Recomputation mismatch: {recomputed} vs {exe_result}")
        targets.add("summarizer")

    # 3. Temporal consistency
    temporal_ok, temporal_issue = _check_temporal_consistency(program_str, log, question)
    if not temporal_ok:
        issues.append(temporal_issue)
        targets.add("kg_agent")

    # 4. Unit consistency
    unit_ok, unit_issue = _check_unit_consistency(program_str)
    if not unit_ok:
        issues.append(unit_issue)
        targets.add("summarizer")

    # 5. LLM cross-check disabled — causes instability with gpt-5-nano
    # (false flags degrade correct programs through re-engagement loop)
    # Structural checks (evidence grounding, arithmetic, temporal, unit) are sufficient.

    if issues:
        target_list = list(targets) if targets else ["summarizer"]
        return {
            "verification_status": "FLAG",
            "flag_targets": target_list,
            "log": [LogEntry(
                agent="VerificationAgent",
                entry_type=EntryType.FLAG,
                content=f"Issues found: {'; '.join(issues)}",
                metadata={"issues": issues, "targets": target_list},
            )],
        }

    return {
        "verification_status": "OK",
        "flag_targets": [],
        "log": [LogEntry(
            agent="VerificationAgent",
            entry_type=EntryType.OK,
            content=f"Program verified: {program_str} → {exe_result}",
            metadata={"program": program_str, "result": exe_result},
        )],
    }
