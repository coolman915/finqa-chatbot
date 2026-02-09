#!/usr/bin/env python3
"""Run ONE example through the full DeALOG pipeline, printing every step."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from finqa_chatbot.config import get_settings
from finqa_chatbot.pipeline import load_dataset
from finqa_chatbot.agents.table_agent import table_agent_node
from finqa_chatbot.agents.context_agent import context_agent_node
from finqa_chatbot.agents.kg_agent import kg_agent_node
from finqa_chatbot.agents.summarizer_agent import summarizer_node
from finqa_chatbot.agents.verification_agent import verification_node
from finqa_chatbot.dsl.executor import eval_program
from finqa_chatbot.dsl.parser import parse_program_to_tokens
from finqa_chatbot.graph.scheduler import init_node, scheduler_node
from finqa_chatbot.schema import EntryType


def banner(title):
    print(f"\n{'='*70}")
    print(f"  STEP: {title}")
    print(f"{'='*70}")


settings = get_settings()
data = load_dataset("dev")
entry = data[5]  # PM/2017 — "what was the change in operating income from 2016 to 2017?"

# ── Show the input ──────────────────────────────────────────────────
banner("INPUT")
print(f"  Entry ID:  {entry['id']}")
print(f"  Question:  {entry['qa']['question']}")
print(f"  Gold prog: {entry['qa']['program']}")
print(f"  Gold ans:  {entry['qa']['exe_ans']}")
print(f"\n  Table ({len(entry['table'])} rows):")
for row in entry["table"][:6]:
    print(f"    | {'  |  '.join(str(c)[:25] for c in row)} |")
if len(entry["table"]) > 6:
    print(f"    ... ({len(entry['table'])-6} more rows)")
pre = [s for s in entry.get("pre_text", []) if s.strip() and s.strip() != "."]
print(f"\n  Text passages: {len(pre)} relevant (of {len(entry.get('pre_text',[]))} pre_text)")
for s in pre[:3]:
    print(f"    \"{s[:100]}...\"")

# ── STEP 0: Init ───────────────────────────────────────────────────
banner("0 — INIT NODE")
state = {
    "entry": entry,
    "question": "", "table": [], "pre_text": [], "post_text": [],
    "log": [], "round_number": 0, "active_agents": [], "max_rounds": settings.max_rounds,
    "candidate_programs": [], "selected_program": "", "program_tokens": [],
    "exe_result": None, "exe_invalid": False,
    "verification_status": "", "flag_targets": [], "final_answer": None,
}
init_result = init_node(state)
state.update(init_result)
print(f"  Initialized state from entry.")
print(f"  Question: {state['question']}")
print(f"  Max rounds: {state['max_rounds']}")

# ── STEP 1: Scheduler (round 1) ───────────────────────────────────
banner("1 — SCHEDULER (round 1)")
sched_result = scheduler_node(state)
state.update(sched_result)
print(f"  Round: {state['round_number']}")
print(f"  Active agents: {state['active_agents']}")
print(f"  → Will fan out to: TableAgent, ContextAgent, KGAgent in parallel")

# ── STEP 2: TableAgent ─────────────────────────────────────────────
banner("2a — TABLE AGENT (deterministic, no LLM)")
t0 = time.time()
ta_result = table_agent_node(state)
ta_time = time.time() - t0
state["log"] = state["log"] + ta_result["log"]
print(f"  Time: {ta_time:.3f}s")
print(f"  Produced {len(ta_result['log'])} LOOKUP entries")
print(f"  Sample entries:")
for e in ta_result["log"][:5]:
    print(f"    {e.to_text()[:100]}")
if len(ta_result["log"]) > 5:
    print(f"    ... and {len(ta_result['log'])-5} more")

# ── STEP 2b: ContextAgent ──────────────────────────────────────────
banner("2b — CONTEXT AGENT (TF-IDF retrieval, no LLM)")
t0 = time.time()
ca_result = context_agent_node(state)
ca_time = time.time() - t0
state["log"] = state["log"] + ca_result["log"]
print(f"  Time: {ca_time:.3f}s")
print(f"  Produced {len(ca_result['log'])} QUOTE entries")
for e in ca_result["log"][:3]:
    print(f"    {e.to_text()[:120]}")

# ── STEP 2c: KGAgent ───────────────────────────────────────────────
banner("2c — KG AGENT (LLM call to extract knowledge graph triplets)")
print(f"  Calling {settings.model_name} to extract KG triplets...")
t0 = time.time()
kg_result = kg_agent_node(state)
kg_time = time.time() - t0
state["log"] = state["log"] + kg_result["log"]
print(f"  Time: {kg_time:.1f}s")
print(f"  Produced {len(kg_result['log'])} KG_TRIPLET entries")
for e in kg_result["log"]:
    print(f"    {e.to_text()[:120]}")

# ── Show accumulated shared log ────────────────────────────────────
banner("SHARED LOG after retrieval agents")
type_counts = {}
for e in state["log"]:
    type_counts[e.entry_type.value] = type_counts.get(e.entry_type.value, 0) + 1
print(f"  Total entries: {len(state['log'])}")
for t, c in sorted(type_counts.items()):
    print(f"    {t}: {c}")

# ── STEP 3: SummarizingAgent ──────────────────────────────────────
banner("3 — SUMMARIZING AGENT (5 LLM calls for self-consistency)")
print(f"  Calling {settings.model_name} × {settings.num_candidates} candidates (temp={settings.candidate_temperature})...")
t0 = time.time()
sum_result = summarizer_node(state)
sum_time = time.time() - t0
saved_log = list(state["log"])
state.update(sum_result)
state["log"] = saved_log + sum_result["log"]
print(f"  Time: {sum_time:.1f}s")
print(f"  Candidates generated ({len(state['candidate_programs'])}):")
for i, prog in enumerate(state["candidate_programs"]):
    print(f"    [{i+1}] {prog}")
print(f"\n  Selected (majority vote + simplicity): {state['selected_program']}")

# ── STEP 4: Executor ──────────────────────────────────────────────
banner("4 — EXECUTOR (deterministic DSL execution)")
program_str = state["selected_program"]
tokens = parse_program_to_tokens(program_str)
invalid_flag, exe_result = eval_program(tokens, state["table"])
state["program_tokens"] = tokens
state["exe_result"] = exe_result
state["exe_invalid"] = bool(invalid_flag)
state["log"] = state["log"] + [
    __import__("finqa_chatbot.schema", fromlist=["LogEntry"]).LogEntry(
        agent="Executor", entry_type=EntryType.ANSWER,
        content=f"Program: {program_str} → Result: {exe_result}",
        metadata={"result": exe_result, "invalid": bool(invalid_flag)},
    )
]
print(f"  Program:  {program_str}")
print(f"  Tokens:   {tokens}")
print(f"  Result:   {exe_result}")
print(f"  Invalid:  {invalid_flag}")

# ── STEP 5: VerificationAgent ─────────────────────────────────────
banner("5 — VERIFICATION AGENT (structural checks + LLM cross-check)")
print(f"  Running 5-step verification:")
print(f"    1. Evidence grounding (every literal in log?)")
print(f"    2. Arithmetic recomputation")
print(f"    3. Temporal consistency (KG periods match question?)")
print(f"    4. Unit consistency (no mixing % and absolute?)")
print(f"    5. LLM cross-check (calling {settings.model_name})...")
t0 = time.time()
ver_result = verification_node(state)
ver_time = time.time() - t0
saved_log = list(state["log"])
state.update(ver_result)
state["log"] = saved_log + ver_result["log"]
print(f"  Time: {ver_time:.1f}s")
print(f"  Status: {state['verification_status']}")
print(f"  Flag targets: {state['flag_targets']}")
print(f"  Log: {ver_result['log'][0].to_text()[:120]}")

# If FLAG, would loop back to scheduler — show what would happen
if state["verification_status"] == "FLAG" and state["round_number"] < state["max_rounds"]:
    print(f"\n  → FLAG detected! Would re-engage: {state['flag_targets']}")
    print(f"  → But stopping here for demo (round {state['round_number']}/{state['max_rounds']})")

# ── FINAL RESULT ──────────────────────────────────────────────────
state["final_answer"] = state["exe_result"]

banner("FINAL RESULT")
gold_ans = entry["qa"]["exe_ans"]
correct = state["final_answer"] == gold_ans
print(f"  Question:     {entry['qa']['question']}")
print(f"  Predicted:    {state['selected_program']}")
print(f"  Pred answer:  {state['final_answer']}")
print(f"  Gold program: {entry['qa']['program']}")
print(f"  Gold answer:  {gold_ans}")
print(f"  CORRECT:      {'YES' if correct else 'NO'}")
print()

# Final log summary
print(f"  Shared log summary ({len(state['log'])} total entries):")
agents = {}
for e in state["log"]:
    agents[e.agent] = agents.get(e.agent, 0) + 1
for a, c in sorted(agents.items()):
    print(f"    {a:25s} → {c} entries")
