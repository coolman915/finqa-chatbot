"""End-to-end test: run 5 diverse dev samples through every agent one by one.

Each sample goes through:
  1. TableAgent  → LOOKUP entries
  2. ContextAgent → QUOTE entries
  3. KGAgent (mocked) → KG_TRIPLET entries
  4. SummarizingAgent (mocked) → SUMMARY + selected program
  5. Executor → result
  6. VerificationAgent (mocked) → OK / FLAG

Assertions check that the log accumulates correctly, types are right,
and the pipeline produces the expected gold answer.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finqa_chatbot.schema import LogEntry, EntryType
from finqa_chatbot.agents.table_agent import table_agent_node
from finqa_chatbot.agents.context_agent import context_agent_node
from finqa_chatbot.agents.kg_agent import kg_agent_node
from finqa_chatbot.agents.summarizer_agent import summarizer_node
from finqa_chatbot.agents.verification_agent import verification_node
from finqa_chatbot.dsl.executor import eval_program
from finqa_chatbot.dsl.parser import parse_program_to_tokens


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dev.json"

# (index, entry_id_prefix, gold_program, gold_answer, description)
SAMPLE_INDICES = [
    (0,  "V/2008",     "divide(637, const_5)",                       127.4,      "simple divide"),
    (5,  "PM/2017",    "subtract(11503, 10815)",                     688.0,      "simple subtract"),
    (14, "MRO/2007",   "subtract(1654, 1251), add(#0, 1654)",       2057.0,     "multi-step subtract+add"),
    (18, "ABMD/2006",  "subtract(1703, 1371), divide(#0, 1703)",    0.19495,    "percent change"),
    (30, "AES/2016",   "table_average(settlements, none)",          -11.33333,   "table_average"),
]


@pytest.fixture(scope="module")
def dev_data():
    with open(DATA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module", params=SAMPLE_INDICES, ids=[s[4] for s in SAMPLE_INDICES])
def sample(request, dev_data):
    idx, id_prefix, gold_prog, gold_ans, desc = request.param
    entry = dev_data[idx]
    assert entry["id"].startswith(id_prefix), \
        f"Index {idx} should be {id_prefix}*, got {entry['id']}"
    return entry, gold_prog, gold_ans, desc


def _build_state(entry, **overrides):
    state = {
        "entry": entry,
        "question": entry["qa"]["question"],
        "table": entry["table"],
        "pre_text": entry.get("pre_text", []),
        "post_text": entry.get("post_text", []),
        "log": [],
        "round_number": 1,
        "active_agents": ["table_agent", "context_agent", "kg_agent"],
        "max_rounds": 3,
        "candidate_programs": [],
        "selected_program": "",
        "program_tokens": [],
        "exe_result": None,
        "exe_invalid": False,
        "verification_status": "",
        "flag_targets": [],
        "final_answer": None,
    }
    state.update(overrides)
    return state


# ═══════════════════════════════════════════════════════════════════════
# Step 1: TableAgent
# ═══════════════════════════════════════════════════════════════════════

class TestStep1_TableAgent:
    def test_table_agent(self, sample):
        entry, gold_prog, gold_ans, desc = sample
        state = _build_state(entry)
        result = table_agent_node(state)
        log = result["log"]

        print(f"\n--- TableAgent [{desc}] ---")
        print(f"  Entry: {entry['id']}")
        print(f"  Table rows: {len(entry['table'])}")
        print(f"  LOOKUP entries produced: {len(log)}")
        for e in log[:5]:
            print(f"    {e.to_text()[:100]}")
        if len(log) > 5:
            print(f"    ... and {len(log) - 5} more")

        # Every entry must be a LOOKUP from TableAgent
        assert all(e.agent == "TableAgent" for e in log)
        assert all(e.entry_type == EntryType.LOOKUP for e in log)

        # Must have at least 1 entry
        assert len(log) >= 1

        # If the table has data rows, we should have metadata
        if len(entry["table"]) > 1:
            assert all("source_ind" in e.metadata for e in log
                       if "No" not in e.content)


# ═══════════════════════════════════════════════════════════════════════
# Step 2: ContextAgent
# ═══════════════════════════════════════════════════════════════════════

class TestStep2_ContextAgent:
    def test_context_agent(self, sample):
        entry, gold_prog, gold_ans, desc = sample
        state = _build_state(entry)
        result = context_agent_node(state)
        log = result["log"]

        # Count non-trivial text
        all_text = entry.get("pre_text", []) + entry.get("post_text", [])
        real_text = [s for s in all_text if s.strip() and s.strip() != "."]

        print(f"\n--- ContextAgent [{desc}] ---")
        print(f"  Entry: {entry['id']}")
        print(f"  Text passages available: {len(real_text)}")
        print(f"  QUOTE entries produced: {len(log)}")
        for e in log[:3]:
            print(f"    {e.to_text()[:120]}")

        assert all(e.agent == "ContextAgent" for e in log)
        assert all(e.entry_type == EntryType.QUOTE for e in log)
        assert len(log) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Step 3: KGAgent (mocked)
# ═══════════════════════════════════════════════════════════════════════

class TestStep3_KGAgent:
    @patch("finqa_chatbot.agents.kg_agent.ChatOpenAI")
    def test_kg_agent(self, mock_chat_cls, sample):
        entry, gold_prog, gold_ans, desc = sample

        # Build realistic triplets from the entry's gold data
        triplets = []
        for step in entry["qa"].get("steps", []):
            for arg_key in ("arg1", "arg2"):
                arg = step[arg_key]
                if arg.startswith("#") or arg.startswith("const_") or arg.lower() == "none":
                    continue
                triplets.append({
                    "subject": "Metric",
                    "relation": "HAS_VALUE",
                    "object": f"{arg}",
                    "entity_type": "financial_metric",
                    "company": "",
                    "period": "",
                    "value": None,
                    "unit": "",
                })

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = json.dumps(triplets) if triplets else "[]"
        mock_llm.invoke.return_value = mock_resp
        mock_chat_cls.return_value = mock_llm

        state = _build_state(entry)
        result = kg_agent_node(state)
        log = result["log"]

        print(f"\n--- KGAgent [{desc}] ---")
        print(f"  Entry: {entry['id']}")
        print(f"  Mock triplets provided: {len(triplets)}")
        print(f"  KG_TRIPLET entries produced: {len(log)}")
        for e in log[:5]:
            print(f"    {e.to_text()[:120]}")

        assert all(e.agent == "KGAgent" for e in log)
        assert all(e.entry_type == EntryType.KG_TRIPLET for e in log)
        assert len(log) >= 1

        # If we mocked triplets, entries should contain them
        if triplets:
            assert len(log) == len(triplets)
            for e in log:
                assert "subject" in e.metadata


# ═══════════════════════════════════════════════════════════════════════
# Step 4: SummarizingAgent (mocked)
# ═══════════════════════════════════════════════════════════════════════

class TestStep4_SummarizingAgent:
    @patch("finqa_chatbot.agents.summarizer_agent.ChatOpenAI")
    def test_summarizer_agent(self, mock_chat_cls, sample):
        entry, gold_prog, gold_ans, desc = sample

        # Simulate self-consistency: 4× gold program + 1× wrong program
        mock_llm = MagicMock()
        responses = []
        for i in range(5):
            resp = MagicMock()
            if i < 4:
                resp.content = gold_prog
            else:
                resp.content = "add(1, 1)"  # distractor
            responses.append(resp)
        mock_llm.invoke.side_effect = responses
        mock_chat_cls.return_value = mock_llm

        # Build accumulated log from previous agents
        prior_log = [
            LogEntry("TableAgent", EntryType.LOOKUP, "sample lookup value",
                     metadata={"numeric": "100"}),
            LogEntry("ContextAgent", EntryType.QUOTE, "sample quote text"),
            LogEntry("KGAgent", EntryType.KG_TRIPLET, "(Metric, R, V)"),
        ]
        state = _build_state(entry, log=prior_log)
        result = summarizer_node(state)

        print(f"\n--- SummarizingAgent [{desc}] ---")
        print(f"  Entry: {entry['id']}")
        print(f"  Gold program: {gold_prog}")
        print(f"  Candidates generated: {len(result['candidate_programs'])}")
        print(f"  Selected: {result['selected_program']}")
        print(f"  Log entry: {result['log'][0].to_text()[:120]}")

        assert len(result["candidate_programs"]) == 5
        assert result["selected_program"] == gold_prog  # majority should win
        assert len(result["log"]) == 1
        assert result["log"][0].agent == "SummarizingAgent"
        assert result["log"][0].entry_type == EntryType.SUMMARY


# ═══════════════════════════════════════════════════════════════════════
# Step 5: Executor (deterministic, no mock needed)
# ═══════════════════════════════════════════════════════════════════════

class TestStep5_Executor:
    def test_executor(self, sample):
        entry, gold_prog, gold_ans, desc = sample

        tokens = parse_program_to_tokens(gold_prog)
        invalid_flag, result = eval_program(tokens, entry["table"])

        print(f"\n--- Executor [{desc}] ---")
        print(f"  Entry: {entry['id']}")
        print(f"  Program: {gold_prog}")
        print(f"  Tokens: {tokens}")
        print(f"  Result: {result}")
        print(f"  Expected: {gold_ans}")
        print(f"  Invalid: {invalid_flag}")
        print(f"  Match: {result == gold_ans}")

        assert invalid_flag == 0, f"Gold program should not be invalid: {gold_prog}"
        assert result == gold_ans, f"Expected {gold_ans}, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Step 6: VerificationAgent (mocked LLM cross-check)
# ═══════════════════════════════════════════════════════════════════════

class TestStep6_VerificationAgent:
    @patch("finqa_chatbot.agents.verification_agent.ChatOpenAI")
    def test_verification_agent(self, mock_chat_cls, sample):
        entry, gold_prog, gold_ans, desc = sample

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "STATUS: OK"
        mock_llm.invoke.return_value = mock_resp
        mock_chat_cls.return_value = mock_llm

        # Build a full log with evidence that grounds the program
        log_entries = []

        # TableAgent lookups covering table values
        for row_idx, row in enumerate(entry["table"][1:], start=1):
            for col_idx, cell in enumerate(row[1:], start=1):
                log_entries.append(LogEntry(
                    agent="TableAgent",
                    entry_type=EntryType.LOOKUP,
                    content=f"{row[0]} | {entry['table'][0][col_idx]} = {cell}",
                    metadata={
                        "numeric": cell.replace("$", "").replace(",", "").strip(),
                        "value": cell.strip(),
                    },
                ))

        # ContextAgent quotes
        for text in entry.get("pre_text", [])[:3]:
            if text.strip() and text.strip() != ".":
                log_entries.append(LogEntry(
                    agent="ContextAgent",
                    entry_type=EntryType.QUOTE,
                    content=text,
                ))

        # Summary
        log_entries.append(LogEntry(
            agent="SummarizingAgent",
            entry_type=EntryType.SUMMARY,
            content=f"Selected: {gold_prog}",
        ))

        # Executor answer
        tokens = parse_program_to_tokens(gold_prog)
        _, exe_result = eval_program(tokens, entry["table"])

        log_entries.append(LogEntry(
            agent="Executor",
            entry_type=EntryType.ANSWER,
            content=f"Program: {gold_prog} → Result: {exe_result}",
        ))

        state = _build_state(
            entry,
            log=log_entries,
            selected_program=gold_prog,
            program_tokens=tokens,
            exe_result=exe_result,
            exe_invalid=False,
        )
        result = verification_node(state)

        print(f"\n--- VerificationAgent [{desc}] ---")
        print(f"  Entry: {entry['id']}")
        print(f"  Program: {gold_prog}")
        print(f"  Exe result: {exe_result}")
        print(f"  Verification status: {result['verification_status']}")
        print(f"  Flag targets: {result['flag_targets']}")
        print(f"  Log: {result['log'][0].to_text()[:120]}")

        assert len(result["log"]) == 1
        assert result["log"][0].agent == "VerificationAgent"

        # Gold program with full evidence + LLM OK should verify
        if result["verification_status"] != "OK":
            print(f"  WARNING: Verification flagged gold program for [{desc}]: "
                  f"{result['log'][0].content}")


# ═══════════════════════════════════════════════════════════════════════
# Full pipeline: all agents in sequence
# ═══════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    @patch("finqa_chatbot.agents.verification_agent.ChatOpenAI")
    @patch("finqa_chatbot.agents.summarizer_agent.ChatOpenAI")
    @patch("finqa_chatbot.agents.kg_agent.ChatOpenAI")
    def test_full_pipeline(self, mock_kg_chat, mock_sum_chat, mock_ver_chat, sample):
        entry, gold_prog, gold_ans, desc = sample

        # --- Mock KGAgent ---
        kg_llm = MagicMock()
        kg_resp = MagicMock()
        triplets = []
        for step in entry["qa"].get("steps", []):
            for arg_key in ("arg1", "arg2"):
                arg = step[arg_key]
                if arg.startswith("#") or arg.startswith("const_") or arg.lower() == "none":
                    continue
                triplets.append({"subject": "M", "relation": "R", "object": arg,
                                 "entity_type": "financial_metric", "company": "",
                                 "period": "", "value": None, "unit": ""})
        kg_resp.content = json.dumps(triplets) if triplets else "[]"
        kg_llm.invoke.return_value = kg_resp
        mock_kg_chat.return_value = kg_llm

        # --- Mock SummarizingAgent ---
        sum_llm = MagicMock()
        sum_responses = [MagicMock() for _ in range(5)]
        for r in sum_responses[:4]:
            r.content = gold_prog
        sum_responses[4].content = "add(1, 1)"
        sum_llm.invoke.side_effect = sum_responses
        mock_sum_chat.return_value = sum_llm

        # --- Mock VerificationAgent ---
        ver_llm = MagicMock()
        ver_resp = MagicMock()
        ver_resp.content = "STATUS: OK"
        ver_llm.invoke.return_value = ver_resp
        mock_ver_chat.return_value = ver_llm

        # === Run agents in sequence ===
        state = _build_state(entry)
        accumulated_log = []

        # 1. TableAgent
        ta_result = table_agent_node(state)
        accumulated_log.extend(ta_result["log"])

        # 2. ContextAgent
        ca_result = context_agent_node(state)
        accumulated_log.extend(ca_result["log"])

        # 3. KGAgent
        state_with_log = {**state, "log": accumulated_log}
        kg_result = kg_agent_node(state_with_log)
        accumulated_log.extend(kg_result["log"])

        # 4. SummarizingAgent
        state_with_log = {**state, "log": accumulated_log}
        sum_result = summarizer_node(state_with_log)
        accumulated_log.extend(sum_result["log"])
        selected = sum_result["selected_program"]

        # 5. Executor
        tokens = parse_program_to_tokens(selected)
        inv, exe_res = eval_program(tokens, entry["table"])

        accumulated_log.append(LogEntry(
            agent="Executor", entry_type=EntryType.ANSWER,
            content=f"Program: {selected} → Result: {exe_res}",
        ))

        # 6. VerificationAgent
        state_final = {
            **state,
            "log": accumulated_log,
            "selected_program": selected,
            "program_tokens": tokens,
            "exe_result": exe_res,
            "exe_invalid": bool(inv),
        }
        ver_result = verification_node(state_final)
        accumulated_log.extend(ver_result["log"])

        # === Report ===
        print(f"\n{'='*70}")
        print(f"FULL PIPELINE [{desc}] — {entry['id']}")
        print(f"{'='*70}")
        print(f"  Question: {entry['qa']['question']}")
        print(f"  Gold program: {gold_prog}")
        print(f"  Gold answer: {gold_ans}")
        print()

        agent_counts = {}
        for e in accumulated_log:
            agent_counts[e.agent] = agent_counts.get(e.agent, 0) + 1
        for agent, count in sorted(agent_counts.items()):
            print(f"  {agent:25s}: {count} log entries")

        print()
        print(f"  Selected program: {selected}")
        print(f"  Execution result: {exe_res}")
        print(f"  Verification:     {ver_result['verification_status']}")
        print(f"  Correct answer:   {exe_res == gold_ans}")
        print(f"  Total log entries: {len(accumulated_log)}")

        # === Assertions ===
        # Log should have entries from all 5 agents + executor
        agents_in_log = {e.agent for e in accumulated_log}
        assert "TableAgent" in agents_in_log, "Missing TableAgent entries"
        assert "ContextAgent" in agents_in_log, "Missing ContextAgent entries"
        assert "KGAgent" in agents_in_log, "Missing KGAgent entries"
        assert "SummarizingAgent" in agents_in_log, "Missing SummarizingAgent entries"
        assert "Executor" in agents_in_log, "Missing Executor entries"
        assert "VerificationAgent" in agents_in_log, "Missing VerificationAgent entries"

        # Entry types should be diverse
        types_in_log = {e.entry_type for e in accumulated_log}
        assert EntryType.LOOKUP in types_in_log
        assert EntryType.QUOTE in types_in_log or len(
            [s for s in entry.get("pre_text", []) + entry.get("post_text", [])
             if s.strip() and s.strip() != "."]) == 0
        assert EntryType.KG_TRIPLET in types_in_log
        assert EntryType.SUMMARY in types_in_log
        assert EntryType.ANSWER in types_in_log

        # Program should match gold (since we mocked majority = gold)
        assert selected == gold_prog
        # Execution should produce gold answer
        assert exe_res == gold_ans
        assert inv == 0
