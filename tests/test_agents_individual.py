"""Individual agent tests — verify each agent produces correct log entries."""

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from finqa_chatbot.schema import LogEntry, EntryType, KGTriplet
from finqa_chatbot.agents.table_agent import table_agent_node, _extract_numeric
from finqa_chatbot.agents.context_agent import context_agent_node
from finqa_chatbot.agents.kg_agent import kg_agent_node, _parse_triplets_response, _dict_to_triplet
from finqa_chatbot.agents.summarizer_agent import (
    summarizer_node, _select_program, _format_table, _format_log, _count_steps,
)
from finqa_chatbot.agents.verification_agent import (
    verification_node,
    _extract_program_literals,
    _check_evidence_grounding,
    _check_unit_consistency,
    _check_temporal_consistency,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def visa_entry():
    """First dev entry — Visa payments volume."""
    return {
        "id": "V/2008/page_17.pdf-1",
        "pre_text": [
            "largest operators of open-loop and closed-loop retail electronic payments networks "
            "the largest operators are visa , mastercard , american express , discover , jcb and diners club .",
            "based on payments volume , total volume , number of transactions and number of cards "
            "in circulation , visa is the largest retail electronic payments network in the world .",
        ],
        "post_text": [
            "( 1 ) visa inc . figures as reported previously in our filings .",
            "source : the nilson report , issue 902 ( may 2008 ) and issue 903 ( may 2008 ) .",
        ],
        "table": [
            ["company", "payments volume ( billions )", "total volume ( billions )",
             "total transactions ( billions )", "cards ( millions )"],
            ["visa inc. ( 1 )", "$ 2457", "$ 3822", "50.3", "1592"],
            ["mastercard", "1697", "2276", "27.0", "916"],
            ["american express", "637", "846", "5.0", "86"],
            ["discover", "112", "120", "2.0", "54"],
            ["jcb", "61", "72", "0.6", "59"],
        ],
        "qa": {
            "question": "what is the average payment volume per transaction for american express?",
            "answer": "127.40",
            "program": "divide(637, const_5)",
            "exe_ans": 127.4,
            "steps": [{"op": "divide1-1", "arg1": "637", "arg2": "const_5", "res": "127.40"}],
        },
    }


@pytest.fixture
def revenue_entry():
    """Entry with text + table about revenue changes."""
    return {
        "id": "test-revenue-entry",
        "pre_text": [
            "total net revenue increased $ 693 million , or 11% ( 11 % ) , to $ 7.0 billion .",
            "net income was $ 1.2 billion in 2006 compared to $ 1.0 billion in 2005 .",
        ],
        "post_text": [
            "the increase was primarily driven by organic growth in all business segments .",
            "operating expenses increased 8% to $ 5.5 billion .",
        ],
        "table": [
            ["year", "net revenue ( billions )", "operating income ( billions )"],
            ["2006", "7.0", "1.5"],
            ["2005", "6.3", "1.2"],
            ["2004", "5.8", "1.0"],
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


@pytest.fixture
def operating_income_entry():
    """Simple table-only entry about operating income change."""
    return {
        "id": "test-opinc-entry",
        "pre_text": [],
        "post_text": [],
        "table": [
            ["( in millions )", "2017", "2016"],
            ["operating income", "11503", "10815"],
            ["net income", "8500", "7200"],
        ],
        "qa": {
            "question": "what was the change in millions of operating income from 2016 to 2017?",
            "answer": "688",
            "program": "subtract(11503, 10815)",
            "exe_ans": 688.0,
            "steps": [{"op": "subtract1-1", "arg1": "11503", "arg2": "10815", "res": "688"}],
        },
    }


def _build_state(entry, **overrides):
    """Helper to build a minimal GraphState dict from an entry."""
    state = {
        "entry": entry,
        "question": entry["qa"]["question"],
        "table": entry["table"],
        "pre_text": entry.get("pre_text", []),
        "post_text": entry.get("post_text", []),
        "log": [],
        "round_number": 1,
        "active_agents": [],
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
# 1. TableAgent
# ═══════════════════════════════════════════════════════════════════════

class TestTableAgent:
    """Test the TableAgent deterministically (no LLM needed)."""

    def test_extract_numeric_dollar(self):
        assert _extract_numeric("$ 2457") == "2457"

    def test_extract_numeric_plain(self):
        assert _extract_numeric("1697") == "1697"

    def test_extract_numeric_decimal(self):
        assert _extract_numeric("50.3") == "50.3"

    def test_extract_numeric_percent(self):
        assert _extract_numeric("27.3%") == "27.3%"

    def test_extract_numeric_text(self):
        assert _extract_numeric("company name") is None

    def test_visa_entry_lookups(self, visa_entry):
        """TableAgent should extract all numeric cells from the Visa table."""
        state = _build_state(visa_entry)
        result = table_agent_node(state)
        log = result["log"]

        # 5 data rows × 4 numeric columns = 20 LOOKUP entries
        assert len(log) >= 15  # at least most cells
        assert all(e.agent == "TableAgent" for e in log)
        assert all(e.entry_type == EntryType.LOOKUP for e in log)

        # Check metadata structure
        for entry in log:
            assert "row_name" in entry.metadata
            assert "column" in entry.metadata
            assert "value" in entry.metadata
            assert "numeric" in entry.metadata
            assert "source_ind" in entry.metadata

        # Check specific values exist
        contents = [e.content for e in log]
        assert any("637" in c and "american express" in c for c in contents), \
            "Should find american express payment volume 637"
        assert any("2457" in c and "visa" in c.lower() for c in contents), \
            "Should find visa payment volume 2457"

    def test_revenue_entry_lookups(self, revenue_entry):
        """TableAgent should extract all numeric cells from revenue table."""
        state = _build_state(revenue_entry)
        result = table_agent_node(state)
        log = result["log"]

        # 3 data rows × 2 numeric columns = 6 LOOKUP entries
        assert len(log) >= 6
        contents = [e.content for e in log]
        assert any("7.0" in c for c in contents), "Should find 7.0 revenue"
        assert any("6.3" in c for c in contents), "Should find 6.3 revenue"

    def test_empty_table(self):
        """Empty table should produce a single informational entry."""
        state = _build_state({"table": [], "pre_text": [], "post_text": [],
                              "qa": {"question": "test"}})
        result = table_agent_node(state)
        assert len(result["log"]) == 1
        assert "No table" in result["log"][0].content or "no" in result["log"][0].content.lower()

    def test_single_row_table(self):
        """Header-only table should produce informational entry."""
        state = _build_state({"table": [["col1", "col2"]], "pre_text": [], "post_text": [],
                              "qa": {"question": "test"}})
        result = table_agent_node(state)
        assert len(result["log"]) == 1

    def test_log_entries_are_serializable(self, visa_entry):
        """LogEntry.to_text() should work on all entries."""
        state = _build_state(visa_entry)
        result = table_agent_node(state)
        for entry in result["log"]:
            text = entry.to_text()
            assert isinstance(text, str)
            assert "TableAgent" in text
            assert "LOOKUP" in text


# ═══════════════════════════════════════════════════════════════════════
# 2. ContextAgent
# ═══════════════════════════════════════════════════════════════════════

class TestContextAgent:
    """Test the ContextAgent deterministically (TF-IDF retrieval, no LLM)."""

    def test_revenue_entry_quotes(self, revenue_entry):
        """ContextAgent should retrieve relevant text passages."""
        state = _build_state(revenue_entry)
        result = context_agent_node(state)
        log = result["log"]

        assert len(log) > 0
        assert all(e.agent == "ContextAgent" for e in log)
        assert all(e.entry_type == EntryType.QUOTE for e in log)

        # The most relevant quote should mention "net revenue"
        contents = [e.content for e in log]
        assert any("net revenue" in c.lower() for c in contents), \
            f"Should retrieve passage mentioning net revenue. Got: {contents}"

    def test_revenue_entry_extracts_numbers(self, revenue_entry):
        """ContextAgent should extract numbers from text passages."""
        state = _build_state(revenue_entry)
        result = context_agent_node(state)
        log = result["log"]

        # Check at least one entry has numbers in metadata
        entries_with_numbers = [e for e in log if e.metadata.get("numbers")]
        assert len(entries_with_numbers) > 0, "Should extract numbers from text passages"

        # The revenue passage should contain $693 million or $7.0 billion
        all_numbers = []
        for e in log:
            all_numbers.extend(e.metadata.get("numbers", []))
        number_text = " ".join(all_numbers)
        assert "693" in number_text or "7.0" in number_text, \
            f"Should extract financial numbers. Got: {all_numbers}"

    def test_visa_entry_quotes(self, visa_entry):
        """ContextAgent should retrieve relevant passages from Visa entry."""
        state = _build_state(visa_entry)
        result = context_agent_node(state)
        log = result["log"]

        assert len(log) > 0
        # Should retrieve text about payments or american express
        contents = " ".join(e.content.lower() for e in log)
        assert "payment" in contents or "american express" in contents or "visa" in contents

    def test_metadata_structure(self, revenue_entry):
        """Each QUOTE entry should have proper metadata."""
        state = _build_state(revenue_entry)
        result = context_agent_node(state)
        for entry in result["log"]:
            assert "source_ind" in entry.metadata
            assert "span" in entry.metadata
            assert entry.metadata["source_ind"].startswith("text_")

    def test_no_text_available(self):
        """ContextAgent should handle entries with no text gracefully."""
        state = _build_state({"pre_text": [], "post_text": [], "table": [],
                              "qa": {"question": "test question"}})
        result = context_agent_node(state)
        assert len(result["log"]) >= 1
        assert "No text" in result["log"][0].content or "no" in result["log"][0].content.lower()

    def test_dot_only_text_filtered(self):
        """Entries with only '.' text should be filtered out."""
        state = _build_state({
            "pre_text": [".", ".", "."],
            "post_text": [".", "."],
            "table": [],
            "qa": {"question": "test"},
        })
        result = context_agent_node(state)
        # Should get the "no text" fallback since dots are filtered
        assert len(result["log"]) >= 1


# ═══════════════════════════════════════════════════════════════════════
# 3. KGAgent
# ═══════════════════════════════════════════════════════════════════════

class TestKGAgent:
    """Test KGAgent with mocked LLM responses."""

    def test_parse_triplets_valid_json(self):
        """Should parse valid JSON triplet array."""
        text = '''[
            {"subject": "Revenue", "relation": "HAS_VALUE_IN_2017", "object": "100 million",
             "entity_type": "financial_metric", "company": "", "period": "2017", "value": 100.0, "unit": "million"}
        ]'''
        result = _parse_triplets_response(text)
        assert len(result) == 1
        assert result[0]["subject"] == "Revenue"

    def test_parse_triplets_with_surrounding_text(self):
        """Should extract JSON array even with surrounding text."""
        text = '''Here are the triplets:
        [{"subject": "A", "relation": "R", "object": "O"}]
        Done.'''
        result = _parse_triplets_response(text)
        assert len(result) == 1

    def test_parse_triplets_invalid(self):
        """Should return empty list for unparseable text."""
        assert _parse_triplets_response("not json at all") == []
        assert _parse_triplets_response("") == []

    def test_dict_to_triplet(self):
        """Should convert dict to KGTriplet dataclass."""
        d = {
            "subject": "Revenue",
            "relation": "HAS_VALUE_IN_2017",
            "object": "100 million",
            "entity_type": "financial_metric",
            "company": "Acme",
            "period": "2017",
            "value": 100.0,
            "unit": "million",
        }
        t = _dict_to_triplet(d)
        assert t.subject == "Revenue"
        assert t.period == "2017"
        assert t.value == 100.0
        assert t.to_text() == "(Revenue, HAS_VALUE_IN_2017, 100 million)"

    def test_dict_to_triplet_missing_fields(self):
        """Should handle missing fields gracefully."""
        t = _dict_to_triplet({"subject": "X", "relation": "Y"})
        assert t.subject == "X"
        assert t.object == ""
        assert t.value is None

    @patch("finqa_chatbot.agents.kg_agent.ChatOpenAI")
    def test_kg_agent_node_success(self, mock_chat_cls, operating_income_entry):
        """KGAgent should produce KG_TRIPLET entries from mocked LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {
                "subject": "OperatingIncome",
                "relation": "HAS_VALUE_IN_2017",
                "object": "11503 million",
                "entity_type": "financial_metric",
                "company": "",
                "period": "2017",
                "value": 11503.0,
                "unit": "million",
            },
            {
                "subject": "OperatingIncome",
                "relation": "HAS_VALUE_IN_2016",
                "object": "10815 million",
                "entity_type": "financial_metric",
                "company": "",
                "period": "2016",
                "value": 10815.0,
                "unit": "million",
            },
        ])
        mock_llm.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_llm

        state = _build_state(operating_income_entry)
        result = kg_agent_node(state)
        log = result["log"]

        assert len(log) == 2
        assert all(e.agent == "KGAgent" for e in log)
        assert all(e.entry_type == EntryType.KG_TRIPLET for e in log)

        # Check metadata
        assert log[0].metadata["subject"] == "OperatingIncome"
        assert log[0].metadata["period"] == "2017"
        assert log[0].metadata["value"] == 11503.0
        assert log[1].metadata["period"] == "2016"

        # Check content contains triplet text
        assert "OperatingIncome" in log[0].content
        assert "HAS_VALUE_IN_2017" in log[0].content

    @patch("finqa_chatbot.agents.kg_agent.ChatOpenAI")
    def test_kg_agent_node_empty_response(self, mock_chat_cls, operating_income_entry):
        """KGAgent should handle empty LLM response gracefully."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "[]"
        mock_llm.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_llm

        state = _build_state(operating_income_entry)
        result = kg_agent_node(state)
        log = result["log"]

        assert len(log) == 1
        assert "No triplets" in log[0].content

    @patch("finqa_chatbot.agents.kg_agent.ChatOpenAI")
    def test_kg_agent_node_llm_failure(self, mock_chat_cls, operating_income_entry):
        """KGAgent should handle LLM exceptions gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API rate limit")
        mock_chat_cls.return_value = mock_llm

        state = _build_state(operating_income_entry)
        result = kg_agent_node(state)
        log = result["log"]

        assert len(log) == 1
        assert "failed" in log[0].content.lower()
        assert log[0].entry_type == EntryType.KG_TRIPLET


# ═══════════════════════════════════════════════════════════════════════
# 4. SummarizingAgent
# ═══════════════════════════════════════════════════════════════════════

class TestSummarizingAgent:
    """Test the SummarizingAgent with mocked LLM."""

    def test_format_table(self):
        table = [["year", "revenue"], ["2006", "7.0"], ["2005", "6.3"]]
        result = _format_table(table)
        assert "year" in result
        assert "7.0" in result
        assert "|" in result

    def test_format_table_empty(self):
        assert _format_table([]) == "(no table)"

    def test_format_log(self):
        log = [
            LogEntry("TableAgent", EntryType.LOOKUP, "income | 2017 = 11503"),
            LogEntry("ContextAgent", EntryType.QUOTE, "revenue grew 11%"),
        ]
        result = _format_log(log)
        assert "TableAgent" in result
        assert "ContextAgent" in result
        assert "11503" in result

    def test_format_log_empty(self):
        assert _format_log([]) == "(no evidence yet)"

    def test_count_steps(self):
        assert _count_steps("divide(637, const_5)") == 1
        assert _count_steps("subtract(7.0, 6.3), divide(#0, 6.3)") == 2
        assert _count_steps("subtract(a, b), multiply(#0, c), divide(#1, d)") == 3

    def test_select_program_single(self):
        """Single candidate should be selected."""
        result = _select_program(["divide(637, const_5)"], [])
        assert result == "divide(637, const_5)"

    def test_select_program_majority(self):
        """Majority result should win."""
        table = [
            ["( in millions )", "2017", "2016"],
            ["operating income", "11503", "10815"],
        ]
        candidates = [
            "subtract(11503, 10815)",  # → 688
            "subtract(11503, 10815)",  # → 688
            "subtract(11503, 10815)",  # → 688
            "add(11503, 10815)",       # → 22318
            "divide(11503, 10815)",    # → different
        ]
        result = _select_program(candidates, table)
        assert result == "subtract(11503, 10815)"

    def test_select_program_prefers_simpler(self):
        """Among tied results, prefer simpler program."""
        table = []
        # Both produce the same result but first is simpler
        candidates = [
            "subtract(100, 50)",                             # 1 step → 50
            "subtract(200, 100), subtract(#0, 50)",          # 2 steps → 50
            "subtract(100, 50)",                             # 1 step → 50
        ]
        result = _select_program(candidates, table)
        assert result == "subtract(100, 50)"

    def test_select_program_empty(self):
        assert _select_program([], []) == ""

    @patch("finqa_chatbot.agents.summarizer_agent.ChatOpenAI")
    def test_summarizer_node(self, mock_chat_cls, operating_income_entry):
        """SummarizingAgent should generate candidates and select one."""
        mock_llm = MagicMock()

        # Simulate 5 LLM calls returning programs
        responses = []
        for prog in [
            "subtract(11503, 10815)",
            "subtract(11503, 10815)",
            "subtract(11503, 10815)",
            "add(11503, 10815)",
            "subtract(11503, 10815)",
        ]:
            resp = MagicMock()
            resp.content = prog
            responses.append(resp)

        mock_llm.invoke.side_effect = responses
        mock_chat_cls.return_value = mock_llm

        log = [
            LogEntry("TableAgent", EntryType.LOOKUP, "operating income | 2017 = 11503",
                     metadata={"numeric": "11503"}),
            LogEntry("TableAgent", EntryType.LOOKUP, "operating income | 2016 = 10815",
                     metadata={"numeric": "10815"}),
        ]
        state = _build_state(operating_income_entry, log=log)
        result = summarizer_node(state)

        assert len(result["candidate_programs"]) == 5
        assert result["selected_program"] == "subtract(11503, 10815)"
        assert len(result["log"]) == 1
        assert result["log"][0].entry_type == EntryType.SUMMARY
        assert result["log"][0].agent == "SummarizingAgent"

    @patch("finqa_chatbot.agents.summarizer_agent.ChatOpenAI")
    def test_summarizer_handles_bad_responses(self, mock_chat_cls, operating_income_entry):
        """SummarizingAgent should handle garbled LLM responses."""
        mock_llm = MagicMock()
        responses = []
        for prog in [
            "subtract(11503, 10815)",          # valid
            "I don't know the answer",         # invalid
            "```\nsubtract(11503, 10815)\n```",  # markdown-wrapped
            "",                                # empty
            "subtract(11503, 10815)",          # valid
        ]:
            resp = MagicMock()
            resp.content = prog
            responses.append(resp)

        mock_llm.invoke.side_effect = responses
        mock_chat_cls.return_value = mock_llm

        state = _build_state(operating_income_entry, log=[])
        result = summarizer_node(state)

        # Should still have candidates (the valid ones)
        assert len(result["candidate_programs"]) >= 2
        # Selected should be a valid program
        assert "subtract(11503, 10815)" in result["selected_program"]


# ═══════════════════════════════════════════════════════════════════════
# 5. VerificationAgent
# ═══════════════════════════════════════════════════════════════════════

class TestVerificationAgent:
    """Test log-grounded verification checks."""

    def test_extract_literals_simple(self):
        lits = _extract_program_literals("subtract(11503, 10815)")
        assert "11503" in lits
        assert "10815" in lits

    def test_extract_literals_with_refs_and_consts(self):
        lits = _extract_program_literals("divide(637, const_5)")
        assert "637" in lits
        assert "const_5" not in lits  # constants excluded
        assert "const" not in " ".join(lits)

    def test_extract_literals_multi_step(self):
        lits = _extract_program_literals("subtract(7.0, 6.3), divide(#0, 6.3)")
        assert "7.0" in lits
        assert "6.3" in lits
        # #0 should NOT appear
        assert all(not l.startswith("#") for l in lits)

    def test_evidence_grounding_all_present(self):
        log = [
            LogEntry("TableAgent", EntryType.LOOKUP, "income | 2017 = 11503",
                     metadata={"numeric": "11503"}),
            LogEntry("TableAgent", EntryType.LOOKUP, "income | 2016 = 10815",
                     metadata={"numeric": "10815"}),
        ]
        grounded, missing = _check_evidence_grounding("subtract(11503, 10815)", log)
        assert grounded is True
        assert missing == []

    def test_evidence_grounding_missing_value(self):
        log = [
            LogEntry("TableAgent", EntryType.LOOKUP, "income | 2017 = 11503",
                     metadata={"numeric": "11503"}),
        ]
        grounded, missing = _check_evidence_grounding("subtract(11503, 10815)", log)
        assert grounded is False
        assert "10815" in missing

    def test_evidence_grounding_from_quote(self):
        """Numbers in QUOTE content should count as evidence."""
        log = [
            LogEntry("ContextAgent", EntryType.QUOTE,
                     "total net revenue increased $ 693 million to $ 7.0 billion"),
        ]
        grounded, missing = _check_evidence_grounding("divide(693, 7.0)", log)
        assert grounded is True

    def test_unit_consistency_both_percent(self):
        ok, issue = _check_unit_consistency("subtract(27.5%, 27.3%)")
        assert ok is True

    def test_unit_consistency_both_absolute(self):
        ok, issue = _check_unit_consistency("subtract(11503, 10815)")
        assert ok is True

    def test_unit_consistency_mixed(self):
        ok, issue = _check_unit_consistency("subtract(100, 27.3%)")
        assert ok is False
        assert "Mixing" in issue

    def test_temporal_consistency_match(self):
        log = [
            LogEntry("KGAgent", EntryType.KG_TRIPLET, "(Rev, HAS_VALUE_IN_2017, 100)",
                     metadata={"period": "2017"}),
            LogEntry("KGAgent", EntryType.KG_TRIPLET, "(Rev, HAS_VALUE_IN_2016, 90)",
                     metadata={"period": "2016"}),
        ]
        ok, issue = _check_temporal_consistency(
            "subtract(100, 90)", log,
            "what was the change in revenue from 2016 to 2017?"
        )
        assert ok is True

    def test_temporal_consistency_wrong_period(self):
        log = [
            LogEntry("KGAgent", EntryType.KG_TRIPLET, "(Rev, HAS_VALUE_IN_2015, 80)",
                     metadata={"period": "2015"}),
        ]
        ok, issue = _check_temporal_consistency(
            "subtract(100, 80)", log,
            "what was the change in revenue from 2016 to 2017?"
        )
        assert ok is False
        assert "2015" in issue

    def test_temporal_consistency_no_kg(self):
        """No KG triplets should always pass temporal check."""
        ok, issue = _check_temporal_consistency("divide(100, 50)", [], "question about 2017?")
        assert ok is True

    def test_temporal_consistency_no_years_in_question(self):
        """No years in question should always pass temporal check."""
        log = [
            LogEntry("KGAgent", EntryType.KG_TRIPLET, "(Rev, R, O)", metadata={"period": "2017"}),
        ]
        ok, issue = _check_temporal_consistency("divide(100, 50)", log, "what is the total?")
        assert ok is True

    @patch("finqa_chatbot.agents.verification_agent.ChatOpenAI")
    def test_verification_node_ok(self, mock_chat_cls, operating_income_entry):
        """Verification should return OK when everything checks out."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "STATUS: OK"
        mock_llm.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_llm

        log = [
            LogEntry("TableAgent", EntryType.LOOKUP, "operating income | 2017 = 11503",
                     metadata={"numeric": "11503"}),
            LogEntry("TableAgent", EntryType.LOOKUP, "operating income | 2016 = 10815",
                     metadata={"numeric": "10815"}),
            LogEntry("SummarizingAgent", EntryType.SUMMARY,
                     "Generated 5 candidates, selected: subtract(11503, 10815)"),
        ]
        state = _build_state(
            operating_income_entry,
            log=log,
            selected_program="subtract(11503, 10815)",
            exe_result=688.0,
            exe_invalid=False,
        )
        result = verification_node(state)

        assert result["verification_status"] == "OK"
        assert result["flag_targets"] == []
        assert len(result["log"]) == 1
        assert result["log"][0].entry_type == EntryType.OK
        assert result["log"][0].agent == "VerificationAgent"

    def test_verification_node_exe_failure(self, operating_income_entry):
        """Verification should FLAG when execution failed."""
        state = _build_state(
            operating_income_entry,
            log=[],
            selected_program="bad_program",
            exe_result="n/a",
            exe_invalid=True,
        )
        result = verification_node(state)

        assert result["verification_status"] == "FLAG"
        assert len(result["flag_targets"]) > 0
        assert result["log"][0].entry_type == EntryType.FLAG

    @patch("finqa_chatbot.agents.verification_agent.ChatOpenAI")
    def test_verification_node_missing_evidence_flags(self, mock_chat_cls, operating_income_entry):
        """Verification should FLAG when evidence is missing."""
        # Empty log — no evidence for the program's numbers
        state = _build_state(
            operating_income_entry,
            log=[],  # no evidence!
            selected_program="subtract(11503, 10815)",
            exe_result=688.0,
            exe_invalid=False,
        )
        result = verification_node(state)

        assert result["verification_status"] == "FLAG"
        assert "table_agent" in result["flag_targets"] or "context_agent" in result["flag_targets"]

    @patch("finqa_chatbot.agents.verification_agent.ChatOpenAI")
    def test_verification_node_llm_flags(self, mock_chat_cls, operating_income_entry):
        """Verification should FLAG when LLM cross-check finds issues."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "STATUS: FLAG\nISSUE: Wrong operation\nTARGETS: SummarizingAgent"
        mock_llm.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_llm

        log = [
            LogEntry("TableAgent", EntryType.LOOKUP, "operating income | 2017 = 11503",
                     metadata={"numeric": "11503"}),
            LogEntry("TableAgent", EntryType.LOOKUP, "operating income | 2016 = 10815",
                     metadata={"numeric": "10815"}),
        ]
        state = _build_state(
            operating_income_entry,
            log=log,
            selected_program="subtract(11503, 10815)",
            exe_result=688.0,
            exe_invalid=False,
        )
        result = verification_node(state)

        assert result["verification_status"] == "FLAG"
        assert "summarizer" in result["flag_targets"]
