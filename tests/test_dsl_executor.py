"""Test DSL executor on all 883 dev gold programs — must be 100%."""

import pytest

from finqa_chatbot.dsl.executor import eval_program
from finqa_chatbot.dsl.parser import parse_program_to_tokens
from finqa_chatbot.evaluation.official import program_tokenization


def test_simple_divide():
    tokens = parse_program_to_tokens("divide(637, const_5)")
    flag, result = eval_program(tokens, [])
    assert flag == 0
    assert result == 127.4


def test_simple_subtract():
    tokens = parse_program_to_tokens("subtract(11503, 10815)")
    flag, result = eval_program(tokens, [])
    assert flag == 0
    assert result == 688.0


def test_multi_step():
    tokens = parse_program_to_tokens("subtract(7.0, 6.3), divide(#0, 6.3)")
    flag, result = eval_program(tokens, [])
    assert flag == 0
    assert result == 0.11111


def test_percentage():
    tokens = parse_program_to_tokens("subtract(27.5, 27.3)")
    flag, result = eval_program(tokens, [])
    assert flag == 0


def test_table_sum(sample_table):
    tokens = parse_program_to_tokens("table_sum(visa, NONE)")
    flag, result = eval_program(tokens, sample_table)
    assert flag == 0


def test_invalid_program():
    tokens = ["badop(", "1", "2", ")", "EOF"]
    flag, result = eval_program(tokens, [])
    assert flag == 1


def test_all_dev_gold_programs(dev_data):
    """Every gold program in the dev set must execute successfully."""
    failures = []
    for entry in dev_data:
        program_str = entry["qa"]["program"]
        gold_res = entry["qa"]["exe_ans"]
        tokens = program_tokenization(program_str)
        flag, result = eval_program(tokens, entry["table"])

        if flag != 0:
            failures.append((entry["id"], program_str, "INVALID"))
        elif result != gold_res:
            failures.append((entry["id"], program_str, f"got {result}, expected {gold_res}"))

    assert len(failures) == 0, (
        f"{len(failures)}/{len(dev_data)} gold programs failed:\n"
        + "\n".join(f"  {eid}: {prog} ({err})" for eid, prog, err in failures[:10])
    )
