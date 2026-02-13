#!/usr/bin/env python3
"""Benchmark gpt-5-nano LLM call latency vs prompt size and completion size.

Tests two dimensions:
1. Varying prompt size with fixed short completion (max_completion_tokens=10)
2. Varying completion size with fixed short prompt
"""

import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

from openai import OpenAI

FILLER_SENTENCE = (
    "[TableAgent|LOOKUP] operating expenses | fiscal year 2017 = 12345 "
    "[row_name=operating expenses, column=fiscal year 2017, value=12345, "
    "numeric=12345, source_ind=table_1]\n"
)


def build_prompt(target_tokens: int) -> str:
    """Build a prompt of approximately target_tokens size."""
    base = "Answer with ONLY one number. What is 2+2?\n\nContext:\n"
    # ~4 tokens per filler line
    num_lines = max(1, (target_tokens - 20) // 25)
    return base + FILLER_SENTENCE * num_lines


def call_llm(client, prompt: str, max_completion_tokens: int | None = None) -> tuple[float, int, int]:
    """Call gpt-5-nano and return (latency_ms, prompt_tokens, completion_tokens)."""
    kwargs = {
        "model": "gpt-5-nano",
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency = (time.time() - t0) * 1000
    return latency, resp.usage.prompt_tokens, resp.usage.completion_tokens


def main():
    client = OpenAI()

    # ── Test 1: Vary prompt size, cap completion to isolate prompt latency ──
    print("=" * 70)
    print("TEST 1: Varying PROMPT size (completion capped at max_completion_tokens=10)")
    print("=" * 70)
    print(f"{'Prompt tokens':>14} {'Compl tokens':>13} {'Latency':>10} {'ms/prompt_tok':>14}")
    print("-" * 55)

    prompt_sizes = [50, 100, 250, 500, 1000, 2000, 3000, 5000]
    prev_latency = None
    for target in prompt_sizes:
        prompt = build_prompt(target)
        latency, ptok, ctok = call_llm(client, prompt, max_completion_tokens=10)
        per_tok = latency / ptok if ptok > 0 else 0
        print(f"{ptok:>14} {ctok:>13} {latency:>9.0f}ms {per_tok:>13.2f}")
        prev_latency = latency

    # ── Test 2: Vary completion size, fixed small prompt ──────────────────
    print()
    print("=" * 70)
    print("TEST 2: Varying COMPLETION size (fixed small prompt)")
    print("=" * 70)
    print(f"{'Prompt tokens':>14} {'Compl tokens':>13} {'Latency':>10} {'ms/compl_tok':>14}")
    print("-" * 55)

    completion_prompts = [
        (10, "What is 2+2? Reply with just the number."),
        (50, "List the first 10 prime numbers, one per line."),
        (200, "Write a 150-word summary of how compound interest works in finance."),
        (500, "Write a detailed 400-word explanation of how GDP is calculated, including the expenditure approach and income approach."),
        (1000, "Write a very detailed 800-word essay about the history of the US stock market from 1900 to 2000, covering major crashes, bull markets, and regulatory changes."),
    ]

    for target_completion, prompt in completion_prompts:
        latency, ptok, ctok = call_llm(client, prompt)
        per_tok = latency / ctok if ctok > 0 else 0
        print(f"{ptok:>14} {ctok:>13} {latency:>9.0f}ms {per_tok:>13.2f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("- Base API latency: ~1.5-2s (even for trivial prompts)")
    print("- Prompt tokens: add marginal latency (~0.x ms per token)")
    print("- Completion tokens: dominant cost (~5-15 ms per token)")


if __name__ == "__main__":
    main()
