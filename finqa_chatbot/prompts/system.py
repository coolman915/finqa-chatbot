"""System prompt and few-shot examples for DSL program generation."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a financial analyst that answers questions by writing executable DSL programs.

Available operations (each takes exactly 2 arguments, except table ops which take 1):
- add(arg1, arg2): addition
- subtract(arg1, arg2): subtraction
- multiply(arg1, arg2): multiplication
- divide(arg1, arg2): division
- exp(arg1, arg2): exponentiation (arg1 ** arg2)
- greater(arg1, arg2): returns "yes" if arg1 > arg2, else "no"
- table_sum(table_row_name, NONE): sum all numeric values in that table row
- table_average(table_row_name, NONE): average of all numeric values in that row
- table_max(table_row_name, NONE): max of all numeric values in that row
- table_min(table_row_name, NONE): min of all numeric values in that row

Available constants: const_2, const_3, const_4, const_5, const_6, const_7, const_8, const_9, const_10, const_100, const_1000, const_10000, const_100000, const_1000000, const_10000000, const_1000000000, const_m1, const_1

Arguments can be:
- Numbers from the provided text/table (use exact string as it appears, e.g. "5829" or "27.3" or "27.3%")
- Constants like const_100
- References to previous step results: #0, #1, #2, etc.
- Table row names (for table operations)

Values field: All numeric values mentioned in the context are listed below for easy reference.

Output format: ONLY output the program as a comma-separated sequence of operations.
Example outputs:
- divide(637, const_5)
- subtract(5829, 5735)
- divide(9413, 20.01), divide(8249, 9.48), subtract(#0, #1)
- add(100, 200), divide(#0, 300)

IMPORTANT RULES:
1. Output ONLY the program, nothing else. No explanation, no "Answer:", no extra text.
2. Use exact numbers as they appear in the context (e.g. "5,829" should be written as "5829" without commas).
3. For percentage values like "27.3%", use "27.3%" as the argument.
4. Use #0, #1, etc. to reference results from previous steps (0-indexed).
5. For table operations, the first argument is the row name (text in the first column).

Common patterns:
- Percent change: subtract(new, old), divide(#0, old) — do NOT multiply by const_100 (percentages are already in decimal form)
- Difference: subtract(val1, val2)
- Ratio: divide(val1, val2)
- Growth rate: subtract(new, old), divide(#0, old)
"""

FEW_SHOT_EXAMPLES = [
    {
        "context": "Table:\n| company | payments volume ( billions ) | total transactions ( billions ) |\n| visa | $ 2457 | 50.3 |\n| mastercard | 1697 | 27.0 |\n| american express | 637 | 5.0 |",
        "question": "what is the average payment volume per transaction for american express?",
        "program": "divide(637, const_5)",
    },
    {
        "context": "Text: The total fair value of shares vested during 2006, 2005, and 2004 was $9,413, $8,249, and $6,418 respectively.\nTable:\n| | 2006 | 2005 | 2004 |\n| weighted average fair value of options granted | $20.01 | $9.48 | $7.28 |",
        "question": "Considering the weighted average fair value of options, what was the change of shares vested from 2005 to 2006?",
        "program": "divide(9413, 20.01), divide(8249, 9.48), subtract(#0, #1)",
    },
    {
        "context": "Table:\n| ( in millions ) | 2017 | 2016 |\n| operating income | 11503 | 10815 |",
        "question": "what was the change in millions of operating income from 2016 to 2017?",
        "program": "subtract(11503, 10815)",
    },
    {
        "context": "Table:\n| | 2004 | 2003 |\n| gross margin percentage | 27.3% | 27.5% |",
        "question": "what was the gross margin decline in fiscal 2004 from 2003?",
        "program": "subtract(27.5, 27.3)",
    },
    {
        "context": "Text: total net revenue increased $ 693 million , or 11% ( 11 % ) , to $ 7.0 billion .\nTable:\n| year | net revenue |\n| 2006 | 7.0 |\n| 2005 | 6.3 |",
        "question": "what is the percent change in total net revenue from 2005 to 2006?",
        "program": "subtract(7.0, 6.3), divide(#0, 6.3)",
    },
]


def format_context(entry: dict, retrieved_facts: list[tuple[str, str]] | None = None) -> str:
    """Build context string from entry and optionally retrieved facts."""
    table_parts = []
    text_parts = []

    if retrieved_facts:
        for ind_key, text in retrieved_facts:
            if "table" in ind_key:
                table_parts.append(text)
            else:
                text_parts.append(text)

    # Include all pre/post text
    all_text = entry.get("pre_text", []) + entry.get("post_text", [])
    if not text_parts:
        # Use full text if no retrieval was done
        text_parts = [s for s in all_text if s.strip() and s.strip() != "."]

    context = ""
    if text_parts:
        context += "Text: " + " ".join(text_parts) + "\n"

    # Always include the full table
    if entry.get("table"):
        header = entry["table"][0]
        context += "Table:\n| " + " | ".join(header) + " |\n"
        for row in entry["table"][1:]:
            context += "| " + " | ".join(row) + " |\n"

    return context.strip()


def build_messages(
    entry: dict,
    retrieved_facts: list[tuple[str, str]] | None = None,
    few_shot_examples: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for the LLM."""
    context = format_context(entry, retrieved_facts)
    question = entry["qa"]["question"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    examples = few_shot_examples or FEW_SHOT_EXAMPLES
    for ex in examples:
        messages.append({
            "role": "user",
            "content": f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}",
        })
        messages.append({"role": "assistant", "content": ex["program"]})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })
    return messages
