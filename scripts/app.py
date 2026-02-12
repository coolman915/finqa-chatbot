#!/usr/bin/env python3
"""Streamlit chatbot UI for the FinQA DeALOG pipeline.

Launch:
    streamlit run scripts/app.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

import streamlit as st

from finqa_chatbot.config import get_settings
from finqa_chatbot.pipeline import load_dataset, run_single
from finqa_chatbot.evaluation.official import _relaxed_equal

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="FinQA Chatbot", page_icon="📊", layout="wide")
st.title("📊 FinQA Chatbot")
st.caption("Agentic DSL synthesis for numerical reasoning over financial documents")

settings = get_settings()

# ── Sidebar: configuration display ───────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    st.markdown(f"**Model:** `{settings.model_name}`")
    st.markdown(f"**Temperature:** `{settings.temperature}`")
    st.markdown(f"**Max rounds:** `{settings.max_rounds}`")
    st.markdown(f"**Candidates:** `{settings.num_candidates}`")
    st.markdown(f"**Few-shot examples:** `{settings.num_few_shot}`")
    st.divider()
    st.markdown("**Mode**")
    mode = st.radio("Select mode", ["Browse dataset", "Free-form question"],
                    label_visibility="collapsed")

# ── Mode: Browse dataset ─────────────────────────────────────────────
if mode == "Browse dataset":
    col1, col2 = st.columns([1, 3])
    with col1:
        split = st.selectbox("Split", ["dev", "test"])
    data = load_dataset(split)
    with col2:
        index = st.number_input("Example index", min_value=0,
                                max_value=len(data) - 1, value=0, step=1)

    entry = data[index]
    st.subheader(f"Entry: `{entry['id']}`")

    # Show question and gold answer
    st.markdown(f"**Question:** {entry['qa']['question']}")
    gold_prog = entry["qa"]["program"]
    gold_ans = entry["qa"]["exe_ans"]
    st.markdown(f"**Gold program:** `{gold_prog}`")
    st.markdown(f"**Gold answer:** `{gold_ans}`")

    # Show table
    with st.expander("Table", expanded=True):
        table = entry["table"]
        if table:
            header = table[0]
            rows = table[1:]
            st.table(dict(zip(header, zip(*rows))) if rows else {})

    # Show text context
    pre = entry.get("pre_text", [])
    post = entry.get("post_text", [])
    if pre or post:
        with st.expander("Text context"):
            if pre:
                st.markdown("**Pre-text:**")
                for s in pre:
                    if s.strip() and s.strip() != ".":
                        st.markdown(f"- {s}")
            if post:
                st.markdown("**Post-text:**")
                for s in post:
                    if s.strip() and s.strip() != ".":
                        st.markdown(f"- {s}")

    # Run pipeline
    if st.button("Run Pipeline", type="primary", key="run_browse"):
        with st.spinner("Running DeALOG pipeline..."):
            result = run_single(entry)

        st.divider()
        st.subheader("Result")

        pred_prog = result.get("raw_program", "")
        pred_ans = result.get("exe_result", "n/a")
        rounds = result.get("rounds_used", "?")

        st.markdown(f"**Predicted program:** `{pred_prog}`")
        st.markdown(f"**Predicted answer:** `{pred_ans}`")
        st.markdown(f"**Rounds used:** {rounds}")

        # Correctness check
        correct = _relaxed_equal(pred_ans, gold_ans)
        if correct:
            st.success(f"CORRECT  —  predicted `{pred_ans}` matches gold `{gold_ans}`")
        else:
            st.error(f"INCORRECT  —  predicted `{pred_ans}`, expected `{gold_ans}`")

        if "error" in result:
            st.warning(f"Pipeline error: {result['error']}")

# ── Mode: Free-form question ─────────────────────────────────────────
else:
    st.markdown("Paste a financial table and ask a question.")

    table_input = st.text_area(
        "Table (paste CSV — comma or tab separated, one row per line)",
        height=200,
        placeholder="Header1, Header2, Header3\nRow1Val1, Row1Val2, Row1Val3\n...",
    )
    pre_text = st.text_area("Context text (optional, one sentence per line)",
                            height=100)
    question = st.text_input("Question")

    if st.button("Ask", type="primary", key="run_freeform"):
        if not table_input.strip() or not question.strip():
            st.warning("Please provide both a table and a question.")
        else:
            # Parse table input
            sep = "\t" if "\t" in table_input else ","
            table = []
            for line in table_input.strip().splitlines():
                row = [cell.strip() for cell in line.split(sep)]
                table.append(row)

            pre_lines = [s.strip() for s in pre_text.strip().splitlines()
                         if s.strip()] if pre_text.strip() else []

            # Build a synthetic entry
            entry = {
                "id": "freeform",
                "qa": {
                    "question": question,
                    "program": "",
                    "exe_ans": None,
                },
                "table": table,
                "pre_text": pre_lines,
                "post_text": [],
            }

            with st.spinner("Running DeALOG pipeline..."):
                result = run_single(entry)

            st.divider()
            st.subheader("Result")

            pred_prog = result.get("raw_program", "")
            pred_ans = result.get("exe_result", "n/a")
            rounds = result.get("rounds_used", "?")

            st.markdown(f"**Predicted program:** `{pred_prog}`")
            st.markdown(f"**Predicted answer:** `{pred_ans}`")
            st.markdown(f"**Rounds used:** {rounds}")

            if "error" in result:
                st.warning(f"Pipeline error: {result['error']}")
