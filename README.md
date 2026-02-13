# FinQA Chatbot

Agentic DSL synthesis system for numerical reasoning over financial documents, built on the [FinQA dataset](https://arxiv.org/abs/2109.00122).

**Key results:** 76.0% execution accuracy, 82.8% adjusted accuracy (with LLM eval) on the 883-example dev set using `gpt-5-nano`.

## Architecture

The system uses a **LangGraph state machine** implementing a decentralized multi-agent protocol with a shared log:

```
                    ┌─────────┐
                    │  Init   │
                    └────┬────┘
                         ▼
                   ┌───────────┐
              ┌────│ Scheduler │◄────────────────┐
              │    └───────────┘                  │
              ▼         ▼          ▼              │
        ┌──────────┐ ┌─────────┐ ┌────────┐      │
        │  Table   │ │ Context │ │   KG   │      │
        │  Agent   │ │  Agent  │ │ Agent  │      │
        └────┬─────┘ └────┬────┘ └───┬────┘      │
             └─────────┬──┘──────────┘            │
                       ▼                          │
                ┌─────────────┐                   │
                │ Summarizer  │                   │
                │(self-consist│ency)              │
                └──────┬──────┘                   │
                       ▼                          │
                 ┌──────────┐                     │
                 │ Executor │                     │
                 │  (DSL)   │                     │
                 └────┬─────┘                     │
                      ▼                           │
                ┌───────────┐    FLAG             │
                │ Verifier  │─────────────────────┘
                └─────┬─────┘
                      │ OK
                      ▼
                 ┌──────────┐
                 │ Finalize │
                 └──────────┘
```

**Agents:**

| Agent | Role | LLM? |
|-------|------|------|
| **TableAgent** | Extracts structured lookups from the financial table | No |
| **ContextAgent** | Retrieves relevant text passages via TF-IDF | No |
| **KGAgent** | Extracts knowledge-graph triplets from context | Yes |
| **Summarizer** | Synthesizes DSL programs (5 candidates, majority vote) | Yes |
| **Executor** | Runs the DSL program deterministically | No |
| **Verifier** | Validates results (evidence grounding, arithmetic, temporal, unit checks) | Partial |

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd finqa-chatbot
pip install -e .            # core dependencies
pip install -e ".[dev]"     # + pytest
pip install -e ".[ui]"      # + streamlit
```

### 2. Configure

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-5-nano       # optional, default: gpt-5-nano
MAX_ROUNDS=6                 # optional, default: 3
NUM_CANDIDATES=5             # optional, default: 5
LANGCHAIN_API_KEY=ls-...     # optional, for LangSmith tracing
MONGODB_URI=mongodb://localhost:27017  # optional, for tracing & result storage
```

### 3. MongoDB setup (optional, for tracing & result storage)

MongoDB stores run results, predictions, and per-node step-level traces. The system works without it, but tracing and batch evaluation features require it.

```bash
# macOS (Homebrew)
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Docker
docker run -d --name mongodb -p 27017:27017 mongodb/mongodb-community-server:latest

# Verify connection
python -c "from pymongo import MongoClient; print(MongoClient('mongodb://localhost:27017').server_info()['version'])"
```

When MongoDB is available, the system automatically:
- Saves run metadata, predictions, and evaluation results to the `finqa` database
- Records per-node step-level traces (timing, outputs, LLM token usage) for debugging
- Supports resume on interrupted batch runs (`--resume` flag)

### 4. Build embeddings cache (optional, improves few-shot selection)

```bash
python scripts/build_embeddings_cache.py
```

### 5. Run your first example

```bash
python scripts/run_single.py --index 5
```

## CLI Command Reference

### Interactive chatbot UI

```bash
streamlit run scripts/app.py
```

Opens a web UI with two modes: browse dataset examples or paste custom tables with free-form questions.

### Single example

```bash
python scripts/run_single.py --index 5
python scripts/run_single.py --entry_id "Single/2015/page_38.pdf-2"
python scripts/run_single.py --split test --index 0
```

Each run saves step-level traces to MongoDB (if configured), showing per-node timing, outputs, and LLM token usage.

### Step-by-step demo

```bash
python scripts/run_one_explained.py
```

Runs one example with verbose output showing every agent step, shared log contents, and verification details.

### Batch evaluation

```bash
# Run a specific range
python scripts/run_eval.py --split dev --start 0 --end 50 --workers 4

# Run first N examples
python scripts/run_eval.py --split dev --max_examples 100 --workers 16

# Full evaluation with LLM judge
python scripts/run_eval.py --split dev --workers 16 --llm-judge

# Resume interrupted run
python scripts/run_eval.py --split dev --resume --workers 16
```

Tested with up to 100 concurrent workers with zero API errors.

### MongoDB tracing & debugging

Step-level traces capture per-node execution details (timing, outputs, LLM token usage, errors) similar to LangSmith:

```bash
# View step-by-step trace for a specific example
python scripts/query_results.py trace <run_id> <entry_id>

# View per-node aggregate stats (avg/max latency, token usage, error count)
python scripts/query_results.py node-stats <run_id>

# View run summary and failures
python scripts/query_results.py failures <run_id>
```

### LLM failure evaluation

When predictions fail, an LLM (gpt-4o) can judge whether the prediction is actually correct (catches false negatives) and classify the failure reason.

```bash
# During pipeline — evaluate failures in real-time
python scripts/run_eval.py --split dev --max_examples 20 --llm-eval

# Post-processing — evaluate stored failures after the fact
python scripts/query_results.py evaluate <run_id>

# View failures with LLM classifications
python scripts/query_results.py failures <run_id>
```

Failure categories: `correct_alternate`, `wrong_number`, `wrong_computation`, `sign_error`, `scale_error`, `missing_step`, `extra_step`, `wrong_approach`, `invalid_program`, `rounding_error`.

### Quick benchmarks

```bash
python scripts/run_5_live.py       # 5 examples, quick sanity check
python scripts/run_20_examples.py  # 20 examples, broader coverage
```

### LLM latency benchmark

```bash
python scripts/benchmark_llm_latency.py
```

Measures gpt-5-nano latency vs prompt size and completion size. Key finding: prompt tokens add negligible latency (~0.x ms/token), while completion tokens dominate at ~8-14 ms/token.

### Failure analysis

```bash
python scripts/analyze_failures_full.py              # analyze output/predictions_dev.json
python scripts/test_failures.py --reeval              # re-evaluate specific failures
```

### Embeddings cache

```bash
python scripts/build_embeddings_cache.py              # build OpenAI embedding cache (.npy)
```

### Dataset analysis

```bash
jupyter notebook notebooks/dataset_analysis.ipynb
```

Interactive notebook with 11 sections: split statistics, question type classification, program/operation analysis, table characteristics, text context, answer distribution, complexity analysis, evidence source & fact analysis (reproduces FinQA paper statistics), company analysis, and summary.

### Test suite

```bash
pytest tests/                                          # all tests
pytest tests/test_dsl_executor.py -v                   # DSL executor (883 gold programs)
pytest tests/test_agents.py -v                         # agent unit tests
pytest tests/test_graph.py -v                          # graph integration tests
```

## Project Structure

```
finqa-chatbot/
├── finqa_chatbot/              # Main package
│   ├── config.py               # Pydantic settings (env/dotenv)
│   ├── pipeline.py             # run_single / run_batch entry points
│   ├── schema.py               # GraphState, LogEntry, KGTriplet types
│   ├── agents/
│   │   ├── table_agent.py      # Deterministic table value extraction
│   │   ├── context_agent.py    # TF-IDF text passage retrieval
│   │   ├── kg_agent.py         # LLM-based KG triplet extraction
│   │   ├── summarizer_agent.py # DSL program synthesis (self-consistency)
│   │   └── verification_agent.py  # Multi-check result verification
│   ├── dsl/
│   │   ├── executor.py         # DSL program evaluator (10 operations)
│   │   ├── operations.py       # Operation definitions (add, subtract, ...)
│   │   └── parser.py           # Program string → token list parser
│   ├── evaluation/
│   │   ├── official.py         # exe_acc, prog_acc, relaxed matching, LLM judge
│   │   ├── llm_eval.py         # LLM failure evaluation and classification (gpt-4o)
│   │   └── langsmith_eval.py   # LangSmith evaluation dataset upload
│   ├── graph/
│   │   ├── workflow.py         # LangGraph StateGraph construction
│   │   ├── scheduler.py        # Round management and agent routing
│   │   ├── callbacks.py        # Per-node step-level tracing (timing, outputs, LLM tokens)
│   │   └── state.py            # GraphState type definition
│   ├── prompts/
│   │   ├── summarizer.py       # DSL synthesis prompt templates (14 few-shot examples)
│   │   ├── kg_extraction.py    # KG triplet extraction prompts
│   │   ├── verification.py     # Verification prompts
│   │   └── system.py           # System prompts
│   ├── retrieval/              # Embedding-based retrieval utilities
│   └── storage/
│       └── mongodb.py          # MongoDB store (runs, predictions, traces)
├── scripts/
│   ├── app.py                  # Streamlit chatbot UI
│   ├── run_single.py           # Single example CLI (with MongoDB tracing)
│   ├── run_one_explained.py    # Step-by-step verbose demo
│   ├── run_eval.py             # Batch evaluation CLI
│   ├── run_5_live.py           # Quick 5-example benchmark
│   ├── run_20_examples.py      # 20-example benchmark
│   ├── benchmark_llm_latency.py   # LLM latency vs prompt/completion size
│   ├── build_embeddings_cache.py  # OpenAI embeddings cache builder
│   ├── query_results.py           # MongoDB query CLI (runs, failures, trace, node-stats)
│   ├── load_dataset_mongo.py      # Load dataset into MongoDB
│   ├── analyze_failures_full.py   # Failure analysis script
│   └── test_failures.py        # Targeted failure re-evaluation
├── tests/
│   ├── test_dsl_executor.py    # DSL executor tests (100% on 883 gold programs)
│   ├── test_agents.py          # Agent unit tests
│   ├── test_graph.py           # Graph integration tests
│   ├── test_verification.py    # Verification agent tests
│   └── test_kg_extraction.py   # KG extraction tests
├── notebooks/
│   └── dataset_analysis.ipynb  # Dataset statistics and characteristics (11 sections)
├── data/                       # FinQA dataset (dev.json, test.json, train.json)
├── output/                     # Evaluation results and predictions
├── docs/
│   ├── technical_report.md     # Technical report
│   └── papers/                 # Reference papers (FinQA, DeALOG, KG-reasoning)
└── pyproject.toml              # Package configuration
```

## Results

| Metric | Value | Count |
|--------|-------|-------|
| Execution accuracy | 76.0% | 671 / 883 |
| Program accuracy | 67.0% | 592 / 883 |
| Adjusted exe_acc (LLM eval) | 82.8% | 731 / 883 |
| Invalid programs | 0.2% | 2 / 883 |
| Average rounds | 1.10 | — |

**Model:** gpt-5-nano | **Max rounds:** 5 | **Temperature:** 0.0 | **Candidates:** 5

### Evaluation methodology

- **exe_acc**: Does the predicted program produce the correct numerical answer? Uses relaxed matching with sign tolerance, scale factor correction (10x/100x/1000x), and 5% relative tolerance (verified zero false positives on dev set).
- **prog_acc**: Is the predicted program structurally equivalent to the gold program? Uses symbolic comparison, const_100 normalization, trailing-step stripping, and same-ops matching.
- **LLM judge**: For cases where exe_acc passes but prog_acc fails, an LLM judges whether the programs are semantically equivalent.
- **LLM failure evaluation**: For predictions that fail exe_acc or prog_acc, a gpt-4o call judges whether the prediction is actually correct (alternate valid approach) and classifies the failure reason into one of 10 categories. Available via `--llm-eval` flag or post-hoc via `query_results.py evaluate`.

See [`docs/technical_report.md`](docs/technical_report.md) for the full technical report.

## References

Research papers used in the design of this system (included in `docs/papers/`):

1. **FinQA: A Dataset of Numerical Reasoning over Financial Data** — Chen et al., 2022 ([arXiv:2109.00122](https://arxiv.org/abs/2109.00122))
   The dataset and DSL this system is built on. Defines the 10-operation DSL, gold program annotations, and evaluation methodology (exe_acc, prog_acc).

2. **DeALOG: Decentralized Multi-Agents Log-Mediated Reasoning Framework** — Chakraborty et al., 2026 ([arXiv:2602.00996](https://arxiv.org/abs/2602.00996))
   The multi-agent architecture pattern we adopt. Introduces specialized agents (Table, Context, Visual, Summarizing, Verification) coordinating through a shared natural-language log rather than a central planner.

3. **Structure First, Reason Next: Enhancing a Large Language Model using Knowledge Graph for Numerical Reasoning in Financial Documents** — Mishra & Anil, 2026 ([arXiv:2601.07754](https://arxiv.org/abs/2601.07754))
   Motivates our KGAgent design. Demonstrates that extracting Knowledge Graphs from financial documents and feeding structured information alongside LLM predictions improves numerical reasoning accuracy by ~12% over vanilla LLMs.
