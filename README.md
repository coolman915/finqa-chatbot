# FinQA Chatbot

Agentic DSL synthesis system for numerical reasoning over financial documents, built on the [FinQA dataset](https://arxiv.org/abs/2109.00122).

**Key results:** 80.4% execution accuracy, 74.7% program accuracy (with LLM judge) on the 883-example dev set using `gpt-5-nano`.

## Architecture

The system uses a **LangGraph state machine** implementing the DeALOG (Decentralized Agents with Logs) multi-agent protocol:

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
```

### 3. Build embeddings cache (optional, improves few-shot selection)

```bash
python scripts/build_embeddings_cache.py
```

### 4. Run your first example

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
python scripts/run_eval.py --split dev --max_examples 100 --workers 8

# Full evaluation with LLM judge
python scripts/run_eval.py --split dev --workers 8 --llm-judge

# Resume interrupted run
python scripts/run_eval.py --split dev --resume --workers 8
```

### Quick benchmarks

```bash
python scripts/run_5_live.py       # 5 examples, quick sanity check
python scripts/run_20_examples.py  # 20 examples, broader coverage
```

### Failure analysis

```bash
python scripts/analyze_failures_full.py              # analyze output/predictions_dev.json
python scripts/test_failures.py --reeval              # re-evaluate specific failures
```

### Embeddings cache

```bash
python scripts/build_embeddings_cache.py              # build OpenAI embedding cache (.npy)
```

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
│   │   ├── metrics.py          # Extended metrics and error breakdown
│   │   └── langsmith_eval.py   # LangSmith evaluation dataset upload
│   ├── graph/
│   │   ├── workflow.py         # LangGraph StateGraph construction
│   │   ├── scheduler.py        # DeALOG round management and routing
│   │   ├── callbacks.py        # LangSmith tracing callback
│   │   └── state.py            # GraphState type definition
│   ├── prompts/
│   │   ├── summarizer.py       # DSL synthesis prompt templates
│   │   ├── kg_extraction.py    # KG triplet extraction prompts
│   │   ├── verification.py     # Verification prompts
│   │   └── system.py           # System prompts
│   └── retrieval/              # Embedding-based retrieval utilities
├── scripts/
│   ├── app.py                  # Streamlit chatbot UI
│   ├── run_single.py           # Single example CLI
│   ├── run_one_explained.py    # Step-by-step verbose demo
│   ├── run_eval.py             # Batch evaluation CLI
│   ├── run_5_live.py           # Quick 5-example benchmark
│   ├── run_20_examples.py      # 20-example benchmark
│   ├── build_embeddings_cache.py  # OpenAI embeddings cache builder
│   ├── analyze_failures_full.py   # Failure analysis script
│   └── test_failures.py        # Targeted failure re-evaluation
├── tests/
│   ├── test_dsl_executor.py    # DSL executor tests (100% on 883 gold programs)
│   ├── test_agents.py          # Agent unit tests
│   ├── test_graph.py           # Graph integration tests
│   ├── test_verification.py    # Verification agent tests
│   └── test_kg_extraction.py   # KG extraction tests
├── data/                       # FinQA dataset (dev.json, test.json, train.json)
├── output/                     # Evaluation results and predictions
├── docs/
│   └── technical_report.md     # Technical report
└── pyproject.toml              # Package configuration
```

## Results

| Metric | Value | Count |
|--------|-------|-------|
| Execution accuracy | 80.4% | 710 / 883 |
| Program accuracy | 67.3% | 594 / 883 |
| Program accuracy (LLM judge) | 74.7% | ~660 / 883 |
| Invalid programs | ~3% | ~27 / 883 |
| Average rounds | ~1.8 | — |

**Model:** gpt-5-nano | **Max rounds:** 6 | **Temperature:** 0.0 | **Candidates:** 5

### Evaluation methodology

- **exe_acc**: Does the predicted program produce the correct numerical answer? Uses relaxed matching with sign tolerance, scale factor correction (10x/100x/1000x), and 5% relative tolerance (verified zero false positives on dev set).
- **prog_acc**: Is the predicted program structurally equivalent to the gold program? Uses symbolic comparison, const_100 normalization, trailing-step stripping, and same-ops matching.
- **LLM judge**: For cases where exe_acc passes but prog_acc fails, an LLM judges whether the programs are semantically equivalent.

See [`docs/technical_report.md`](docs/technical_report.md) for the full technical report.
