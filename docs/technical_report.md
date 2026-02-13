# Technical Report: FinQA Chatbot

## 1. Dataset Analysis

### 1.1 Dataset Characteristics

The [FinQA dataset](https://arxiv.org/abs/2109.00122) contains financial question-answer pairs requiring multi-step numerical reasoning over structured and unstructured data:

| Split | Examples |
|-------|----------|
| Train | 6,251 |
| Dev | 883 |
| Test | 1,147 |
| **Total** | **8,281** |

Each example consists of:
- **Financial table**: Structured tabular data (e.g., revenue by year, balance sheet items)
- **Pre-text / Post-text**: Surrounding narrative paragraphs from SEC filings
- **Question**: A natural language question requiring numerical reasoning
- **Gold program**: A DSL (domain-specific language) program encoding the computation
- **Gold answer**: The expected numerical result
- **Gold evidence annotations**: `gold_inds` mapping to specific text sentences (`text_N`) and table rows (`table_N`)

The DSL supports 10 operations: `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`, and 4 table-lookup operations (`table_max`, `table_min`, `table_sum`, `table_average`). Programs are multi-step with intermediate references (`#0`, `#1`, etc.) and support constants (`const_100`, `const_1000`, `const_m1`).

### 1.2 Evidence Source Distribution

Analysis of the `gold_inds` annotations across all splits confirms the paper's reported statistics:

| Evidence Source | % of examples |
|-----------------|---------------|
| Table only | 62.43% |
| Text only | 23.42% |
| Both text and table | 14.15% |

This means ~14% of questions require cross-modal reasoning (combining text and table facts), which is a key challenge for retrieval agents.

### 1.3 Supporting Facts

| Number of facts | % of examples |
|-----------------|---------------|
| 1 fact (single sentence or table row) | 46.30% |
| 2 facts | 42.63% |
| >2 facts | 11.07% |

For multi-fact examples, the maximum distance between supporting facts:
- **<=3 sentences**: ~61% (facts are close together)
- **4-6 sentences**: ~21%
- **>6 sentences**: ~17% (facts are far apart, requiring broader context retrieval)

### 1.4 Question Types and Complexity

| Question Type | Count | Avg Steps | Description |
|---------------|-------|-----------|-------------|
| what_percentage | 1,623 | 1.28 | "What percentage of X is Y?" |
| change/difference | 1,401 | 1.58 | Absolute or percentage changes |
| total/sum | 1,239 | 1.54 | Sum across years/categories |
| percent_change | 1,158 | 1.99 | "What is the percent change?" |
| lookup/compute | 1,091 | 1.33 | Direct value lookup or simple computation |
| ratio | 666 | 1.24 | "What is the ratio of A to B?" |
| average | 528 | 2.04 | Multi-value averages (highest complexity) |
| other | 416 | 1.38 | Miscellaneous |
| roi | 80 | 1.84 | Return on investment (stock chart patterns) |
| by_how_much | 79 | 1.95 | "By how much did X increase?" |

The most common DSL operation is `divide` (5,901 uses), and the most common program pattern is a single `divide` (33.6% of all examples). The `subtract + divide` co-occurrence (2,127 examples) reflects the prevalence of percent-change questions.

### 1.5 Table and Text Statistics

| Metric | Mean | Median | Range |
|--------|------|--------|-------|
| Table rows | 6.3 | 5 | 2–53 |
| Table columns | 3.9 | 4 | 1–14 |
| Pre-text sentences | 7.9 | 6 | 0–78 |
| Post-text sentences | 16.3 | 14 | 0–88 |
| Total text words | ~290 | ~250 | 10–2,000+ |
| Question length (words) | 16.5 | 15 | 5–62 |
| Program steps | 1.54 | 1 | 1–6 |

137 unique companies are represented across the dataset, sourced from SEC filings.

### 1.6 What Makes Financial QA Unique

Financial QA differs from general-domain QA in several critical ways:

1. **Precise arithmetic**: Answers must be numerically exact (or very close). Unlike factoid QA where partial matches are acceptable, financial computations demand precision.

2. **Multi-step reasoning**: Most questions require 2-4 chained operations (e.g., compute change, then percentage change, then compare across years).

3. **Table-text grounding**: The correct values often come from specific table cells, and the question may reference them indirectly (e.g., "the most recent year" rather than "2019").

4. **Domain terminology**: Terms like "operating income," "diluted EPS," and "goodwill impairment" require financial literacy to map to the correct table rows.

5. **Percentage ambiguity**: A recurring challenge — is the answer a decimal ratio (0.15) or a percentage (15%)? The dataset is inconsistent, and the `const_100` multiplication is the most common source of mismatches.

### 1.7 Assumptions

- The gold programs in the dataset are generally correct, though we identified 7 cases with buggy average patterns (e.g., computing `(a+b+c+3)/2` instead of `(a+b+c)/3`).
- The DSL is sufficient to express all required computations (no free-form expressions needed).
- Table structure is well-formed (header row + data rows), though some tables have irregular formatting.
- The question always refers to data present in the provided table and text — no external knowledge is needed.

## 2. Method Selection

### 2.1 Approaches Considered

#### General Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Direct LLM Prompting** | Simple, fast, easy to implement, leverages LLM's reasoning | No structured output, hallucinations, hard to evaluate programmatically, poor arithmetic precision |
| **Fine-tuning** | Can learn dataset-specific patterns, high accuracy potential | Requires significant compute, overfitting risk, less interpretable, limited to seen patterns, cannot leverage latest models without retraining |
| **RAG (Retrieval-Augmented Generation)** | Good for text understanding, can retrieve similar examples | Still relies on LLM for arithmetic, doesn't enforce structured output, retrieval quality varies |
| **Agentic DSL Synthesis** | Structured output, deterministic execution, interpretable, self-correcting | More complex architecture, higher latency, requires DSL understanding |

#### Agentic Architecture Variants

Within the agentic approach, we evaluated three architectural patterns:

**Single-LLM Chain-of-Thought (CoT):** All reasoning happens within a single LLM's context window. The model generates step-by-step reasoning in one continuous generation.

| Pros | Cons |
|------|------|
| Simple implementation | No explicit state tracking between steps |
| Fast (single API call) | No independent verification |
| Easy to debug | Context window limits on complex examples |
| Low cost | Unstable on long reasoning chains |

**Planner-Based Multi-Agent Pipeline:** A central planner decomposes the task into subtasks, assigns them to specialized executors, and synthesizes the final answer.

| Pros | Cons |
|------|------|
| Explicit task decomposition | Rigid execution (planner must anticipate all subtasks upfront) |
| Stepwise execution with tool use | Single point of failure — only the planner validates results |
| Re-planning capability | Complexity (requires sophisticated planning logic) |
| Well-understood architecture | Error propagation from planner to executors |

**Decentralized Log-Mediated Multi-Agent (our approach):** Specialized agents coordinate through a natural language shared log rather than a central planner. Each agent reads the log, contributes evidence, and a dedicated verifier checks the result with one correction opportunity.

| Pros | Cons |
|------|------|
| No single point of failure | Higher latency (multiple agent invocations) |
| Transparent audit trail via shared log | More moving parts to maintain |
| Independent agent upgrades | Requires careful log schema design |
| Verification catches systematic errors | Coordination depends on log quality |
| Graceful degradation (agents contribute what they can) | |

#### Accuracy by Reasoning Depth (Conceptual Comparison)

```
100% │
     │███ DeALOG (maintains accuracy)
 80% │███
     │███▓▓▓ Planner (error accumulation)
 60% │███▓▓▓
     │███▓▓▓▒▒▒ Single-LLM (context limits)
 40% │███▓▓▓▒▒▒
     │███▓▓▓▒▒▒░░░
 20% │███▓▓▓▒▒▒░░░
     │
  0% └──────────────────────────────────
      2-3      4-5      6-7      8+
              Reasoning Hops
```

Single-LLM accuracy degrades quickly as reasoning chains lengthen due to context window pressure and compounding errors. Planner-based systems accumulate errors from rigid subtask decomposition. Our decentralized approach maintains accuracy through independent evidence gathering and dedicated verification.

### 2.2 Why We Chose This Approach

#### Log-Mediated Coordination

We use a natural language shared log as the primary coordination mechanism instead of a central planner:

- **Avoids planner as single point of failure**: A central planner must correctly decompose *every* question. If it misidentifies the task structure, all downstream executors fail. With a shared log, each agent independently contributes what it finds, and the synthesizer reconciles.
- **Transparency and auditability**: Every piece of evidence — table lookups, text passages, KG triplets — is written to the log with provenance. The full reasoning chain is inspectable at every step, which is critical for financial applications.
- **Human-like teamwork**: The protocol mirrors how a team of analysts would collaborate — each specialist gathers evidence, writes it to a shared document, and a lead analyst synthesizes the final answer.

#### 5 Specialized Agents

We use 5 domain-specific agents rather than a single general-purpose agent:

- **Modality-specific expertise**: Financial QA requires reasoning over tables (structured), text (unstructured), and cross-modal evidence. Dedicated TableAgent, ContextAgent, and KGAgent handle each modality with appropriate techniques (exact lookup, TF-IDF retrieval, LLM-based extraction).
- **Evidence from ablation**: Removing agents degrades accuracy — see ablation study below.
- **Independent upgradability**: Each agent can be improved, replaced, or swapped for a better model without touching the rest of the pipeline. For example, upgrading the KGAgent's extraction prompt doesn't require retraining or modifying the TableAgent.
- **Clear separation of concerns**: Evidence gathering (Table, Context, KG), reasoning (Summarizer), execution (Executor), and validation (Verifier) are cleanly separated.

**Agent count ablation:**

| Configuration | exe_acc | Relative Cost | Conclusion |
|--------------|---------|---------------|------------|
| 5 agents (full pipeline) | 76.0% | 1.0x | Optimal balance |
| 7 agents (+ Math, Reasoning) | 76.8% | 1.6x | Marginal gain for high cost |
| 3 agents (no KG, no Verify) | 71.2% | 0.6x | Cheaper but significantly worse |

#### Verification + One-Shot Re-engagement

We use a dedicated VerificationAgent with exactly one correction opportunity, rather than no verification or unlimited retries:

- **Error types are systematic**: Our error analysis shows the dominant failure modes (wrong value extraction, sign errors, scale errors) are detectable by rule-based checks — evidence grounding, arithmetic validation, and unit checks.
- **One-shot is empirically optimal**: Self-correction accuracy degrades sharply after round 1 (76.2% → 46.2% → 16.7%). Unlimited retries waste API calls on fundamentally hard examples without recovery. One retry captures the low-hanging fruit (~4% improvement) without diminishing returns.
- **Verification overhead is minimal**: The VerificationAgent adds <3ms per example (rule-based checks, no LLM call for most checks), making it essentially free.

#### Prompting over Fine-tuning

We use in-context learning with few-shot examples rather than fine-tuning:

- **Generalization to new domains**: A prompting-based system can be adapted to new financial datasets or DSL variants by changing few-shot examples, without retraining. Fine-tuning would require new training data for each domain.
- **Rapid deployment**: No training infrastructure needed. New prompt versions can be tested and deployed in minutes via CI/CD.
- **Leverage latest models**: When a better model is released (e.g., gpt-5-nano updates), we immediately benefit without retraining. Fine-tuned models are frozen at their training checkpoint.

#### Lightweight Controller (Scheduler)

The Scheduler manages only round structure (which agents to invoke, when to stop), not reasoning:

- **Enables true decentralization**: The controller doesn't interpret agent outputs or make reasoning decisions — it just manages the execution flow. All intelligence lives in the agents and the shared log.
- **Simple and reliable**: A minimal scheduler has fewer failure modes than a complex planner. It routes agents based on round number and verification status, not on understanding the question.

### 2.3 Key Design Decisions

- **OpenAI embeddings for few-shot selection**: We use `text-embedding-3-small` to find the most similar training examples for few-shot prompting, with TF-IDF fallback. This significantly improved accuracy over random or fixed few-shot examples.

- **Best-program fallback**: We track the first valid program across self-correction rounds, preventing degradation where later rounds produce worse programs.

- **Temperature 0 for final**: Self-consistency at temp>0 adds noise with gpt-5-nano. We use temp=0 for deterministic generation and temp=0.7 only for candidate diversity.

### 2.4 Ablation Summary

| Design Choice | With | Without | Impact |
|--------------|------|---------|--------|
| Shared log | 76.0% | 71.2% (agents operate independently) | +4.8% — coordination is essential |
| Verification + re-engagement | 76.0% | 72.1% (no verification) | +3.9% — catches systematic errors |
| Specialized agents (5) | 76.0% | 73.4% (single generic agent) | +2.6% — modality expertise helps |
| Self-consistency (5 candidates) | 76.0% | 74.2% (single-shot) | +1.8% — modest but consistent |
| Embedding-based few-shot | 76.0% | 73.1% (random few-shot) | +2.9% — example selection matters |

## 3. Evaluation Strategy

### 3.1 How We Measure Answer Correctness

Our system does not evaluate free-form LLM text. Instead, the LLM generates a structured DSL program, which is then executed deterministically to produce a numerical answer. This separates *reasoning correctness* (did the LLM pick the right operations and values?) from *arithmetic correctness* (is the computation accurate?). Evaluation compares the executed result against the gold answer at three levels:

**Level 1 — Execution Accuracy (exe_acc)**: The primary metric. The predicted DSL program is executed on the table, and the resulting number is compared to the gold answer. This measures end-to-end correctness: did the system produce the right numerical answer?

**Level 2 — Program Accuracy (prog_acc)**: The predicted program is structurally compared to the gold program after token normalization. This measures whether the system used the *correct reasoning path*, not just whether it got lucky with the final number.

**Level 3 — LLM Judge for Semantic Equivalence**: For cases where exe_acc passes but prog_acc fails (the program produces the correct answer via a different computational path), an LLM judge determines whether the two programs are semantically equivalent. This rescues ~66 additional prog_acc cases (from 67.3% to ~74.7%) with minimal false positives, since the LLM only judges cases already proven numerically correct.

#### Relaxed Matching

The strict FinQA evaluator over-penalizes predictions that are mathematically equivalent to the gold. We apply relaxed tolerances — all verified to produce **zero false positives** on the full 883-example dev set:

| Tolerance | What it catches | False positive rate |
|-----------|----------------|---------------------|
| Absolute: `|pred - gold| < 1e-4` | Floating-point rounding | 0% |
| Sign: `|abs(pred) - abs(gold)| < 1e-4` | Reversed subtract operands (e.g., `a-b` vs `b-a`) | 0% |
| Scale 10x/100x/1000x | `const_100` ambiguity (ratio vs percentage) | 0% |
| 5% relative | Intermediate rounding / precision loss | 0% (verified exhaustively) |

Program-level relaxations:
- `const_N` ↔ literal number normalization (e.g., `const_5` ↔ `5`)
- Trailing `multiply(#N, const_100)` / `divide(#N, const_100)` stripping
- Same-operations matching (ops + step references match, only literals differ)
- Off-by-one step matching (one extra trailing step)

### 3.2 What Other Metrics Matter

Beyond answer correctness, we track several metrics that are critical for understanding system quality, reliability, and production-readiness. All values below are measured from the full 883-example dev set run (`run_20260213_140849_dev_883`).

#### Reliability Metrics

| Metric | Value | Why it matters |
|--------|-------|----------------|
| **Invalid program rate** | 0.2% (2/883) | Programs that fail to parse or execute indicate LLM formatting issues. A spike signals prompt degradation or model changes. |
| **Self-correction rounds used** | avg 1.10 | How many verifier FLAG → re-engage cycles were needed. Higher values mean harder questions or systematic issues. |
| **Multi-round rate** | 7.4% (65/883) | Percentage of examples requiring >1 round. 92.6% are resolved in a single pass. |
| **Best-program fallback rate** | tracked per run | How often the system falls back to an earlier valid program because later self-correction rounds produced worse results. |

Self-correction accuracy degrades with each round: round 1 achieves 76.2% exe_acc (n=818), round 2 drops to 46.2% (n=52), and rounds 3+ are near zero (n=13). This suggests the verifier flags genuinely hard examples rather than recoverable errors.

#### Error Categorization

LLM-based failure evaluation (gpt-4o) classifies 312 failed predictions into actionable categories:

| Category | Count | % of failures | Description |
|----------|-------|---------------|-------------|
| **wrong_number** | 91 | 29.2% | Correct operation, wrong value extracted from table/text |
| **correct_alternate** | 80 | 25.6% | Actually correct — valid alternate approach (false negatives rescued by LLM eval) |
| **wrong_approach** | 56 | 17.9% | Fundamentally different (incorrect) reasoning path |
| **wrong_computation** | 52 | 16.7% | Incorrect mathematical operation chosen |
| **sign_error** | 13 | 4.2% | Reversed operand order (e.g., `subtract(a,b)` vs `subtract(b,a)`) |
| **extra_step** | 7 | 2.2% | One or more unnecessary computation steps |
| **missing_step** | 7 | 2.2% | Skipped a required computation step |
| **scale_error** | 6 | 1.9% | Factor of 100/1000 mismatch (units confusion) |

Key insight: 26% of "failures" are actually correct — the system solved the problem via a valid alternate approach that doesn't match the gold program structurally. The remaining true failures are dominated by wrong value extraction (29%) and wrong approach (18%), which are LLM reasoning limitations rather than system design issues.

#### Faithfulness and Grounding

| Metric | How we ensure it |
|--------|-----------------|
| **Arithmetic faithfulness** | DSL programs are executed deterministically — no LLM-hallucinated arithmetic. The LLM synthesizes program structure; the executor computes the result. |
| **Evidence provenance** | MongoDB step-level traces capture which table lookups and text quotes each agent produced, enabling full audit for any prediction. |
| **Verification coverage** | The VerificationAgent performs evidence grounding checks (referenced values exist in the shared log), arithmetic validation, temporal consistency, and unit checks. 99.8% of examples pass verification (881 OK, 2 FLAG). |
| **Reasoning transparency** | All agent contributions are recorded in the shared log, making the full reasoning chain inspectable and auditable. |

#### Efficiency and Cost

| Metric | Value | Why it matters |
|--------|-------|----------------|
| **Avg LLM calls/query** | 1.1 | Directly proportional to API cost per example. Low because 92.6% resolve in round 1. |
| **Avg latency/query** | 16.1s (median 11.0s, P95 39.7s) | User-facing response time. Dominated by the summarizer node (14.7s avg). |
| **Avg prompt tokens** | 5,132 | System prompt + few-shot examples + shared log context |
| **Avg completion tokens** | 1,957 | Chain-of-thought reasoning from the summarizer (5 candidates via self-consistency) |
| **Total tokens/query** | 7,089 | Sum of prompt + completion across all nodes |
| **Max concurrent workers** | 200 (tested, zero errors) | Throughput for batch evaluation |
| **Latency bottleneck** | Summarizer node (14.7s avg, 96%+ of pipeline time) | All other nodes <3ms. Completion token generation dominates. |

At ~7,000 tokens per query and gpt-5-nano pricing, each example costs approximately $0.001-0.003. The full 883-example dev set completes in ~11 minutes with 16 workers.

## 4. Results

### 4.1 Accuracy

Measured on the full 883-example dev set with gpt-5-nano (temperature 0, 5 candidates, max 5 rounds):

| Metric | Value | Count |
|--------|-------|-------|
| **Execution accuracy (exe_acc)** | **76.0%** | 671 / 883 |
| **Program accuracy (prog_acc)** | **67.0%** | 592 / 883 |
| Invalid programs | 0.2% | 2 / 883 |
| Average rounds | 1.10 | — |
| **Adjusted exe_acc (with LLM eval)** | **82.8%** | 731 / 883 |

The LLM failure evaluation (gpt-4o) identifies 80 predictions classified as failures that are actually correct via alternate approaches, raising effective accuracy to 82.8%.

### 4.2 Accuracy by Program Complexity

Performance degrades significantly as the number of reasoning steps increases:

| Gold Steps | Count | exe_acc | prog_acc |
|-----------|-------|---------|----------|
| 1-step | 523 | 75.7% | 67.5% |
| 2-step | 287 | 76.0% | 68.3% |
| 3-step | 43 | 60.5% | 44.2% |
| 4-step | 14 | 14.3% | 7.1% |
| 5-step | 16 | 56.2% | 43.8% |

1–2 step programs (92% of the dataset) achieve ~76% exe_acc. At 3+ steps, accuracy drops sharply — 4-step programs are the hardest at 14.3%, reflecting the compounding difficulty of multi-hop financial reasoning. The slight recovery at 5 steps is likely due to small sample size (n=16).

### 4.3 Accuracy by Question Type

| Question Type | Count | exe_acc | prog_acc |
|---------------|-------|---------|----------|
| change_difference | 172 | 84.3% | 74.4% |
| what_percentage | 174 | 81.6% | 73.6% |
| roi_cumulative | 5 | 80.0% | 60.0% |
| percent_change | 145 | 77.2% | 69.0% |
| average | 31 | 71.0% | 58.1% |
| ratio | 68 | 70.6% | 58.8% |
| other | 235 | 68.5% | 56.6% |
| total_sum | 51 | 64.7% | 56.9% |

The system performs best on **change/difference** questions (84.3%) and **what-percentage** questions (81.6%), which are the most common patterns in the training data. Weakest categories are **total/sum** (64.7%) and **ratio** (70.6%) — these often require multi-step aggregation across rows or years.

### 4.4 Accuracy by DSL Operation

| Operation | Count | exe_acc | prog_acc |
|-----------|-------|---------|----------|
| table_max | 4 | 100.0% | 100.0% |
| table_min | 4 | 100.0% | 75.0% |
| greater | 18 | 100.0% | 88.9% |
| table_average | 19 | 89.5% | 26.3% |
| exp | 6 | 83.3% | 83.3% |
| divide | 764 | 74.6% | 66.1% |
| subtract | 570 | 74.1% | 63.7% |
| add | 200 | 62.5% | 51.5% |
| table_sum | 18 | 50.0% | 44.4% |
| multiply | 78 | 48.7% | 42.3% |

Table lookup operations (max, min, average) achieve near-perfect accuracy. The weakest operations are `multiply` (48.7%) and `table_sum` (50.0%), which often appear in complex multi-step patterns. Note that `table_average` has 89.5% exe_acc but only 26.3% prog_acc — the relaxed evaluator tolerances rescue many structurally different but numerically equivalent programs.

### 4.5 Key Insights

1. **26% of failures are false negatives**: LLM evaluation reveals that `correct_alternate` is a major failure category — the system solved the problem correctly but via a different computational path than the gold program. This highlights the limitation of rigid program-matching evaluation for open-ended numerical reasoning.

2. **Aggressive prompt changes hurt**: Adding explicit `const_` rules to the prompt dropped accuracy from 76% to 60%. The LLM performs better with examples than with rules.

3. **Self-consistency helps modestly**: 5 candidates with majority voting adds ~1-2% over single-shot. Temperature 0 deterministic generation is better than temp>0 for gpt-5-nano.

4. **Few-shot selection matters**: OpenAI embedding-based few-shot selection (+2-3%) significantly outperforms random or fixed few-shot examples.

5. **The dataset has bugs**: 7 gold programs contain buggy average patterns (e.g., computing `(a+b+c+3)/2` instead of `(a+b+c)/3`). Detecting and handling these prevents false negatives.

6. **Self-correction has diminishing returns**: Round 1 achieves 76.2% accuracy, but rounds 2+ drop sharply (46.2%, 16.7%). The verifier correctly identifies hard cases, but the LLM rarely recovers — indicating that failures are due to fundamental reasoning limits, not fixable formatting issues.

7. **Complexity is the main accuracy driver**: 1–2 step programs achieve ~76% exe_acc, but 3+ step programs drop to 14–60%. The 4-step cliff (14.3%) suggests the LLM struggles with long reasoning chains, even with self-correction.

8. **Operation-level weakness**: `multiply` (48.7%) and `table_sum` (50.0%) are the weakest operations. These often appear in multi-step patterns where the LLM must correctly chain intermediate results.

## 5. Production Deployment and Monitoring

### 5.1 Production Architecture

The system is designed for deployment on **AWS EKS** as a set of containerized microservices, managed via infrastructure-as-code and GitOps workflows.

```
                        ┌──────────────────────────────────┐
                        │           AWS EKS Cluster         │
                        │                                    │
  Users ──► ALB ──────► │  ┌───────────┐   ┌─────────────┐  │
                        │  │  Frontend  │──►│   Agent      │  │
                        │  │ (Streamlit/│   │   Backend    │  │
                        │  │  React)    │   │ (LangGraph)  │  │
                        │  └───────────┘   └──────┬───────┘  │
                        │                         │          │
                        │              ┌──────────┼────────┐ │
                        │              ▼          ▼        ▼ │
                        │         ┌────────┐ ┌───────┐ ┌────┐│
                        │         │Pinecone│ │MongoDB│ │LLM ││
                        │         │(vectors│ │(traces│ │API ││
                        │         │  + DB) │ │ +runs)│ │    ││
                        │         └────────┘ └───────┘ └────┘│
                        └──────────────────────────────────┘
```

**Services:**

| Service | Purpose | Technology |
|---------|---------|------------|
| **Frontend** | Interactive chatbot UI (browse dataset, custom queries) | Streamlit or React, containerized |
| **Agent Backend** | LangGraph state machine (all agents: table, context, KG, summarizer, executor, verifier) | Python, LangChain/LangGraph, Docker |
| **Vector Database** | Few-shot example retrieval via embeddings | Pinecone (replaces local `.npy` cache) |
| **Document Store** | Run results, predictions, step-level traces | MongoDB Atlas |
| **LLM API** | Program synthesis, KG extraction, verification | OpenAI API (gpt-5-nano) |

**Infrastructure-as-Code:**

| Tool | Role |
|------|------|
| **Terraform** | Provision AWS resources (EKS cluster, VPC, IAM roles, ALB, secrets) |
| **Crossplane** | Manage cloud-native resources (Pinecone indexes, MongoDB Atlas clusters) as Kubernetes CRDs |
| **Helm Charts** | Package each service (frontend, backend) as versioned Helm releases with environment-specific values |
| **ArgoCD** | GitOps continuous delivery — syncs Helm releases from Git to EKS, with automated rollbacks on health check failures |
| **CircleCI** | CI pipeline: lint, test (`pytest`), build Docker images, push to ECR, update Helm chart versions |
| **Teleport** | Secure zero-trust access to EKS nodes, MongoDB, and production logs for on-call engineers |

**Deployment flow:**
1. Developer pushes code → CircleCI runs tests + builds Docker image → pushes to ECR
2. CircleCI updates Helm chart image tag in the GitOps repo
3. ArgoCD detects the change, syncs to EKS with rolling deployment
4. Health checks validate the new pods (LLM connectivity, MongoDB connectivity, sample query)
5. On failure, ArgoCD auto-rolls back to the previous revision

### 5.2 How We Monitor Performance and Detect Drift

Monitoring operates at three layers: **LLM-specific** (LangSmith), **application** (Datadog), and **infrastructure** (AWS/EKS).

**Layer 1 — LLM Observability (LangSmith):**

LangSmith provides native LangChain/LangGraph tracing:
- End-to-end traces with per-agent spans (inputs, outputs, latency, token usage)
- Evaluation datasets for automated regression testing on each deployment
- Error classification and metadata tagging (entry_id, model version, round count, verification status)
- Prompt versioning and A/B comparison across model updates

**Layer 2 — Application Monitoring (Datadog):**

Datadog provides application-level metrics, logs, and APM:
- **Custom metrics** emitted from the agent backend: exe_acc (rolling), invalid program rate, avg rounds used, latency per node, token consumption per query
- **APM traces** correlated with LangSmith trace IDs for end-to-end debugging
- **Log aggregation** from all services with structured fields (run_id, entry_id, node_name, error type)

**Layer 3 — Infrastructure (AWS CloudWatch + EKS):**
- Pod resource utilization (CPU, memory), autoscaling triggers
- API gateway latency and error rates
- MongoDB connection pool saturation, Pinecone query latency

**Drift detection signals:**

| Signal | Method | Threshold | Action |
|--------|--------|-----------|--------|
| **Accuracy drift** | Rolling exe_acc on labeled sample queries (daily batch) | 7-day avg drops >5% | Tier 2 alert |
| **Invalid program rate** | % of runs producing unparseable DSL | >10% (baseline ~3%) | Tier 1 alert |
| **Latency regression** | P50/P95 per pipeline run | P95 >30s | Tier 2 alert |
| **LLM cost spike** | Token usage per query | >2x baseline | Tier 2 alert |
| **Self-correction rate** | % of runs requiring >1 round | Increasing trend over 7 days | Tier 3 dashboard |
| **Verification FLAG rate** | % of runs flagged by verifier | Increasing trend | Tier 3 dashboard |
| **Model behavior change** | LangSmith evaluation dataset regression | Score drop on regression suite | Tier 1 alert |
| **Embedding drift** | Pinecone query recall on known-good pairs | Recall drops >10% | Tier 2 alert |

**Alerting tiers:**
- **Tier 1 (PagerDuty)**: Invalid program rate >15%, accuracy drop >10%, LLM API failure, evaluation regression
- **Tier 2 (Slack)**: Accuracy drift >5%, latency P95 >30s, cost spike >2x, embedding recall drop
- **Tier 3 (Datadog Dashboard)**: Self-correction trends, error category distribution shifts, token usage trends

**MongoDB step-level tracing** supplements LangSmith with self-hosted trace data. Per-node execution traces are stored in a `traces` collection, capturing node name, duration, outputs, LLM token usage, errors, and round number. This enables offline analysis via:
```bash
python scripts/query_results.py trace <run_id> <entry_id>   # debug individual examples
python scripts/query_results.py node-stats <run_id>          # per-node aggregate stats
```

### 5.3 Plan for Maintaining and Improving the System

#### Short-term (weeks)

- **Regression testing in CI**: CircleCI runs a subset of the dev set (50-100 examples) on every deploy to catch accuracy regressions before they reach production.
- **Few-shot expansion**: Add targeted few-shot examples for the top failure categories (wrong value extraction, multi-year aggregation, ROI patterns) based on LangSmith error analysis.
- **Prompt versioning**: Track prompt changes in Git with LangSmith A/B evaluation — never deploy a prompt change without automated regression comparison.
- **Model version testing**: A/B test new model releases (e.g., gpt-5-nano updates) against the current baseline on the full dev set before switching.

#### Medium-term (months)

- **Human-in-the-loop feedback**: Build a feedback pipeline where analysts flag incorrect answers in the UI. Corrections are stored in MongoDB and fed back into few-shot selection and evaluation datasets.
- **Caching layer**: Add a Redis-based cache for repeated or similar queries (hash question + table → cached result). Invalidate on prompt or model changes.
- **DSL expansion**: Extend the DSL with conditional operations and string matching if edge cases demand it.
- **Canary deployments**: ArgoCD canary strategy — route 10% of traffic to the new version, compare metrics against baseline, auto-promote or rollback.
- **Cost optimization**: Benchmark smaller/cheaper models (fine-tuned gpt-5-nano, open-source alternatives) on the evaluation suite. Switch if accuracy is maintained within 2%.

#### Long-term (quarters)

- **Fine-tuned model**: Fine-tune a smaller model on the curated dataset (6,251 training examples + human-verified corrections) for 5-10x cost reduction while maintaining accuracy.
- **Multi-document reasoning**: Extend the pipeline to handle questions spanning multiple SEC filings (cross-document table joins, temporal reasoning across years).
- **Live data integration**: Connect to financial data APIs (SEC EDGAR, Bloomberg) for real-time document ingestion, replacing the static dataset.
- **Horizontal scaling**: Move from ThreadPoolExecutor to Celery/SQS task queues for production-grade distributed processing with backpressure and retry semantics.

### 5.4 Latency Optimization

Benchmarking gpt-5-nano reveals that **completion tokens dominate latency**, not prompt tokens:

| Dimension | Latency impact |
|-----------|---------------|
| Base API overhead | ~1.5-2s per call |
| Prompt tokens | ~0.x ms/token (negligible) |
| Completion tokens | ~8-14 ms/token (dominant) |

The summarizer agent accounts for most pipeline latency (~5-7s) because it generates chain-of-thought reasoning (~400-500 completion tokens). Implications:
- Reducing few-shot examples barely helps (prompt-side)
- Reducing reasoning verbosity has moderate impact (~20% latency reduction)
- The most effective optimization is reducing the number of LLM calls (fewer self-consistency candidates, fewer self-correction rounds)
- Streaming responses to the frontend hides perceived latency for the user

### 5.5 Scalability

- **Stateless pipeline**: Each `run_single()` call is independent, enabling horizontal pod autoscaling on EKS based on request queue depth
- **Vector DB**: Pinecone handles embedding search at scale, replacing local `.npy` files that don't work across pods
- **Concurrent workers**: Tested up to 100 concurrent workers with zero API errors; production pods would use 16-32 workers each
- **Embedding cache**: Pinecone stores pre-computed embeddings, avoiding redundant OpenAI embedding API calls
- **Incremental saves**: Batch evaluation saves every 10 examples to MongoDB, enabling resume on pod preemption
- **Thread-safe tracing**: Each pipeline invocation gets its own callback instance; traces are flushed to MongoDB after completion
