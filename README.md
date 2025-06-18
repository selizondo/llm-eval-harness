# LLM Eval Harness

A reusable evaluation framework for LLM-powered systems. Define test cases with golden answers, grade model outputs with an LLM judge, and track scores across model versions to catch regressions before they ship.

**Stack:** Python · Anthropic Claude Haiku (judge) · SQLite · Rich terminal

---

## The Problem

When you change anything in an LLM system — swap the model, adjust the prompt, tune chunk size in RAG, update the retrieval strategy — you need a repeatable way to measure whether quality went up or down. Without it, you're shipping on intuition.

This harness gives you a number. Run it before a change. Run it after. Compare.

---

## Architecture

```
Test cases (JSONL)
       │
       ▼
┌─────────────────────────────────────────┐
│  Eval Loop                              │
│                                         │
│  for each case:                         │
│    model_answer = model_fn(question)    │
│    scores = judge(question,             │
│                   golden, answer)       │
│    store → SQLite                       │
└─────────────────────────────────────────┘
       │
       ▼
Rich terminal dashboard + regression report
```

The model under test is a plain Python callable `(question: str) -> str` — swap in any model, pipeline, or agent without changing the harness.

---

## Files

| File | Purpose |
|------|---------|
| `evals/harness.py` | Core eval loop — model-agnostic, stores results in SQLite |
| `evals/judge.py` | LLM-as-judge — scores correctness, groundedness, conciseness (1–5) |
| `evals/metrics.py` | Aggregates results — accuracy, hallucination rate, latency, cost, regressions |
| `evals/dashboard.py` | Rich terminal display — summary table + per-case breakdown |
| `evals/cases/rag_qa.jsonl` | 24 ML/AI test cases for the RAG pipeline |
| `evals/cases/agent.jsonl` | 5 test cases for the tool-use agent |
| `main.py` | CLI entrypoint |

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

---

## Usage

```bash
# Run eval against Claude Haiku directly (baseline — no retrieval)
python main.py run --cases evals/cases/rag_qa.jsonl --model anthropic_direct --tag "haiku_baseline"

# Run eval against the RAG pipeline (requires Ollama running)
python main.py run --cases evals/cases/rag_qa.jsonl --model rag --tag "rag_chunk256"

# Compare to a previous run — prints regression report
python main.py run --cases evals/cases/rag_qa.jsonl --model rag --tag "rag_chunk512" --compare <run_id>

# List all historical runs
python main.py list-runs

# Print per-case breakdown for a run
python main.py show-cases <run_id>

# Quick smoke test (first 3 cases)
python main.py run --cases evals/cases/rag_qa.jsonl --model anthropic_direct --tag "smoke" --limit 3
```

---

## Scoring Rubric

Each answer is scored 1–5 on three axes by Claude Haiku at temperature=0:

| Axis | What it measures | Failure signal |
|------|-----------------|----------------|
| **Correctness** | Key facts from the golden answer present and accurate | Missed concepts, wrong definitions |
| **Groundedness** | No hallucinations or unsupported claims | Invented citations, confident falsehoods |
| **Conciseness** | Appropriately scoped — not padded, not too terse | Rambling answers, one-line non-answers |

**Accuracy@4:** % of cases with average score ≥ 4.0 — the primary headline metric.  
**Hallucination rate:** % of cases with groundedness < 3 — the safety signal.

---

## Scenario: Catching a RAG Regression

You're tuning the RAG pipeline and want to know whether increasing chunk size from 256 to 512 words helps or hurts answer quality.

**Step 1 — Baseline with chunk size 256:**
```bash
python main.py run \
  --cases evals/cases/rag_qa.jsonl \
  --model rag \
  --tag "rag_chunk256"
```

```
Run a1b2c3d4  |  model: rag_chunk256  |  24 cases

  [01/24] ✓ rag_001    C=5 G=5 P=4  avg=4.7  model=3102ms
  [02/24] ✓ rag_002    C=5 G=4 P=4  avg=4.3  model=2891ms
  [03/24] ~ rag_003    C=4 G=3 P=4  avg=3.7  model=3250ms
  ...

  Metric                       Value
 ────────────────────────────────────
  Cases evaluated                 24
  Accuracy@4 (avg ≥ 4.0)         79%
  Hallucination rate               4%
  Avg correctness               4.42
  Avg groundedness              4.21
  Avg conciseness               4.08
  p50 latency               3,102 ms
  p95 latency               4,890 ms
  Judge cost (est.)           $0.038
```

**Step 2 — Rerun with chunk size 512, compare to baseline:**
```bash
python main.py run \
  --cases evals/cases/rag_qa.jsonl \
  --model rag \
  --tag "rag_chunk512" \
  --compare a1b2c3d4
```

```
Run e5f6g7h8  |  model: rag_chunk512  |  compared to: a1b2c3d4

  Metric                       Value
 ────────────────────────────────────
  Cases evaluated                 24
  Accuracy@4 (avg ≥ 4.0)         58%   ← dropped 21 points
  Hallucination rate              17%   ← up from 4%
  Avg correctness               3.88
  Avg groundedness              3.33   ← notable drop
  Avg conciseness               4.12
  p50 latency               3,340 ms
  p95 latency               5,100 ms
  Judge cost (est.)           $0.041

⚠  Regressions detected (5 cases):
   • rag_005  (RAG vs fine-tuning)
   • rag_006  (LoRA explanation)
   • rag_018  (LSTM vs Transformer)
   • rag_020  (transformer architecture)
   • rag_021  (feature engineering)
```

The harness caught it: larger chunks dilute the embedding signal, retrieval becomes less precise, and groundedness drops — the model starts filling gaps with hallucinations instead of retrieved context. **Don't ship chunk size 512. Stay at 256.**

This is the scenario the chunking experiment in the RAG pipeline confirmed empirically. The eval harness gives you the same signal at the answer quality level, not just the retrieval score level.

---

## Test Case Format

```json
{
  "id": "rag_001",
  "input": "What is the attention mechanism in transformers?",
  "golden_answer": "The attention mechanism computes a weighted sum of values...",
  "tags": ["transformers", "architecture"],
  "difficulty": "medium"
}
```

24 cases in `evals/cases/rag_qa.jsonl` covering: transformers, optimization, regularization, evaluation metrics, NLP, and ML fundamentals. 5 cases in `evals/cases/agent.jsonl` covering factual lookup and multi-step reasoning.

---

## Design Decisions

**LLM-as-judge instead of exact match**
Golden answers for open-ended ML questions can't be matched exactly — two correct answers may use different phrasing. LLM-as-judge captures semantic equivalence. The tradeoff: judge scores have variance (~±0.3 across runs at temperature=0). Mitigations: fixed temperature, structured JSON output, and running the judge twice on disputed cases.

**When LLM-as-judge fails**
The judge can be fooled by confident-sounding wrong answers (gives high groundedness to plausible hallucinations) and can penalize terse-but-correct answers. Always spot-check cases near the scoring threshold (avg 3.0–4.0). For high-stakes eval, supplement with human review on the tail.

**SQLite over a hosted metrics store**
Zero config, runs locally, trivially queryable. The schema supports trend analysis and regression detection without a server. For a team setting, swap to Postgres or plug into an observability platform (Langfuse, W&B).

**Model-agnostic callable interface**
The harness doesn't know what the model is — it receives a `(str) -> str` function. This makes it trivial to eval any model: direct API, RAG pipeline, agent, fine-tuned model, or a mock for testing. See `main.py` for the three built-in wrappers.

---

## What I'd Do With More Time

- **Human annotation pipeline** — flag low-confidence judge scores (avg 2.5–3.5) for human review; build a simple UI to collect labels
- **A/B testing framework** — run two model variants on the same cases simultaneously, compute statistical significance of score differences
- **CI integration** — GitHub Actions workflow that runs the harness on every PR and posts a score summary as a comment; block merge if hallucination rate exceeds threshold
- **Broader case coverage** — add adversarial cases (ambiguous questions, out-of-scope queries, prompt injection attempts) to stress-test groundedness
- **Cost tracking per model** — extend the schema to track model token usage, not just judge tokens, for true cost-per-query comparison across model variants
