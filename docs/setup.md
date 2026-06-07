# Setup and Usage

## Key Concepts

**LLM-as-judge:** Using an LLM to score other LLM outputs on semantic dimensions (correctness, groundedness, conciseness). Better than exact-match for open-ended questions where multiple correct answers exist. Trade-off: judge scoring has variance (~±0.3 points at temperature=0); requires a strong judge (Claude, GPT-4o); can be fooled by confident-sounding hallucinations.

**Regression detection:** Comparing scores across runs to catch degradation. If Accuracy@4 drops from 79% to 58% after a change, you've regressed. This harness flags case-by-case regressions and compares metrics side-by-side. The threshold is configurable — default 1.0 point on a 1–5 scale (one full grade letter) to avoid false positives from judge variance.

**Golden answers:** Reference correct answers for test cases. "Correct" doesn't mean unique — for "What is attention?" both "Self-attention is a mechanism allowing tokens to attend to all other tokens" and "Attention weights the relevance of different tokens" are correct. The judge evaluates semantic equivalence, not string matching.

**Model-agnostic interface:** The harness receives a `(str) -> str` function — it doesn't care if it's wrapping a local Ollama model, a RAG pipeline, an agent, or an API call. This means one harness can eval any model variation, making comparisons consistent across Haiku vs Sonnet, prompt v1 vs v2, retrieval-based vs fine-tuned.

---

## Prerequisites

- Python 3.10+
- `ANTHROPIC_API_KEY` in `.env` (or use Ollama as judge: see below)
- Sibling repos cloned into the same parent directory if evaluating RAG or agent models

## Quick Start

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env: uncomment ANTHROPIC_API_KEY

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Smoke test (3 cases, Claude Haiku judge)
python main.py run --cases evals/cases/rag_qa.jsonl --model anthropic_direct --tag "smoke" --limit 3

# 3. Full eval against the RAG pipeline (Ollama must be running)
python main.py run --cases evals/cases/rag_qa.jsonl --model rag --tag "rag_chunk256"

# 4. Compare two runs
python main.py run --cases evals/cases/rag_qa.jsonl --model rag --tag "rag_chunk512" --compare <run_id>
```

**No Anthropic key?** Set `JUDGE_MODEL=ollama` in `.env` to point the judge at a local Ollama model. Any engineer can run the full eval suite without credentials.

## Commands

```bash
# Eval against Claude directly (baseline, no retrieval)
python main.py run --cases evals/cases/rag_qa.jsonl --model anthropic_direct --tag "haiku_baseline"

# Eval against the RAG pipeline
python main.py run --cases evals/cases/rag_qa.jsonl --model rag --tag "rag_chunk256"

# Compare to a prior run with tighter regression sensitivity
python main.py run --cases evals/cases/rag_qa.jsonl --model rag --tag "rag_chunk512" \
  --compare <run_id> --regression-threshold 0.5

# List all historical runs
python main.py list-runs

# Print per-case breakdown for a run
python main.py show-cases <run_id>
```

**`--regression-threshold DELTA`** (default 1.0): flag a case as regressed if its average score dropped by at least DELTA points. The default of 1.0 is one full point on the 1 to 5 scale, corresponding to 3 sigma above judge variance (approximately plus or minus 0.3 per run). Lower values increase sensitivity but also increase false positives.

## Worked Example: Catching a RAG Regression

Baseline run with chunk=256:

```
Run a1b2c3d4  |  model: rag_chunk256  |  24 cases

  Accuracy@4 (avg >= 4.0)         79%
  Hallucination rate               4%
  Avg groundedness              4.21
  Judge cost (est.)           $0.038
```

Rerun with chunk=512, compared to baseline:

```
Run e5f6g7h8  |  model: rag_chunk512  |  compared to: a1b2c3d4

  Accuracy@4 (avg >= 4.0)         58%   <- dropped 21 points
  Hallucination rate              17%   <- up from 4%
  Avg groundedness              3.33    <- notable drop

  Regressions detected (5 cases):
    rag_005, rag_006, rag_018, rag_020, rag_021
```

Larger chunks dilute the embedding signal. Retrieval becomes less precise and the model fills gaps with hallucinations instead of retrieved context. Do not ship chunk=512.

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

24 cases in `evals/cases/rag_qa.jsonl`: transformers, optimization, regularization, evaluation metrics, NLP fundamentals.
5 cases in `evals/cases/agent.jsonl`: factual lookup and multi-step reasoning.

## Sibling Repos

Some model targets expect sibling repos cloned into the same parent directory:

| `--model` flag | Requires |
|----------------|---------|
| `rag` | [rag-pipeline-from-scratch](https://github.com/selizondo/rag-pipeline-from-scratch) |
| `agent` | [llm-agent-tool-use](https://github.com/selizondo/llm-agent-tool-use) |
| `anthropic_direct` | No dependency |
| `ollama` | Ollama running locally |

## Code Layout

| File | Purpose |
|------|---------|
| `evals/harness.py` | Core eval loop: model-agnostic, stores results in SQLite |
| `evals/judge.py` | LLM-as-judge: scores correctness, groundedness, conciseness (1 to 5) |
| `evals/metrics.py` | Aggregates results: accuracy, hallucination rate, latency, cost, regressions |
| `evals/dashboard.py` | Rich terminal display: summary table and per-case breakdown |
| `evals/cases/rag_qa.jsonl` | 24 ML/AI test cases for the RAG pipeline |
| `evals/cases/agent.jsonl` | 5 test cases for the tool-use agent |
| `main.py` | CLI entrypoint |
