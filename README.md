# LLM Eval Harness

![Tests](https://github.com/selizondo/llm-eval-harness/actions/workflows/test.yml/badge.svg)

A model regression looks like a customer complaint, not a stack trace. You swap a model, adjust a prompt, tune chunk size in a RAG pipeline, and three weeks later users are reporting wrong answers. No exception was raised. No alert fired. The system "worked." Until it didn't.

This harness gives every change a number. Run it before. Run it after. The difference tells you whether to ship.

**Stack:** Python · Claude Haiku (judge) · SQLite · Rich terminal

## Results

Caught on the first real use: increasing RAG chunk size from 256 to 512 words dropped Accuracy@4 by 21 points and pushed hallucination rate from 4% to 17%.

| Metric | chunk=256 (baseline) | chunk=512 (regressed) |
|--------|---------------------|-----------------------|
| Accuracy@4 | **79%** | 58% |
| Hallucination rate | 4% | 17% |
| Avg groundedness | 4.21 | 3.33 |
| Judge cost per run | $0.038 | $0.041 |

5 cases flagged as regressed. The harness prevented a chunk size change from shipping undetected.

## How It Works

### Three-axis scoring, not a single number

Each answer is scored 1 to 5 on correctness, groundedness, and conciseness independently. A single composite score hides failure modes that need different fixes. High correctness with low groundedness means the model is hallucinating plausible content. High correctness with low conciseness means verbose padding. Three axes means three different intervention signals.

### Regression threshold derived from judge variance

At temperature=0, judge scoring variance is approximately plus or minus 0.3 points per run. The default regression threshold of 1.0 point gives a 3 sigma margin above noise: a case is flagged only when the drop is unambiguous, not when it sits within normal judge variation. The threshold is configurable via `--regression-threshold` if tighter sensitivity is needed.

### Model-agnostic callable interface

The harness receives a `(str) -> str` function and does not know what is behind it. RAG pipeline, fine-tuned model, direct API call, agent: one harness evaluates all of them. Adding a new model variant requires one wrapper function, not a harness change. Every comparison uses the same scoring tool, making results comparable across models, prompts, and retrieval strategies.

**Companion post:** "Don't Guess. Measure." (AI Systems in Production series, coming soon)
**Related projects:** [rag-pipeline-from-scratch](https://github.com/selizondo/rag-pipeline-from-scratch) (baseline established with this harness: 72% Accuracy@4) · [llm-drift-monitor](https://github.com/selizondo/llm-drift-monitor) (production counterpart: catches degradation between releases)

---

## Go Deeper

| Audience | Doc |
|----------|-----|
| Business and product context | [Product and Cost](docs/product.md) |
| Running the code | [Setup and Usage](docs/setup.md) |
| Engineering decisions | [Design and Tradeoffs](docs/engineering.md) |
| What breaks and why | [Failure Modes](docs/failures.md) |
