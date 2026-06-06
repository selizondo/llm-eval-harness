# Design and Tradeoffs

Decisions made during build, with reasoning and the scale or complexity boundary where each breaks down.

---

## Three-Axis Scoring vs Single Score

**Decision:** Judge scores on three independent 1 to 5 axes: correctness, groundedness, conciseness.

**Why:** A single score conflates failure modes that require different interventions. High correctness with low groundedness means the model is hallucinating contextually plausible content. High correctness with low conciseness means verbose padding. Low correctness with high groundedness means the model is staying in-context but missing the point. Measuring all three prevents gaming a single metric and surfaces the right fix for each failure type.

**Tradeoff:** More complex judge prompt and more output to parse. JSONDecodeError risk increases with output size. Mitigated by defensive regex stripping of code fences before parsing.

---

## Claude Haiku as Judge

**Decision:** `claude-haiku-4-5` at `temperature=0` as the evaluation judge.

**Why:** Approximately 100x cheaper than Sonnet or GPT-4o for a 24-case eval. Sufficient for relative comparisons (does RAG improve over baseline?). Temperature=0 for determinism across runs.

**Tradeoff:** Absolute scores are Haiku-specific. The same eval run on GPT-4o will produce different numbers. Report results as relative deltas vs baseline, not as ground-truth quality scores. A slightly different prompt wording can swing Haiku scores by approximately plus or minus 0.5.

**Scale boundary:** At 200+ test cases, Haiku at ~$1/1M tokens still only costs ~$0.05 to $0.10 per full eval run. Cost is not the binding constraint at this scale. Latency is (~100 req/min rate limit).

---

## Regression Threshold = 1.0

**Decision:** Flag a test case as regressed when average score drops by 1.0 point or more vs the prior run.

**Why:** A 1.0 drop on a 1 to 5 scale is roughly one full quality tier: moved from "mostly correct" to "partially correct." At temperature=0, judge variance is approximately plus or minus 0.3 per run. A threshold of 1.0 gives a 3 sigma margin above noise, avoiding false positives from judge variation.

**Tradeoff:** May miss slow drift (0.3 points per week over 3 months). Configurable via `--regression-threshold` if tighter sensitivity is needed.

---

## SQLite for Result Persistence

**Decision:** Single `eval.db` SQLite file with `runs` and `results` tables.

**Why:** Zero infrastructure. Portable, queryable with standard SQL. Complete run metadata (model_tag, config, timestamp) stored with every result so runs from months apart are comparable.

**Tradeoff:** Single-writer. If two eval runs are triggered concurrently (parallel CI jobs), the second run hits `OperationalError: database is locked`. Per-case writes (not batched) make this more sensitive.

**Scale boundary:** At more than 50 concurrent eval runs, switch to PostgreSQL. At more than 100 cases per run, batch writes (every 10 to 20 rows) reduce write overhead by 5 to 10x.

---

## Per-Case Writer vs Batch Writes

**Decision:** Each result is written to SQLite immediately after the judge returns.

**Why:** If the eval crashes mid-run, all completed results are preserved. No need to re-run the whole eval from the beginning.

**Tradeoff:** 5 to 10x slower than batching. At 24 cases this is imperceptible. At 500+ cases it adds meaningful overhead.

---

## JSONL Test Cases (Static, Versioned)

**Decision:** Test cases stored as `.jsonl` files committed to the repo.

**Why:** Static test sets enable comparing runs across weeks or months. Dynamic test generation (sampling from a live dataset) makes historical comparison impossible because the test set shifts under you.

**Tradeoff:** Cannot scale to 1M+ dynamically generated cases. For the eval harness use case (targeted regression testing), static sets are the right call.

---

## model_fn Callable Interface

**Decision:** Models are injected as `Callable[[str], str]`: any function that takes a question and returns a string.

**Why:** The harness is model-agnostic. Adding a new model type (a fine-tuned model, an OpenAI model, a local agent) requires writing a single wrapper function, not modifying the harness. The three existing wrappers (`anthropic_direct`, `rag_pipeline`, `agent`) show the pattern.

**Tradeoff:** No timeout on `model_fn` calls. If a wrapper hangs (Ollama down, network issue), the eval blocks indefinitely. Fix: wrap with `ThreadPoolExecutor` and a configurable timeout.

---

## What Was Cut

| Cut | Reason | Upgrade trigger |
|-----|--------|-----------------|
| Timeout on model_fn | Not needed for demo scale | Any production use where external services can hang |
| Judge retry logic | One-shot sufficient for demo | Transient API errors in CI/CD |
| Parallel eval workers | Single-threaded is simpler | More than 100 cases where latency matters |
| External alerting (Slack, PagerDuty) | Out of scope for local eval tool | CI/CD integration requiring team notifications |
| A/B significance testing | Statistical framework adds complexity | When score differences between model variants need confidence intervals |

---

## Scale Boundaries

| Component | Current implementation | Breaks at | Migration path |
|-----------|----------------------|-----------|----------------|
| SQLite storage | Single-writer file-level lock | >1 concurrent writer | Per-run database files or PostgreSQL |
| Anthropic rate limit | Sequential judge calls | Tier 1: ~50 req/min | Add sleep between cases or upgrade tier |
| Eval throughput | Sequential model + judge | ~22 to 43 cases/min | Async or thread pool for judge calls |
| model_fn timeout | Hardcoded 30s in harness.py | Any hanging model | Configurable via CLI (not yet implemented) |
| SQLite row count | No practical limit | 10M+ rows without index | Add index on run_id and case_id |

**Throughput estimates:**

| Bottleneck | Latency | Cases/minute |
|------------|---------|-------------|
| Direct Anthropic model | 800 to 1,500ms | 40 to 75 |
| RAG pipeline (Ollama) | 2,000 to 4,000ms | 15 to 30 |
| Judge call | 600 to 1,200ms | 50 to 100 |
| Combined (sequential) | 1,400 to 2,700ms/case | 22 to 43 |

A 24-case eval run completes in roughly 30 to 120 seconds depending on model speed.

---

## Architectural Standard

An eval harness that runs in CI and blocks merge on hallucination regression is the difference between "we think quality improved" and "we know quality improved."

The model-agnostic callable interface is the key design decision. That interface means every comparison (RAG pipeline vs fine-tuned model, Haiku vs Sonnet, prompt v1 vs v2) uses the same measurement tool. Consistent measurement is what makes quality comparisons meaningful across models, versions, and teams.

The dual-backend design (Anthropic for production evals, Ollama for offline runs via `JUDGE_MODEL=ollama`) removes the API key as a gate on running evals locally. Any engineer can run the full eval suite without credentials, which means eval runs in CI without secrets management complexity.
