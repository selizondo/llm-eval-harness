# Architectural Tradeoffs

Decisions made during build, with the reasoning and scale/complexity boundaries.

---

## Three-Axis Scoring vs Single Score

**Decision:** Judge scores on three independent 1–5 axes: correctness, groundedness, conciseness.

**Why:** A single score conflates failure modes that require different interventions. High correctness + low groundedness means the model is hallucinating contextually plausible content. High correctness + low conciseness means verbose padding. Low correctness + high groundedness means the model is staying in-context but missing the point. Measuring all three prevents gaming a single metric.

**Tradeoff:** More complex judge prompt and more output to parse. JSONDecodeError risk increases with output size — mitigated by defensive regex stripping of code fences before parsing.

---

## Claude Haiku as Judge

**Decision:** `claude-haiku-4-5` at `temperature=0` as the evaluation judge.

**Why:** ~100× cheaper than Sonnet or GPT-4o for a 24-case eval. Sufficient for relative comparisons (does RAG improve over baseline?). Temperature=0 for determinism across runs.

**Tradeoff:** Absolute scores are Haiku-specific. The same eval run on GPT-4o will produce different numbers. Report results as *relative deltas vs baseline*, not as ground-truth quality scores. A slightly different prompt wording can swing Haiku scores by ±0.5.

**Scale boundary:** At 200+ test cases, Haiku at ~$1/1M tokens still only costs ~$0.05–0.10 per full eval run. Cost is not the binding constraint — latency is (~100 req/min rate limit).

---

## SQLite for Result Persistence

**Decision:** Single `eval.db` SQLite file with `runs` and `results` tables.

**Why:** Zero infrastructure. Portable, queryable with standard SQL. Complete run metadata (model_tag, config, timestamp) stored with every result so runs from months apart are comparable.

**Tradeoff:** Single-writer. If two eval runs are triggered concurrently, SQLite will serialize or error. Per-case writes (not batched) make this even more sensitive.

**Scale boundary:** At >50 concurrent eval runs, switch to PostgreSQL. At >100 cases per run, batch writes (every 10–20 rows) reduce write overhead by 5–10×.

---

## Per-Case Writer vs Batch Writes

**Decision:** Each result is written to SQLite immediately after the judge returns.

**Why:** If the eval crashes mid-run, all completed results are preserved. No need to re-run the whole eval.

**Tradeoff:** 5–10× slower than batching. At 24 cases this is imperceptible; at 500+ cases it adds meaningful overhead.

---

## JSONL Test Cases (Static, Versioned)

**Decision:** Test cases stored as `.jsonl` files committed to the repo.

**Why:** Static test sets enable comparing runs across weeks or months. Dynamic test generation (sampling from a live dataset) makes historical comparison impossible — the test set shifts under you.

**Tradeoff:** Can't scale to 1M+ dynamically generated cases. For the eval harness use case (targeted regression testing), static sets are the right call.

---

## Regression Threshold = 1.0

**Decision:** Flag a test case as regressed when average score drops by ≥ 1.0 point vs the prior run.

**Why:** A 1.0 drop on a 1–5 scale is roughly "one full quality tier" — moved from "mostly correct" to "partially correct." This threshold avoids alert fatigue from noise (±0.3 is judge variance, not a real regression).

**Tradeoff:** May miss slow drift (0.3 points per week over 3 months). Configurable via CLI flag (`--regression-threshold`) if tighter sensitivity is needed.

---

## model_fn Callable Interface

**Decision:** Models are injected as `Callable[[str], str]` — any function that takes a question and returns a string.

**Why:** The harness is model-agnostic. Adding a new model type (e.g., a fine-tuned model, an OpenAI model) requires writing a single wrapper function, not modifying the harness itself. The three existing wrappers (`anthropic_direct`, `rag_pipeline`, `agent`) show the pattern.

**Tradeoff:** No timeout on `model_fn` calls. If a wrapper hangs (e.g., Ollama down), the eval blocks indefinitely. Fix: wrap with `ThreadPoolExecutor` + configurable timeout.

---

## What Was Cut

| Cut | Reason | Upgrade trigger |
|---|---|---|
| Timeout on model_fn | Not needed for demo scale | Any production use where external services can hang |
| Judge retry logic | One-shot sufficient for demo | Transient API errors in CI/CD |
| Parallel eval workers | Single-threaded is simpler | >100 cases where latency matters |
| External alerting (Slack, PagerDuty) | Out of scope for local eval tool | CI/CD integration requiring team notifications |
