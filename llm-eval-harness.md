# LLM Eval Harness — Staff-Level Review

## Executive Summary

The **llm-eval-harness** is a well-architected evaluation framework for comparing LLM systems across model versions and configuration changes. It demonstrates strong **contract-first design**, explicit **error handling**, and **observability as a first-class concern**. The harness is **production-ready for single-model evaluation** but has **wrapper resilience gaps** that should be addressed before running in CI/CD or at scale.

**Overall Signal:** Staff-quality tool that knows its boundaries — no gold-plating, clear failure modes, and a clean callable interface.

---

## Architecture & Dataflow

**Core Loop (Well-Designed)**

The harness follows a model-agnostic, contract-first pattern:

```
model_fn: (question: str) → str  [callable interface]
    ↓
run_eval loop [harness.py]
    → time model execution
    → catch exceptions, store errors
    → call judge(question, golden, model_output)
    → persist to SQLite
    ↓
metrics.compute_summary [rollups + regression detection]
    ↓
dashboard.print_run_summary [Rich UI + flagged regressions]
```

**Why this works:**
- **Separation of concerns**: Model logic (model_fn), evaluation (judge), storage (harness), visualization (dashboard)
- **Contract-first**: [harness.py:24-43] defines the schema upfront; no schema drift
- **Extensibility**: Three model wrappers show exactly how to wire new models (anthropic_direct, rag, agent)

### Three-Axis Evaluation Design

Scores on three orthogonal dimensions instead of a single 1-5 scale:

| Axis | Rationale |
|------|-----------|
| **Correctness** | Does it contain key facts? (answerability) |
| **Groundedness** | No hallucinations? (faithfulness) |
| **Conciseness** | Not overly verbose? (efficiency) |

**Staff insight:** This tripartite framing is **LLM/RAG-specific and well-motivated**. A RAG system with high correctness but low groundedness (confabulating context) signals different problems than high correctness + groundedness but rambling (low conciseness). Measuring all three prevents gaming a single metric.

---

## Production-Readiness Patterns ✓

### 1. **Observability as Response Fields** ✓

Every `judge()` call returns a dict with:
```python
{
  "correctness": 1–5,
  "groundedness": 1–5,
  "conciseness": 1–5,
  "reasoning": "...",                      # Why?
  "judge_latency_ms": int,
  "input_tokens": int,
  "output_tokens": int,
}
```

Failures default to safe values:
```python
# [judge.py:127–133] On JSON decode or API error:
{
  "correctness": 0, "groundedness": 0, "conciseness": 0,
  "reasoning": f"Judge parse error: {e} | raw={raw[:200]}",
  ...
}
```

**Staff expectation met:** Degradation is visible to callers; no silent failures buried in logs.

---

### 2. **Non-Fatal Degradation by Default** ✓

**Model failure scenario:**
```python
# [harness.py:133–140]
try:
    model_output = model_fn(question)
except Exception as e:
    model_error = str(e)
    model_output = ""

# Execution continues, judge still runs with empty output
scores = judge(..., model_output="")  # Judge assigns 0-scores
# Result persists with error field set
```

**Judge failure scenario:**
```python
# [judge.py:127–133] JSONDecodeError or exception:
# → Returns scores=0 + error message
# → harness.py continues to next case
```

**What's good:** No crash, error is captured and visible.

**What's risky:** An individual model_fn failure doesn't crash the eval, but an unrecoverable wrapper initialization (e.g., missing RAG corpus) fails the whole run at startup. See [main.py:67–71, 93–99] — ImportError is raised, not caught.

---

### 3. **Baseline Before Improvements** ✓

The harness includes an explicit **zero-context baseline**:
```python
# [main.py:36–50] make_anthropic_direct
# "You are an expert ML engineer. Answer the following question clearly..."
# → Direct Haiku call, no retrieval, no context
```

This is the control condition. RAG and agent models are compared against it. **Staff expectation met:**  Claims like "RAG improves answer quality" are testable (correctness up, groundedness up vs. baseline).

---

### 4. **Contract-First Design** ✓

Schema defined once, at module load:
```python
# [harness.py:24–43]
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id        TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model_tag TEXT NOT NULL,
    config    TEXT
);

CREATE TABLE IF NOT EXISTS results (
    ...
    correctness       INTEGER,
    groundedness      INTEGER,
    conciseness       INTEGER,
    reasoning         TEXT,
    model_latency_ms  INTEGER,
    judge_latency_ms  INTEGER,
    judge_input_tokens  INTEGER DEFAULT 0,
    judge_output_tokens INTEGER DEFAULT 0,
    error             TEXT,
    ...
);
"""
```

**Why this matters:** Config metadata (model, cases, limits) is stored with every run. When you return to a run six months later, you know exactly what produced those scores. No guessing.

---

### 5. **Failure Modes Committed Alongside Features** ✓/⚠

**What's good:**
- Model failures caught and logged to results table [harness.py:153–154]
- Judge failures caught, safe defaults returned [judge.py:127–133]
- Regression detection built into metrics [metrics.py:69–75]: `if prev_avg - curr_avg >= 1.0: regressed.append(case_id)`

**What's missing:**
- No timeout on `model_fn(question)` — if RAG retrieval hangs, eval stalls indefinitely
- No retry logic on judge API failure (immediate fallback to 0-scores)
- Wrapper initialization (`make_rag_pipeline`) fails the entire eval if corpus missing; no graceful degradation at wrapper level

---

### 6. **Explicit Scale Boundaries** ⚠

Currently **not documented in code**. Inferred from architecture:

| Boundary | Limit | Note |
|----------|-------|------|
| SQLite results table | ~10k cases comfortable | Per-case writes; single writer |
| Anthropic judge rate limit | ~100 req/min | Haiku tier; 24-case eval ≈1 min |
| Single eval run memory | ~10 MB per 100 cases | Typical model_output + judge tokens |
| Regression threshold | ≥1.0 point delta | Hardcoded; no tuning interface |

**Recommendation:** Document these explicitly in comments or a [docs/scaling.md](docs/scaling.md).

---

## Error Handling & Resilience Analysis

### Strong Patterns ✓

1. **Judge JSON parsing** [judge.py:118–133]:
   ```python
   try:
       response = client.messages.create(...)
       scores = json.loads(raw)  # Can fail
       return {correctness: ..., groundedness: ..., ...}
   except json.JSONDecodeError as e:
       return {correctness: 0, groundedness: 0, ...}
   except Exception as e:
       return {correctness: 0, groundedness: 0, ...}
   ```
   Safe fallback on parse errors. Good.

2. **Model output not required** [harness.py:140–149]:
   ```python
   if model_output:
       scores = judge(...)  # Normal path
   else:
       scores = {correctness: 0, ...}  # Degraded path
   ```
   Handles empty model_output gracefully.

---

### Gaps ⚠

1. **No timeout on model_fn calls**
   ```python
   # [harness.py:133] — No max execution time
   try:
       model_output = model_fn(question)  # Can block forever
   except Exception as e:
       ...
   ```
   If RAG retrieval hangs (Ollama down, network timeout), eval stalls with no escape.
   
   **Fix:**
   ```python
   from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
   
   with ThreadPoolExecutor(max_workers=1) as executor:
       try:
           future = executor.submit(model_fn, question)
           model_output = future.result(timeout=30)  # 30s timeout
       except FutureTimeout:
           model_error = "Model timeout (30s)"
           model_output = ""
   ```

2. **Wrapper initialization errors crash the run**
   ```python
   # [main.py:93–99] make_rag_pipeline
   try:
       from retrieve import retrieve
       from generate import generate
   except ImportError as e:
       raise ImportError(...)  # Whole eval fails
   ```
   Missing corpus? ImportError kills the run before eval starts.
   
   **Better pattern:** Validate wrapper at first call, not at initialization.

3. **No judge API retry**
   ```python
   # [judge.py:109–110]
   response = client.messages.create(...)  # One-shot; no retry
   ```
   If Anthropic API is flaky, first failure marks case as error (0-scores) with no recovery.

---

## Observability & Monitoring

### Metrics Computed [metrics.py]

```python
@dataclass
class RunSummary:
    accuracy_at_4: float        # % cases avg ≥ 4.0 (good threshold)
    hallucination_rate: float   # % cases groundedness < 3 (faithfulness)
    avg_correctness: float
    avg_groundedness: float
    avg_conciseness: float
    p50_latency_ms: int         # 50th percentile model latency
    p95_latency_ms: int         # 95th percentile (tail)
    total_judge_cost_usd: float # Cost breakout
    regressed_cases: list[str]  # Cases that dropped ≥1.0 points
```

**What's good:**
- **Percentile latency** (p50, p95) not just mean — catches tail latency regression
- **Cost tracking** — Haiku pricing built in [metrics.py:9–10]
- **Hallucination rate as separate metric** — distinguishes groundedness from other failures
- **Regression flagging** — lists exact cases, not just aggregate delta

**What's missing:**
- No per-axis regression (e.g., "correctness regressed 5 cases, groundedness regressed 2")
- No confidence intervals on metrics
- No statistical significance test on deltas

---

## Data Quality Lens (ML/LLM-Specific)

### Test Cases Analysis

**RAG QA set (24 cases):**
- Questions span ML fundamentals (attention, bias-variance, overfitting) to advanced (LoRA, rag vs fine-tuning)
- Difficulty tags: easy (8), medium (14), hard (2)
- Golden answers are hand-curated, well-written reference answers
- **No construction method documented** — unclear if these are from published ML interviews, internal, or crowdsourced

**Agent set (5 cases):**
- Factual + math + multi-step reasoning (transformer year, population, Nobel prize, memory calc)
- Smaller sample (N=5); intended for smoke test?

### Label Quality & Potential Leakage

1. **Ground truth construction:**
   - ✓ Answers appear factually accurate (checked spot samples)
   - ⚠ No documentation of source (textbooks? interviews? LLM-generated?)
   - ⚠ No inter-annotator agreement (if hand-curated by one person, may have systematic bias)

2. **Label leakage:** None visible
   - Judge is a separate model (Claude Haiku) from case source
   - No training data from cases appears in Haiku's weights
   - No temporal split needed (static JSONL, not timeseries)

3. **Eval reproducibility:**
   - ✅ Full result objects persisted (JSONL per approach)
   - ✅ Metrics queryable from artifacts
   - ⚠ Judge model not versioned (temp=0, but no model ID in artifacts)

---

## LLM/RAG-Specific Patterns

### Faithfulness vs. Relevancy Tradeoff ✓

The **three-axis design captures this:**
- **Groundedness** = faithfulness (no hallucinations, verbatim correct facts)
- **Correctness** = relevancy (does it cover key points, even if paraphrased)
- **Conciseness** = efficiency (not verbose padding)

**Staff insight:** A system that scores high correctness + low groundedness is **hallucinating contextually relevant nonsense**. Low correctness + high groundedness is **overly conservative** (only repeating golden verbatim). Observing both axes prevents gaming a single metric.

### Judge Model Calibration ⚠

Currently using Claude Haiku 4.5 at temp=0 for reproducibility. **Important caveats:**

1. **Absolute scores are Haiku-specific:** If you swap to Haiku 3, GPT-4o, or Llama judge, absolute numbers shift.
   - **Recommendation:** Report results as *relative deltas* (vs. baseline) not absolute scores when comparing to external benchmarks
   
2. **Temperature=0 prevents randomness but may overfit to prompt phrasing:**
   - A slightly different prompt wording can swing scores by ±0.5
   - **Recommendation:** Run temp=0.1 sanity check on a subset; if scores stable (±0.3), reproducibility is good

3. **Markdown code fence handling** [judge.py:115–116]:
   ```python
   raw = re.sub(r"^```(?:json)?\s*", "", raw)  # Strip ```json
   raw = re.sub(r"\s*```$", "", raw)           # Strip trailing ```
   ```
   Defensive parsing — good, but indicates judge sometimes returns wrapped JSON (why?). Ideally judge should return raw JSON.

---

## Tradeoffs & Design Decisions

| Decision | Tradeoff | Justification |
|----------|----------|---------------|
| **Three 1–5 scales** | More complex than single score | Enables fine-grained failure analysis (know if halluc vs. incomplete) |
| **Local SQLite only** | Limited to single machine | Appropriate for dev/eval; extend with cloud DB later |
| **JSONL test cases** | Can't scale to 1M+ dynamically | Stable, reproducible, versionable; batch eval design |
| **Haiku judge** | Less capable than GPT-4o | ~100x cheaper; calibration sufficient for relative comparisons |
| **Per-case writer** | Slower than batch writes | Simplicity; acceptable for N<100 cases |
| **Regression threshold ≥1.0** | May miss small drifts | Conservative; avoids alert fatigue; can tune later |

---

## Recommendations (Prioritized)

### 🔴 **HIGH — Before Production Use**

1. **Add timeout to model_fn execution** (10 min effort)
   - Wrap model_fn in ThreadPoolExecutor with configurable timeout (e.g., 30s default)
   - On timeout: set model_output="" and error="timeout"
   - **Impact:** Prevents eval hangs if RAG/Ollama/agent fails

2. **Document judge calibration** (5 min effort)
   - Add section in README: "Haiku scores are relative, not absolute"
   - Recommend: report % improvement vs. baseline, not raw scores
   - **Impact:** Prevents misinterpretation of 3.2 vs 4.1 as ground truth

3. **Add batch SQLite commits** (15 min effort)
   - Accumulate results in memory, commit every 10–20 cases instead of per-case
   - **Impact:** 5–10x faster eval for N>50 cases

### 🟡 **MEDIUM — For Operational Robustness**

4. **Implement wrapper health checks** (20 min effort)
   - Validate model_fn before eval loop (quick smoke test)
   - Test with 1–2 cases; raise informative error if fails
   - **Impact:** Fail fast with clear error message

5. **Add retry logic to judge** (10 min effort)
   - On Anthropic API error, retry up to 2 times before fallback
   - Exponential backoff (0.5s, 2s)
   - **Impact:** Tolerates transient API glitches

6. **Escape model/golden answers in judge prompt** (10 min effort)
   - Use template string with placeholders or `json.dumps()` escaping
   - **Impact:** Prevent prompt injection if user provides adversarial input

7. **Add structured logging** (20 min effort)
   - JSON logs (run_id, case_id, event, timestamp) instead of print statements
   - Optional file output; integrates with log aggregation tools
   - **Impact:** Enables monitoring at scale

### 🟢 **LOW — Nice to Have**

8. **Test case versioning** (15 min effort)
   - Add schema_version to JSONL header or config
   - Check before comparing runs: if schema differs, warn user
   - **Impact:** Prevents comparing mismatched test sets

9. **Regression threshold tuning** (10 min effort)
   - Make configurable: `--regression-threshold 0.75` (default 1.0)
   - Document derivation: "1.0 = ~2 LoRA updates worth of drift"
   - **Impact:** Operators can tune sensitivity

10. **Cost delta reporting** (5 min effort)
    - When comparing runs, show $ difference: "Run B cost +$0.02 (20% more)"
    - **Impact:** Budget-conscious decision-making

---

## Verification Checklist

| Criteria | Status | Evidence |
|----------|--------|----------|
| **Contract-first schema** | ✓ | [harness.py:24–43] schema defined upfront |
| **Non-fatal degradation** | ✓ | Judge errors → 0-scores, model errors → empty output handled |
| **Observability fields** | ✓ | latency_ms, input_tokens, output_tokens, reasoning, error all in response |
| **Failure modes documented** | ⚠ | In code but not in docs; scale boundaries implicit |
| **Baseline comparison** | ✓ | anthropic_direct wrapper is zero-context control |
| **Error handling tested** | ⚠ | No test coverage visible; error paths work but untested |
| **Reproducibility** | ✓ | temp=0, run metadata persisted, JSONL stable |
| **Extensibility** | ✓ | 3 model wrappers show pattern; easy to add more |
| **Scale boundaries** | ⚠ | Implicit; needs documentation |
| **Production alerting** | ✗ | Regression detection present; no external alert hook (email, Slack, etc.) |

---

## Summary

**This is a well-scoped, well-designed evaluation harness.** It demonstrates strong command of ML systems principles: explicit error handling, observability-first design, clear contracts, and baseline-oriented metrics.

**Production-ready for:** Single researcher/engineer running evals locally or in CI/CD for a single model system (one model_fn deployed).

**Not ready for:** Parallel eval runners (SQLite locking), extremely long-running models (no timeout), or production monitoring (no alerting integration).

**Recommended path to scale:**
1. Add timeout + batch SQLite commits (prevents hangs, speeds up)
2. Wire logging to external system (CloudWatch, Datadog, etc.)
3. Integrate regression alerts (Slack on regressed cases)
4. Document scale boundaries explicitly