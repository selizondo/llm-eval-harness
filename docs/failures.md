# Failure Scenarios

Failure modes for the eval harness. "Handled" means a non-fatal path exists. "Documented gap" means the failure is understood but detection is not yet implemented.

---

## Failure 1: `model_fn` Hangs Indefinitely (Documented Gap)

**What breaks:** The `model_fn` callable has no timeout. If a wrapper stalls (Ollama down, network issue, large generation), `run_eval` blocks indefinitely per case. For a 24-case eval, one hung model call stalls the entire run.

**Status:** Documented gap. Noted in `docs/tradeoffs.md` under the `model_fn` callable interface section.

**Detection (planned):** Wrap `model_fn` call with `ThreadPoolExecutor` + `Future.result(timeout=30)`:
```python
with ThreadPoolExecutor(max_workers=1) as ex:
    future = ex.submit(model_fn, case["question"])
    try:
        output = future.result(timeout=30)
    except TimeoutError:
        output = "[ERROR: model_fn timed out after 30s]"
```

---

## Failure 2: `ANTHROPIC_API_KEY` Not Set (Handled)

**What breaks:** Judge uses Claude Haiku via the Anthropic SDK. If the key is missing, all judge calls fail.

**Status:** Handled — `run_eval` checks `ANTHROPIC_API_KEY` at startup and raises `EnvironmentError` with a clear message before any cases run. No partial results are produced with a missing key.

---

## Failure 3: Cases File Missing or Empty (Handled)

**What breaks:** `load_cases()` is called before the run starts.

**Status:** Handled — `FileNotFoundError` raised if file doesn't exist; `ValueError("No cases found")` raised if file is empty. Both surface before any DB writes.

---

## Failure 4: SQLite Lock Under Concurrent Runs (Documented Gap)

**What breaks:** SQLite supports one writer at a time. If two eval runs are triggered concurrently (e.g., parallel CI jobs), the second run will hit `OperationalError: database is locked`. Per-case writes (not batched) make this more likely — each result write holds the lock briefly.

**Status:** Documented gap. Per the tradeoffs doc, this is acceptable for local use; the fix for CI/CD is per-run database files (`evals_{run_id}.db`) or switching to PostgreSQL.

---

## Failure 5: Judge Returns Malformed JSON (Handled)

**What breaks:** The LLM judge is asked to return structured JSON (`{"correctness": N, ...}`). If the response contains extra text, code fences, or truncated JSON, parsing fails.

**Status:** Handled — `judge.py` strips code fences with regex before `json.loads()`; a `try/except json.JSONDecodeError` falls back to `None` scores with the error stored in the `error` field. The case is recorded, not dropped.

**Observable:** `error` field in the `results` table is non-null for affected cases. `dashboard.py` surfaces parse failures in the run summary.

---

## Failure 6: Regression Threshold False Alarm (Design Boundary)

**What breaks:** The regression threshold is 1.0 point drop on a 1–5 scale. At low case counts (e.g., 5 cases per category), a single judge variance swing of ±0.3 on one case is amplified. Two judge variance events in the same category produce a 0.6-point swing — close to the threshold.

**Status:** Not a bug — a documented design boundary. Threshold is configurable via `--regression-threshold`. At n≥20 cases per category, judge variance averages out and the threshold is reliable.
