# Scale Boundaries

Explicit limits for the eval harness. These are the points where the current architecture breaks down and what to do about it.

---

## SQLite Storage

**Single-writer constraint**
SQLite uses file-level locking. Concurrent eval runs writing to the same `evals.db` will encounter `SQLITE_BUSY` errors — only one writer can hold the lock at a time. This is fine for a single researcher running sequential evals; it breaks for parallel CI jobs or multi-user setups.

- **Workaround (single machine):** Serialize eval runs or use separate `--db` paths per job.
- **Migration path (team/CI):** Swap to PostgreSQL or a hosted metrics store (Langfuse, W&B). The `harness.py` storage layer is isolated enough to re-point with a thin adapter.

**Row limit**
SQLite handles tens of millions of rows without issue, but query performance degrades noticeably around 10M rows in the `results` table without explicit indexes on `run_id` and `case_id`. The current eval set (24–100 cases per run) reaches 10M rows only after ~100k eval runs — not a near-term concern.

- **If needed:** Add `CREATE INDEX idx_results_run_id ON results(run_id);` after schema creation.

---

## Anthropic Judge Rate Limits

The judge calls Claude Haiku once per eval case. Rate limits depend on your Anthropic usage tier:

| Tier | Requests/min (approx.) | Tokens/min (approx.) |
|------|------------------------|----------------------|
| Free / Tier 1 | 50 req/min | 50k tokens/min |
| Tier 2 | 1,000 req/min | 100k tokens/min |
| Tier 4+ | 4,000+ req/min | 400k+ tokens/min |

At a typical judge latency of 600–1,200ms per call, a single sequential eval run saturates at roughly **50–100 cases/minute** — well within Tier 1 limits. If you hit rate limit errors, the judge retry logic (0.5s → 2.0s backoff) absorbs brief bursts; sustained rate limiting requires adding a `time.sleep()` between cases or upgrading your tier.

---

## Evaluation Throughput

Estimated throughput for a sequential eval run:

| Bottleneck | Latency (typical) | Cases/minute |
|------------|-------------------|--------------|
| Direct Anthropic model (model_fn) | 800–1,500ms | 40–75 |
| RAG pipeline (model_fn, Ollama) | 2,000–4,000ms | 15–30 |
| Judge call | 600–1,200ms | 50–100 |
| **Combined (sequential)** | **1,400–2,700ms/case** | **22–43** |

A 24-case eval run completes in roughly **30–120 seconds** depending on model speed.

**To go faster:** Run multiple eval processes with separate `--db` paths and aggregate results manually. Parallel judge calls within a single run are not implemented — adding `asyncio` or a thread pool for the judge would be the lever.

---

## Model Timeout

The `model_fn` timeout is hardcoded at 30 seconds (see `harness.py`). This covers:
- Direct Anthropic calls: typically complete in under 5s
- RAG pipelines with Ollama: typically 2–10s; hangs possible if Ollama is down

If your model_fn consistently needs more than 30s, the timeout is not currently configurable via CLI — edit `harness.py` directly.

---

## Memory

Each result object in memory is roughly 2–5KB (model output + judge response + metadata). A 100-case eval run holds ~500KB in the SQLite connection buffer before the final commit. No concern until eval sets exceed ~10k cases in a single run.
