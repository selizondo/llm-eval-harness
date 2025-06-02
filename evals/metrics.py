"""
metrics.py — Aggregate eval results and compute summary statistics.
"""

import sqlite3
from dataclasses import dataclass

# Haiku pricing as of 2025 (per million tokens)
HAIKU_INPUT_PRICE_PER_M = 1.00
HAIKU_OUTPUT_PRICE_PER_M = 5.00


@dataclass
class RunSummary:
    run_id: str
    model_tag: str
    n_cases: int
    accuracy_at_4: float        # % cases with avg score >= 4.0
    hallucination_rate: float   # % cases with groundedness < 3
    avg_correctness: float
    avg_groundedness: float
    avg_conciseness: float
    p50_latency_ms: int
    p95_latency_ms: int
    total_judge_cost_usd: float
    regression_vs: str | None = None
    regressed_cases: list[str] | None = None


def compute_summary(db_path: str, run_id: str, compare_run_id: str | None = None) -> RunSummary:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT model_tag FROM runs WHERE id = ?", (run_id,))
    row = cur.fetchone()
    model_tag = row["model_tag"] if row else run_id

    cur.execute("""
        SELECT case_id, correctness, groundedness, conciseness,
               model_latency_ms, judge_input_tokens, judge_output_tokens
        FROM results WHERE run_id = ?
    """, (run_id,))
    rows = cur.fetchall()

    if not rows:
        conn.close()
        return RunSummary(run_id=run_id, model_tag=model_tag, n_cases=0,
                          accuracy_at_4=0, hallucination_rate=0,
                          avg_correctness=0, avg_groundedness=0, avg_conciseness=0,
                          p50_latency_ms=0, p95_latency_ms=0, total_judge_cost_usd=0)

    n = len(rows)
    avg_score = lambda key: sum(r[key] for r in rows) / n  # noqa: E731

    scores_avg = [(r["correctness"] + r["groundedness"] + r["conciseness"]) / 3 for r in rows]
    accuracy_at_4 = sum(1 for s in scores_avg if s >= 4.0) / n
    hallucination_rate = sum(1 for r in rows if r["groundedness"] < 3) / n

    latencies = sorted(r["model_latency_ms"] for r in rows)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[min(int(n * 0.95), n - 1)]

    total_input = sum(r["judge_input_tokens"] for r in rows)
    total_output = sum(r["judge_output_tokens"] for r in rows)
    cost = (total_input / 1_000_000 * HAIKU_INPUT_PRICE_PER_M +
            total_output / 1_000_000 * HAIKU_OUTPUT_PRICE_PER_M)

    # Regression detection
    regressed = []
    if compare_run_id:
        cur.execute("""
            SELECT case_id,
                   (correctness + groundedness + conciseness) / 3.0 AS avg_score
            FROM results WHERE run_id = ?
        """, (compare_run_id,))
        prev = {r["case_id"]: r["avg_score"] for r in cur.fetchall()}
        for r in rows:
            curr_avg = (r["correctness"] + r["groundedness"] + r["conciseness"]) / 3.0
            prev_avg = prev.get(r["case_id"])
            if prev_avg is not None and prev_avg - curr_avg >= 1.0:
                regressed.append(r["case_id"])

    conn.close()
    return RunSummary(
        run_id=run_id,
        model_tag=model_tag,
        n_cases=n,
        accuracy_at_4=accuracy_at_4,
        hallucination_rate=hallucination_rate,
        avg_correctness=avg_score("correctness"),
        avg_groundedness=avg_score("groundedness"),
        avg_conciseness=avg_score("conciseness"),
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        total_judge_cost_usd=cost,
        regression_vs=compare_run_id,
        regressed_cases=regressed if compare_run_id else None,
    )


def list_runs(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, model_tag, timestamp, config FROM runs ORDER BY timestamp DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
