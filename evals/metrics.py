"""
metrics.py — Aggregate eval results and compute summary statistics.

Key design choices:
  - Three-axis scoring (correctness, groundedness, conciseness) instead of a
    single 1-5 scale. WHY: a RAG system with high correctness but low groundedness
    is hallucinating plausible-sounding content. Low correctness but high
    groundedness means the system is overly conservative (just repeating the source).
    Observing all three axes pinpoints *which kind* of failure is happening.

  - accuracy_at_4: percentage of cases where the average score >= 4.0.
    WHY 4.0 as threshold: on a 1-5 scale, 4.0 corresponds to "good with minor
    issues". Below 4 means something is meaningfully wrong. This threshold is
    deliberately conservative — claim improvement only when things are clearly good.

  - Regression detection fires at a >= 1.0 drop in average score per case.
    WHY 1.0: one full point on the 1-5 scale is a clearly noticeable quality drop.
    Smaller thresholds (0.5) would generate too many alerts on judge variance.
"""

import sqlite3
from dataclasses import dataclass

# Haiku pricing as of 2025-05 (per million tokens).
# Update these when Claude pricing changes — they affect cost tracking.
HAIKU_INPUT_PRICE_PER_M = 1.00
HAIKU_OUTPUT_PRICE_PER_M = 5.00

# Cases with average score >= this threshold are counted as "accurate"
ACCURACY_THRESHOLD = 4.0

# Cases with groundedness < this are counted as hallucinations
HALLUCINATION_THRESHOLD = 3

# A drop of >= this in average score flags a regression between runs
REGRESSION_DELTA_THRESHOLD = 1.0


def accuracy_at_k(scores: list[dict], threshold: float = ACCURACY_THRESHOLD) -> float:
    """Fraction of cases where mean(correctness, groundedness, conciseness) >= threshold."""
    if not scores:
        return 0.0
    passing = sum(
        1 for s in scores
        if (s.get("correctness", 0) + s.get("groundedness", 0) + s.get("conciseness", 0)) / 3 >= threshold
    )
    return passing / len(scores)


def average_score(scores: list[dict]) -> float:
    """Mean of per-case average scores across all three axes."""
    if not scores:
        return 0.0
    return sum(
        (s.get("correctness", 0) + s.get("groundedness", 0) + s.get("conciseness", 0)) / 3
        for s in scores
    ) / len(scores)


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


def compute_summary(
    db_path: str,
    run_id: str,
    compare_run_id: str | None = None,
    regression_threshold: float = REGRESSION_DELTA_THRESHOLD,
) -> RunSummary:
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
    accuracy_at_4 = sum(1 for s in scores_avg if s >= ACCURACY_THRESHOLD) / n
    hallucination_rate = sum(1 for r in rows if r["groundedness"] < HALLUCINATION_THRESHOLD) / n

    latencies = sorted(r["model_latency_ms"] for r in rows)

    # WHY sorted-index percentile instead of numpy:
    #   numpy is not a dependency of this harness — keeping it pure stdlib avoids
    #   environment setup issues. The sorted-index approach is exact for discrete
    #   latency measurements (integer milliseconds).
    #
    # WHY min(..., n-1) for p95:
    #   int(n * 0.95) can equal n when n is small (e.g., n=20 → int(19.0) = 19,
    #   which is valid for a 0-indexed list of length 20). The min() guard is a
    #   belt-and-suspenders safety to prevent IndexError on edge-case small lists.
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[min(int(n * 0.95), n - 1)]

    total_input = sum(r["judge_input_tokens"] for r in rows)
    total_output = sum(r["judge_output_tokens"] for r in rows)
    cost = (total_input / 1_000_000 * HAIKU_INPUT_PRICE_PER_M +
            total_output / 1_000_000 * HAIKU_OUTPUT_PRICE_PER_M)

    # Regression detection: compare each case against the same case in a previous run.
    # WHY per-case instead of aggregate: an aggregate improvement can hide regressions
    # on specific cases. Tracking per-case lets you identify *which* questions got worse,
    # which is actionable (you can go look at those specific outputs).
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
            if prev_avg is not None and prev_avg - curr_avg >= regression_threshold:
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
