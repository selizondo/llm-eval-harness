"""
harness.py — Core eval loop: load cases → run model → judge → store results.

The model under test is passed as a callable: (question: str) -> str.
This makes the harness model-agnostic — wire any model, pipeline, or agent.

Usage (from CLI via main.py):
    python main.py --cases evals/cases/rag_qa.jsonl --model anthropic_direct --tag "haiku_baseline"
    python main.py --cases evals/cases/rag_qa.jsonl --model rag --tag "rag_v1"
    python main.py --cases evals/cases/agent.jsonl  --model agent --tag "agent_v1"
"""

import json
import os
import sqlite3
import time
import uuid
from typing import Callable


DB_PATH = "./evals.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id        TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model_tag TEXT NOT NULL,
    config    TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    case_id           TEXT NOT NULL,
    input             TEXT NOT NULL,
    model_output      TEXT,
    golden_answer     TEXT,
    correctness       INTEGER,
    groundedness      INTEGER,
    conciseness       INTEGER,
    reasoning         TEXT,
    model_latency_ms  INTEGER,
    judge_latency_ms  INTEGER,
    judge_input_tokens  INTEGER DEFAULT 0,
    judge_output_tokens INTEGER DEFAULT 0,
    error             TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def load_cases(jsonl_path: str) -> list[dict]:
    cases = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_eval(
    cases_path: str,
    model_fn: Callable[[str], str],
    model_tag: str,
    config: dict | None = None,
    db_path: str = DB_PATH,
    verbose: bool = True,
    limit: int | None = None,
) -> str:
    """
    Run the eval loop.

    Args:
        cases_path:  Path to JSONL test cases file.
        model_fn:    Callable that takes a question string and returns an answer string.
        model_tag:   Human-readable identifier for this run (e.g., "rag_chunk256_v1").
        config:      Arbitrary dict of run metadata (stored as JSON).
        db_path:     Path to SQLite database.
        verbose:     Print progress as cases are evaluated.
        limit:       Only run the first N cases (useful for quick smoke tests).

    Returns:
        run_id (str) — use with metrics.compute_summary() to get aggregated stats.
    """
    from .judge import judge
    import anthropic

    init_db(db_path)

    run_id = str(uuid.uuid4())[:8]
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    cases = load_cases(cases_path)
    if limit:
        cases = cases[:limit]

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO runs (id, timestamp, model_tag, config) VALUES (?, ?, ?, ?)",
        (run_id, timestamp, model_tag, json.dumps(config or {})),
    )
    conn.commit()

    if verbose:
        print(f"\nRun {run_id}  |  model={model_tag}  |  {len(cases)} cases\n")

    for i, case in enumerate(cases):
        case_id = case["id"]
        question = case["input"]
        golden = case["golden_answer"]

        # --- Run model under test ---
        model_output = None
        model_error = None
        t0 = time.time()
        try:
            model_output = model_fn(question)
        except Exception as e:
            model_error = str(e)
            model_output = ""
        model_latency_ms = int((time.time() - t0) * 1000)

        # --- Judge ---
        if model_output:
            scores = judge(question, golden, model_output, client=client)
        else:
            scores = {
                "correctness": 0, "groundedness": 0, "conciseness": 0,
                "reasoning": f"Model error: {model_error}",
                "judge_latency_ms": 0, "input_tokens": 0, "output_tokens": 0,
            }

        avg = (scores["correctness"] + scores["groundedness"] + scores["conciseness"]) / 3
        if verbose:
            status = "✓" if avg >= 4 else "~" if avg >= 3 else "✗"
            print(
                f"  [{i+1:02d}/{len(cases)}] {status} {case_id:<12} "
                f"C={scores['correctness']} G={scores['groundedness']} "
                f"P={scores['conciseness']}  avg={avg:.1f}  "
                f"model={model_latency_ms}ms"
            )

        conn.execute("""
            INSERT INTO results
              (run_id, case_id, input, model_output, golden_answer,
               correctness, groundedness, conciseness, reasoning,
               model_latency_ms, judge_latency_ms,
               judge_input_tokens, judge_output_tokens, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, case_id, question, model_output, golden,
            scores["correctness"], scores["groundedness"], scores["conciseness"],
            scores["reasoning"],
            model_latency_ms, scores["judge_latency_ms"],
            scores.get("input_tokens", 0), scores.get("output_tokens", 0),
            model_error,
        ))
        conn.commit()

    conn.close()
    if verbose:
        print(f"\nRun complete: {run_id}")
    return run_id
