"""
dashboard.py — Rich terminal display for eval results.
"""

import sqlite3

from rich import box
from rich.console import Console
from rich.table import Table

from .metrics import compute_summary, list_runs

console = Console()


def _score_color(score: float) -> str:
    if score >= 4.0:
        return "green"
    if score >= 3.0:
        return "yellow"
    return "red"


def print_run_summary(
    db_path: str,
    run_id: str,
    compare_run_id: str | None = None,
    regression_threshold: float = 1.0,
):
    summary = compute_summary(db_path, run_id, compare_run_id, regression_threshold=regression_threshold)

    title = f"Eval Run: [bold]{summary.run_id}[/bold]  |  model: [cyan]{summary.model_tag}[/cyan]"
    if summary.regression_vs:
        title += f"  |  compared to: [dim]{summary.regression_vs}[/dim]"
    console.print(f"\n{title}")

    # Summary metrics table
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Metric", style="dim", width=24)
    t.add_column("Value", justify="right")

    acc_color = _score_color(summary.accuracy_at_4 * 5)
    hal_color = "green" if summary.hallucination_rate < 0.1 else ("yellow" if summary.hallucination_rate < 0.25 else "red")

    t.add_row("Cases evaluated", str(summary.n_cases))
    t.add_row("Accuracy@4 (avg ≥ 4.0)", f"[{acc_color}]{summary.accuracy_at_4:.0%}[/{acc_color}]")
    t.add_row("Hallucination rate", f"[{hal_color}]{summary.hallucination_rate:.0%}[/{hal_color}]")
    t.add_row("Avg correctness", f"[{_score_color(summary.avg_correctness)}]{summary.avg_correctness:.2f}[/{_score_color(summary.avg_correctness)}]")
    t.add_row("Avg groundedness", f"[{_score_color(summary.avg_groundedness)}]{summary.avg_groundedness:.2f}[/{_score_color(summary.avg_groundedness)}]")
    t.add_row("Avg conciseness", f"[{_score_color(summary.avg_conciseness)}]{summary.avg_conciseness:.2f}[/{_score_color(summary.avg_conciseness)}]")
    t.add_row("p50 latency", f"{summary.p50_latency_ms:,} ms")
    t.add_row("p95 latency", f"{summary.p95_latency_ms:,} ms")
    t.add_row("Judge cost (est.)", f"${summary.total_judge_cost_usd:.4f}")
    console.print(t)

    if summary.regressed_cases:
        console.print(f"[red bold]⚠  Regressions detected ({len(summary.regressed_cases)} cases):[/red bold]")
        for c in summary.regressed_cases:
            console.print(f"   • {c}")
    elif summary.regression_vs:
        console.print("[green]✓  No regressions vs previous run[/green]")


def print_case_detail(db_path: str, run_id: str):
    """Print per-case results table for a run."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT case_id, correctness, groundedness, conciseness,
               model_latency_ms, reasoning, error, model_output
        FROM results WHERE run_id = ?
        ORDER BY (correctness + groundedness + conciseness) ASC
    """, (run_id,))
    rows = cur.fetchall()
    conn.close()

    t = Table(title=f"Case Details — run {run_id}", box=box.SIMPLE_HEAVY, show_lines=False)
    t.add_column("ID", style="dim", width=12)
    t.add_column("C", justify="center", width=3)
    t.add_column("G", justify="center", width=3)
    t.add_column("P", justify="center", width=3)
    t.add_column("Avg", justify="center", width=5)
    t.add_column("ms", justify="right", width=7)
    t.add_column("Reasoning", width=55)

    for r in rows:
        avg = (r["correctness"] + r["groundedness"] + r["conciseness"]) / 3
        color = _score_color(avg)
        note = r["error"] or r["reasoning"] or ""
        t.add_row(
            r["case_id"],
            f"[{_score_color(r['correctness'])}]{r['correctness']}[/{_score_color(r['correctness'])}]",
            f"[{_score_color(r['groundedness'])}]{r['groundedness']}[/{_score_color(r['groundedness'])}]",
            f"[{_score_color(r['conciseness'])}]{r['conciseness']}[/{_score_color(r['conciseness'])}]",
            f"[{color}]{avg:.1f}[/{color}]",
            f"{r['model_latency_ms']:,}",
            note[:80],
        )
    console.print(t)


def print_runs_list(db_path: str):
    """Print all historical runs."""
    runs = list_runs(db_path)
    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    t = Table(title="All Eval Runs", box=box.SIMPLE)
    t.add_column("Run ID", style="cyan", width=10)
    t.add_column("Model Tag", width=30)
    t.add_column("Timestamp", width=20)

    for r in runs:
        t.add_row(r["id"], r["model_tag"], r["timestamp"])
    console.print(t)
