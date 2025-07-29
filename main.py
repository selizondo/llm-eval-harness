"""
main.py — CLI entrypoint for the LLM eval harness.

Usage:
    # Evaluate Claude Haiku directly on RAG QA cases
    python main.py --cases evals/cases/rag_qa.jsonl --model anthropic_direct --tag "haiku_baseline"

    # Evaluate with comparison to a previous run (regression detection)
    python main.py --cases evals/cases/rag_qa.jsonl --model anthropic_direct --tag "haiku_v2" --compare <run_id>

    # Show per-case detail for a run
    python main.py --show-cases <run_id>

    # List all historical runs
    python main.py --list-runs

    # Quick smoke test (first 3 cases only)
    python main.py --cases evals/cases/rag_qa.jsonl --model anthropic_direct --tag "smoke" --limit 3
"""

import argparse
import os
import sys

from evals.harness import run_eval
from evals.dashboard import print_run_summary, print_case_detail, print_runs_list

DB_PATH = "./evals.db"


# ---------------------------------------------------------------------------
# Model wrappers — add your own here
# ---------------------------------------------------------------------------

def make_anthropic_direct(model: str = "claude-haiku-4-5"):
    """Direct Anthropic call — no retrieval. Baseline for comparison."""
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Set it before running the eval harness.")
    client = anthropic.Anthropic(api_key=api_key)

    def fn(question: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0,
            system=(
                "You are an expert ML engineer. Answer the following question clearly "
                "and concisely, as you would in a technical interview."
            ),
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text.strip()

    return fn


def make_rag_pipeline(
    corpus_path: str = "../rag-pipeline-from-scratch/corpus",
    db_path: str = "../rag-pipeline-from-scratch/chroma_db_hf",
    chunk_size: int = 256,
    overlap: int = 32,
    top_k: int = 5,
    rerank: bool = False,
    ollama_model: str = "llama3.2",
):
    """
    Wrapper for the RAG pipeline from Project 01.
    Requires Ollama running locally and the corpus already ingested.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../rag-pipeline-from-scratch"))
    try:
        from retrieve import retrieve
        from generate import generate
    except ImportError as e:
        raise ImportError(f"RAG pipeline not found: {e}. Point --rag-path to rag-pipeline-from-scratch/")

    def fn(question: str) -> str:
        chunks = retrieve(question, top_k=top_k, db_path=db_path, rerank=rerank)
        return generate(question, chunks, model=ollama_model)

    return fn


def make_agent():
    """Wrapper for the LLM agent from Project 02."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../llm-agent-tool-use"))
    try:
        from agent.agent import run as agent_run
    except ImportError as e:
        raise ImportError(f"Agent not found: {e}. Point to llm-agent-tool-use/")

    def fn(question: str) -> str:
        result = agent_run(question, verbose=False)
        return result["answer"]

    return fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_CHOICES = ["anthropic_direct", "rag", "agent"]


def main():
    parser = argparse.ArgumentParser(description="LLM eval harness")
    subparsers = parser.add_subparsers(dest="command")

    # run command
    run_p = subparsers.add_parser("run", help="Run evaluation")
    run_p.add_argument("--cases", required=True, help="Path to JSONL test cases")
    run_p.add_argument("--model", choices=MODEL_CHOICES, default="anthropic_direct")
    run_p.add_argument("--tag", required=True, help="Human-readable label for this run")
    run_p.add_argument("--compare", metavar="RUN_ID", help="Run ID to compare against for regression detection")
    run_p.add_argument("--limit", type=int, help="Only evaluate first N cases")
    run_p.add_argument("--db", default=DB_PATH)
    run_p.add_argument("--detail", action="store_true", help="Print per-case table after summary")
    # RAG-specific options
    run_p.add_argument("--rag-db", default="../rag-pipeline-from-scratch/chroma_db_hf")
    run_p.add_argument("--rag-corpus", default="../rag-pipeline-from-scratch/corpus")
    run_p.add_argument("--rerank", action="store_true")
    run_p.add_argument("--top-k", type=int, default=5)

    # show-cases command
    show_p = subparsers.add_parser("show-cases", help="Show per-case results for a run")
    show_p.add_argument("run_id")
    show_p.add_argument("--db", default=DB_PATH)

    # list-runs command
    list_p = subparsers.add_parser("list-runs", help="List all historical runs")
    list_p.add_argument("--db", default=DB_PATH)

    # summary command
    sum_p = subparsers.add_parser("summary", help="Print summary for a run")
    sum_p.add_argument("run_id")
    sum_p.add_argument("--compare", metavar="RUN_ID")
    sum_p.add_argument("--db", default=DB_PATH)

    args = parser.parse_args()

    if args.command == "list-runs":
        print_runs_list(args.db)

    elif args.command == "show-cases":
        print_case_detail(args.db, args.run_id)

    elif args.command == "summary":
        print_run_summary(args.db, args.run_id, args.compare)

    elif args.command == "run":
        # Build model function
        if args.model == "anthropic_direct":
            model_fn = make_anthropic_direct()
        elif args.model == "rag":
            model_fn = make_rag_pipeline(db_path=args.rag_db, top_k=args.top_k, rerank=args.rerank)
        elif args.model == "agent":
            model_fn = make_agent()

        config = {"model": args.model, "cases": args.cases, "limit": args.limit}

        run_id = run_eval(
            cases_path=args.cases,
            model_fn=model_fn,
            model_tag=args.tag,
            config=config,
            db_path=args.db,
            verbose=True,
            limit=args.limit,
        )

        print_run_summary(args.db, run_id, args.compare)
        if args.detail:
            print_case_detail(args.db, run_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
