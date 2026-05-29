.PHONY: bootstrap test smoke eval list clean

bootstrap:
	UV_PROJECT_ENVIRONMENT=.venv uv sync

test:
	uv run pytest

smoke:
	uv run python main.py run --cases evals/cases/rag_qa.jsonl --model anthropic_direct --tag smoke --limit 3

eval:
	uv run python main.py run --cases evals/cases/rag_qa.jsonl --model rag --tag rag_eval

eval-compare:
	@echo "Usage: make eval-compare COMPARE=<run_id>"
	uv run python main.py run --cases evals/cases/rag_qa.jsonl --model rag --tag rag_eval --compare $(COMPARE)

list:
	uv run python main.py list-runs

clean:
	rm -f evals.db
