.PHONY: install smoke eval list clean

install:
	pip install -r requirements.txt

smoke:
	python main.py run --cases evals/cases/rag_qa.jsonl --model anthropic_direct --tag smoke --limit 3

eval:
	python main.py run --cases evals/cases/rag_qa.jsonl --model rag --tag rag_eval

eval-compare:
	@echo "Usage: make eval-compare COMPARE=<run_id>"
	python main.py run --cases evals/cases/rag_qa.jsonl --model rag --tag rag_eval --compare $(COMPARE)

list:
	python main.py list-runs

clean:
	rm -f evals.db
