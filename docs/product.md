# Product and Cost

This document frames the project for a technical business reviewer: what organizational risk it addresses, how it earns trust, what it costs, and when a team should build something like this versus buying a managed solution.

---

## The Business Problem

AI engineering teams operate without a safety net. A model swap, a prompt change, a retrieval parameter adjustment: any of these can degrade output quality. The degradation does not raise an exception. It shows up as a drop in user satisfaction, an increase in support tickets, or a missed SLA that gets traced back to a model change made weeks ago.

Most teams discover regressions through user feedback, not instrumentation. By the time the problem surfaces, the cause is buried in a changelog. The business cost is not just the regression: it is the time to diagnose, reproduce, and fix something that could have been caught in 30 seconds before it shipped.

---

## Trust Surface

**What can go wrong in a bare LLM system:**

- A prompt change improves responses on the cases the team tested and degrades 20% of production queries nobody tested
- A model upgrade changes tone, reasoning depth, or citation style in ways that look fine in manual review but fail on edge cases
- A RAG configuration change shifts retrieval quality in ways that only show up on semantically similar but differently phrased questions

**How this harness addresses each:**

- Static JSONL test cases cover edge cases, not just the happy path. Cases for RAG include ambiguous questions, multi-hop reasoning, and out-of-context queries that expose retrieval gaps.
- Three-axis scoring (correctness, groundedness, conciseness) catches failures that a single composite score masks. A model that becomes more verbose after a prompt change will fail conciseness without dropping correctness. A single score would average those out.
- Regression threshold is derived from judge variance (3 sigma above noise), not set arbitrarily. This means alerts are real regressions, not measurement noise.

**What is not addressed here:** This harness runs on-demand in a development or CI context. It does not monitor live production traffic. For continuous production monitoring between releases, see [llm-drift-monitor](https://github.com/selizondo/llm-drift-monitor).

---

## Cost Model

| Scale | Cases per run | Judge cost | Time to run |
|-------|--------------|------------|-------------|
| Smoke test | 3 | ~$0.005 | ~10s |
| Standard eval | 24 | ~$0.038 | ~60 to 90s |
| Extended eval | 100 | ~$0.15 | ~5 to 10min |
| Large eval | 500 | ~$0.75 | ~20 to 40min |

Judge cost uses Claude Haiku at approximately $1/1M tokens. Ollama reduces judge cost to zero for offline or CI use with no API key.

**Model inference cost (if evaluating a hosted model):** Separate from judge cost. At 24 cases with Claude Haiku as the model under test, add approximately $0.01 to $0.05 for model calls, depending on output length.

**Total cost for a standard regression check:** Under $0.10 per run. Running twice a day across a team of five engineers costs under $100/month.

**Inflection points:**
- Above 500 cases per run: sequential eval latency (20 to 40 min) becomes the constraint. Parallelize judge calls with asyncio or a thread pool.
- Above 50 concurrent runs: SQLite single-writer bottleneck. Migrate to PostgreSQL or per-run database files.

---

## Market Context

The LLM evaluation tooling market split into two camps in 2023 to 2025: managed cloud platforms (LangSmith, Braintrust, Galileo, Langfuse) and framework-level instrumentation (LangChain callbacks, RAGAS). Both have strong adoption. Both have the same gap: they require buying into an external service or a specific framework before you can measure anything.

Teams building custom pipelines (RAG from scratch, fine-tuned models, multi-step agents) often cannot plug into managed eval platforms without adapter work that takes longer than building a minimal harness. This project demonstrates that a useful eval loop is 4 Python files and a JSONL test set: harness, judge, metrics, dashboard. No external service required to get to a number.

The signal for a hiring reviewer: the engineer understands what the tools are doing, not just how to configure them.

---

## Deployment Constraints

**CI integration:** The harness runs as a standard Python script. GitHub Actions integration requires one workflow step and a stored `ANTHROPIC_API_KEY` secret. Block merge if `hallucination_rate > threshold` or `accuracy_at_4 < baseline`. The SQLite database persists between CI runs if stored as an artifact.

**Ollama fallback:** Set `JUDGE_MODEL=ollama` in `.env` to run the judge locally. Useful for CI environments where API credentials are not available, or for cost-sensitive development workflows.

**Latency SLA for blocking CI:** A 24-case eval run completes in 30 to 120 seconds. This is acceptable as a pre-merge gate on a team that ships LLM changes multiple times per day. At 100+ cases, consider running the eval as a post-merge check rather than a blocking pre-merge gate.

**On-call implications:** No production alerting in this tool. Failures surface as non-zero exit codes in CI. If the eval harness itself fails (Anthropic API down, SQLite locked), the CI step fails and the team investigates. No silent failures.

---

## Build vs Buy

**Build (this approach) when:**

- The team has a custom pipeline (not LangChain-native) and cannot plug directly into managed eval platforms without adapter work
- The team needs eval results that persist across weeks and months for trend analysis, not just the latest run
- Cost is a constraint and Claude Haiku or a local Ollama judge is sufficient for relative comparisons
- The team wants to own the scoring rubric rather than depend on a vendor's definition of "quality"

**Buy or use a managed platform (LangSmith, Braintrust, Langfuse) when:**

- The team is already on LangChain or a framework with native integrations to these platforms
- Real-time production trace monitoring is required alongside regression testing
- The team needs a visual UI for stakeholders who are not comfortable with terminal output
- The eval volume exceeds 10k cases per day (managed platforms handle scale automatically)

**The judgment call a Staff Engineer owns:** Managed eval platforms add a vendor dependency and a recurring cost that compounds as the team scales. A custom harness adds maintenance overhead. The right call depends on whether the team's pipeline is standard enough to fit a managed platform's model. For custom RAG pipelines and agents, the harness is typically faster to wire up than adapting a managed platform. Once the pipeline matures and the team standardizes, migrating to a managed platform becomes worth it.
