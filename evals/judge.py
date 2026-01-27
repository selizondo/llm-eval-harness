"""
judge.py — LLM-as-judge: score a model's answer against a golden answer.

Uses Claude Haiku for cost-efficient, reproducible scoring.
Returns scores 1–5 on three axes: correctness, groundedness, conciseness.

Temperature=0 for reproducibility. Two runs should agree within ±0.3 on average.
"""

import json
import os
import re
import time

import anthropic

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are an expert evaluator of AI/ML question-answering systems.
Your job is to score a model's answer against a golden reference answer on three axes.
Be consistent and calibrated: a score of 3 means acceptable but not great, 5 means excellent.
Always return valid JSON — nothing else."""

JUDGE_PROMPT = """Evaluate the following answer to an ML/AI question.

QUESTION:
{question}

GOLDEN ANSWER (reference):
{golden_answer}

MODEL ANSWER (to evaluate):
{model_answer}

Score the model answer on these three axes (1–5 each):

- correctness: Does the answer contain the key facts from the golden answer?
  5 = all key facts present and accurate
  4 = most key facts present, minor omissions
  3 = some key facts present, noticeable gaps
  2 = few key facts, significant errors or omissions
  1 = wrong or completely off-topic

- groundedness: Does the answer stay factual with no hallucinations or unsupported claims?
  5 = every claim is accurate and supported
  4 = mostly accurate, one minor unsupported detail
  3 = some unsupported or questionable claims
  2 = notable hallucinations present
  1 = answer is mostly fabricated or contradicts known facts

- conciseness: Is the answer appropriately scoped — not too verbose, not too terse?
  5 = tight, well-scoped answer that covers what's needed
  4 = slightly over/under but still useful
  3 = noticeably padded or too brief to be useful
  2 = very long with filler, or so brief it's unhelpful
  1 = extreme verbosity or one-line non-answer

Return ONLY this JSON:
{{
  "correctness": <1-5>,
  "groundedness": <1-5>,
  "conciseness": <1-5>,
  "reasoning": "<one sentence explaining the scores>"
}}"""


def judge(
    question: str,
    golden_answer: str,
    model_answer: str,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """
    Score a model answer. Returns:
        {correctness, groundedness, conciseness, reasoning, judge_latency_ms}
    On judge failure, returns scores of 0 with error in reasoning.
    """
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set. Set it before running the eval harness.")
        client = anthropic.Anthropic(api_key=api_key)

    prompt = JUDGE_PROMPT.format(
        question=question,
        golden_answer=golden_answer,
        model_answer=model_answer,
    )

    # WHY retry instead of immediate fallback:
    #   Anthropic's API occasionally returns transient errors (rate limits, 5xx).
    #   A single failure should not permanently mark a case as 0-scored — one retry
    #   resolves most transient glitches. Two retries with exponential backoff
    #   (0.5s → 2.0s) keeps total added latency under 3s while tolerating brief
    #   service hiccups. On all retries exhausted, fall back to 0-scores rather
    #   than raising — the eval run should complete, not crash.
    BACKOFF_DELAYS = [0.5, 2.0]  # seconds between retry attempts
    MAX_ATTEMPTS = 1 + len(BACKOFF_DELAYS)  # 3 total: 1 initial + 2 retries

    t0 = time.time()
    last_error: str = ""

    for attempt in range(MAX_ATTEMPTS):
        if attempt > 0:
            time.sleep(BACKOFF_DELAYS[attempt - 1])
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=256,
                temperature=0,          # deterministic for reproducibility
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = int((time.time() - t0) * 1000)
            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            scores = json.loads(raw)
            return {
                "correctness": int(scores["correctness"]),
                "groundedness": int(scores["groundedness"]),
                "conciseness": int(scores["conciseness"]),
                "reasoning": scores.get("reasoning", ""),
                "judge_latency_ms": latency_ms,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except json.JSONDecodeError as e:
            last_error = f"Judge parse error: {e} | raw={raw[:200]}"
            # Parse errors are deterministic — retrying the same call won't help.
            break
        except Exception as e:
            last_error = f"Judge error: {e}"
            # Transient API error — retry if attempts remain.

    return {
        "correctness": 0, "groundedness": 0, "conciseness": 0,
        "reasoning": last_error,
        "judge_latency_ms": int((time.time() - t0) * 1000),
        "input_tokens": 0, "output_tokens": 0,
        "error": "judge_failed",
    }
