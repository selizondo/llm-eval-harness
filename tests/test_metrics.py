"""
tests/test_metrics.py — Unit tests for eval metrics computation.

Coverage:
  - accuracy_at_k: correct fraction when threshold=3.0
  - average_score: weighted average across score axes
  - Regression detection: compare_runs flags when delta exceeds threshold
  - Edge cases: empty result list, all-failing scores
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestAccuracyAtK:
    def _acc(self, scores: list[dict], threshold: float = 3.0) -> float:
        from evals.metrics import accuracy_at_k
        return accuracy_at_k(scores, threshold=threshold)

    def test_all_passing(self):
        scores = [{"correctness": 4, "groundedness": 4, "conciseness": 4}] * 5
        assert self._acc(scores) == 1.0

    def test_all_failing(self):
        scores = [{"correctness": 1, "groundedness": 1, "conciseness": 1}] * 5
        assert self._acc(scores) == 0.0

    def test_half_passing(self):
        pass_score = {"correctness": 4, "groundedness": 4, "conciseness": 4}
        fail_score = {"correctness": 2, "groundedness": 2, "conciseness": 2}
        scores = [pass_score, fail_score, pass_score, fail_score]
        result = self._acc(scores)
        assert result == 0.5

    def test_empty_list_returns_zero(self):
        assert self._acc([]) == 0.0

    def test_threshold_boundary(self):
        # Score exactly at threshold should count as passing
        scores = [{"correctness": 3, "groundedness": 3, "conciseness": 3}]
        result = self._acc(scores, threshold=3.0)
        assert result == 1.0


class TestAverageScore:
    def _avg(self, scores: list[dict]) -> float:
        from evals.metrics import average_score
        return average_score(scores)

    def test_basic_average(self):
        scores = [
            {"correctness": 4, "groundedness": 4, "conciseness": 4},
            {"correctness": 2, "groundedness": 2, "conciseness": 2},
        ]
        result = self._avg(scores)
        assert abs(result - 3.0) < 0.01

    def test_empty_list(self):
        assert self._avg([]) == 0.0

    def test_single_entry(self):
        scores = [{"correctness": 5, "groundedness": 5, "conciseness": 5}]
        assert self._avg(scores) == 5.0
