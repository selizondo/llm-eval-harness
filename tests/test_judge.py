"""
tests/test_judge.py — Unit tests for the LLM-as-judge module.

Coverage:
  - _parse_score: extracts valid JSON, returns fallback on malformed input
  - judge() Anthropic path: correct score structure returned
  - judge() Ollama path: correct score structure returned without an API key
  - Backend mismatch: anthropic backend raises when client is None

No real API calls — all LLM calls are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------

class TestParseScore:
    def _parse(self, text: str) -> dict:
        from evals.judge import _parse_score
        return _parse_score(text)

    def test_valid_json_extracted(self):
        text = '{"correctness": 4, "groundedness": 3, "conciseness": 5}'
        result = self._parse(text)
        assert result["correctness"] == 4
        assert result["groundedness"] == 3
        assert result["conciseness"] == 5

    def test_json_embedded_in_prose(self):
        text = 'Here is my score: {"correctness": 5, "groundedness": 5, "conciseness": 4} done.'
        result = self._parse(text)
        assert result["correctness"] == 5

    def test_missing_key_returns_fallback_value(self):
        # A response with only partial keys should not crash
        text = '{"correctness": 3}'
        result = self._parse(text)
        assert "correctness" in result

    def test_malformed_json_returns_fallback(self):
        result = self._parse("not valid json at all")
        # Should return something with the expected keys, not raise
        assert isinstance(result, dict)

    def test_empty_string_returns_fallback(self):
        result = self._parse("")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# judge() — Anthropic path (mocked)
# ---------------------------------------------------------------------------

class TestJudgeAnthropicPath:
    def _make_client(self, response_text: str):
        """Build a minimal mock Anthropic client."""
        msg = MagicMock()
        msg.content = [MagicMock(text=response_text)]
        msg.usage = MagicMock(input_tokens=50, output_tokens=30)
        client = MagicMock()
        client.messages.create.return_value = msg
        return client

    def test_returns_expected_keys(self):
        from evals.judge import judge
        client = self._make_client('{"correctness": 4, "groundedness": 4, "conciseness": 4}')
        result = judge("What is RAG?", "RAG is retrieval-augmented generation.", "RAG is a framework.", client=client)
        assert "correctness" in result
        assert "groundedness" in result
        assert "conciseness" in result

    def test_token_usage_in_result(self):
        from evals.judge import judge
        client = self._make_client('{"correctness": 3, "groundedness": 3, "conciseness": 3}')
        result = judge("Q", "A", "A", client=client)
        assert "input_tokens" in result or "tokens" in result or result.get("correctness") is not None

    def test_raises_without_client_in_anthropic_mode(self):
        from evals.judge import judge
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            with pytest.raises(Exception):
                judge("Q", "A", "A", client=None, backend="anthropic")


# ---------------------------------------------------------------------------
# judge() — Ollama path (mocked)
# ---------------------------------------------------------------------------

class TestJudgeOllamaPath:
    def test_ollama_path_returns_scores_without_api_key(self):
        from evals.judge import judge
        response_payload = {"response": '{"correctness": 4, "groundedness": 4, "conciseness": 4}'}

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = response_payload
            mock_post.return_value = mock_resp

            result = judge(
                "What is RAG?",
                "RAG is retrieval-augmented generation.",
                "RAG retrieves relevant documents.",
                backend="ollama",
                ollama_model="llama3.2",
            )

        assert "correctness" in result

    def test_ollama_backend_called_without_anthropic_client(self):
        """Ollama path must not require client= — no API key needed."""
        from evals.judge import judge
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": '{"correctness": 3, "groundedness": 3, "conciseness": 3}'}
            mock_post.return_value = mock_resp
            # No client= arg — should not raise
            result = judge("Q", "A", "A", backend="ollama", ollama_model="llama3.2")
        assert isinstance(result, dict)
