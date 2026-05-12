import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import openai


def _build_orchestrator(tmp_path):
    from orchestrator import MedicalOrchestrator

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "sample.txt").write_text("Sample cardiology content.", encoding="utf-8")
    return MedicalOrchestrator(knowledge_base_dir=str(kb))


class TestInputValidation:

    @pytest.mark.parametrize("query", ["", "   ", None])
    @patch("agents.cardiologist.OpenAIEmbeddings")
    @patch("agents.cardiologist.FAISS")
    def test_empty_query_rejected(self, mock_faiss, mock_emb, query, tmp_path):
        orch = _build_orchestrator(tmp_path)
        result = orch.answer(query)
        assert "valid" in result.lower()

    @patch("agents.cardiologist.OpenAIEmbeddings")
    @patch("agents.cardiologist.FAISS")
    def test_too_short_query_rejected(self, mock_faiss, mock_emb, tmp_path):
        orch = _build_orchestrator(tmp_path)
        result = orch.answer("hi")
        assert "short" in result.lower()


class TestAPIFailureHandling:

    @patch("agents.cardiologist.OpenAIEmbeddings")
    @patch("agents.cardiologist.FAISS")
    @patch("orchestrator.client")
    def test_auth_error_handled(self, mock_client, mock_faiss, mock_emb, tmp_path):
        mock_client.responses.create.side_effect = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body=None,
        )
        orch = _build_orchestrator(tmp_path)
        result = orch.answer("What is atrial fibrillation?")
        assert "authentication" in result.lower()

    @patch("agents.cardiologist.OpenAIEmbeddings")
    @patch("agents.cardiologist.FAISS")
    @patch("orchestrator.client")
    def test_rate_limit_handled(self, mock_client, mock_faiss, mock_emb, tmp_path):
        mock_client.responses.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )
        orch = _build_orchestrator(tmp_path)
        result = orch.answer("What is atrial fibrillation?")
        assert "rate limit" in result.lower()

    @patch("agents.cardiologist.OpenAIEmbeddings")
    @patch("agents.cardiologist.FAISS")
    @patch("orchestrator.client")
    def test_connection_error_handled(self, mock_client, mock_faiss, mock_emb, tmp_path):
        mock_client.responses.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        orch = _build_orchestrator(tmp_path)
        result = orch.answer("What is atrial fibrillation?")
        assert "connect" in result.lower()


class TestDataDirectoryErrors:

    @patch("agents.cardiologist.OpenAIEmbeddings")
    def test_missing_directory_raises(self, mock_emb):
        from agents.cardiologist import CardiologistAgent

        with pytest.raises(FileNotFoundError):
            CardiologistAgent(folder_path="/nonexistent/path")

    @patch("agents.cardiologist.OpenAIEmbeddings")
    def test_empty_directory_raises(self, mock_emb, tmp_path):
        from agents.cardiologist import CardiologistAgent

        empty_dir = tmp_path / "empty_kb"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No documents found"):
            CardiologistAgent(folder_path=str(empty_dir))
