import contextlib
import pytest
from unittest.mock import patch, MagicMock
import openai


@pytest.fixture(autouse=True)
def _mock_agent_deps():
    targets = [
        "agents.specialist.YandexNativeEmbeddings",
        "agents.specialist.FAISS",
    ]
    with contextlib.ExitStack() as stack:
        for target in targets:
            stack.enter_context(patch(target))
        yield


def _build_orchestrator(tmp_path):
    from orchestrator import MedicalOrchestrator

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "sample.txt").write_text("Sample cardiology content.", encoding="utf-8")
    return MedicalOrchestrator(knowledge_base_dir=str(kb))


class TestInputValidation:

    @pytest.mark.parametrize("query", ["", "   ", None])
    def test_empty_query_rejected(self, query, tmp_path):
        orch = _build_orchestrator(tmp_path)
        _specialist, response, _evidence = orch.answer(query)
        assert "valid" in response.lower()

    def test_too_short_query_rejected(self, tmp_path):
        orch = _build_orchestrator(tmp_path)
        _specialist, response, _evidence = orch.answer("hi")
        assert "short" in response.lower()


class TestAPIFailureHandling:

    @patch("orchestrator.client")
    def test_auth_error_handled(self, mock_client, tmp_path):
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body=None,
        )
        orch = _build_orchestrator(tmp_path)
        _specialist, response, _evidence = orch.answer("What is atrial fibrillation?")
        assert "authentication" in response.lower()

    @patch("orchestrator.client")
    def test_rate_limit_handled(self, mock_client, tmp_path):
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )
        orch = _build_orchestrator(tmp_path)
        _specialist, response, _evidence = orch.answer("What is atrial fibrillation?")
        assert "rate limit" in response.lower()

    @patch("orchestrator.client")
    def test_connection_error_handled(self, mock_client, tmp_path):
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        orch = _build_orchestrator(tmp_path)
        _specialist, response, _evidence = orch.answer("What is atrial fibrillation?")
        assert "connect" in response.lower()


class TestDataDirectoryErrors:

    def test_missing_directory_raises(self):
        from agents import SpecialistAgent

        with pytest.raises(FileNotFoundError):
            SpecialistAgent(
                name="Cardiologist",
                folder_path="/nonexistent/path",
                role_prompt="test",
                domain_scope="test",
            )

    def test_empty_directory_raises(self, tmp_path):
        from agents import SpecialistAgent

        empty_dir = tmp_path / "empty_kb"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No documents found"):
            SpecialistAgent(
                name="Cardiologist",
                folder_path=str(empty_dir),
                role_prompt="test",
                domain_scope="test",
            )
