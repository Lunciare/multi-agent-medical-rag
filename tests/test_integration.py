import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def _create_sample_data(tmp_path: Path) -> Path:
    kb_dir = tmp_path / "cardiology"
    kb_dir.mkdir()
    (kb_dir / "sample.txt").write_text(
        "Atrial Fibrillation\n\n"
        "Atrial fibrillation (AFib) is an irregular and often rapid heart rate "
        "that occurs when the two upper chambers of the heart experience chaotic "
        "electrical signals.",
        encoding="utf-8",
    )
    return kb_dir


def _mock_openai_response(text: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.output_text = text
    return mock_resp


class TestOrchestratorConstruction:

    @patch("agents.cardiologist.YandexNativeEmbeddings")
    @patch("agents.endocrinologist.YandexNativeEmbeddings")
    @patch("agents.cardiologist.FAISS")
    @patch("agents.endocrinologist.FAISS")
    def test_constructor_accepts_path(self, mock_endo_faiss, mock_cardio_faiss, mock_endo_emb, mock_cardio_emb, tmp_path):
        from orchestrator import MedicalOrchestrator

        kb = _create_sample_data(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        assert orch.knowledge_base_dir == str(kb)
        assert orch.cardiologist is not None
        assert orch.endocrinologist is not None


class TestRouting:

    @patch("agents.cardiologist.YandexNativeEmbeddings")
    @patch("agents.endocrinologist.YandexNativeEmbeddings")
    @patch("agents.cardiologist.FAISS")
    @patch("agents.endocrinologist.FAISS")
    @patch("orchestrator.client")
    def test_route_returns_cardiologist(self, mock_client, mock_endo_faiss, mock_cardio_faiss, mock_endo_emb, mock_cardio_emb, tmp_path):
        from orchestrator import MedicalOrchestrator

        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="cardiologist"))]
        )

        kb = _create_sample_data(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        result = orch.route("What is atrial fibrillation?")
        assert result == "cardiologist"

    @patch("agents.cardiologist.YandexNativeEmbeddings")
    @patch("agents.endocrinologist.YandexNativeEmbeddings")
    @patch("agents.cardiologist.FAISS")
    @patch("agents.endocrinologist.FAISS")
    @patch("orchestrator.client")
    def test_route_returns_endocrinologist(self, mock_client, mock_endo_faiss, mock_cardio_faiss, mock_endo_emb, mock_cardio_emb, tmp_path):
        from orchestrator import MedicalOrchestrator

        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="endocrinologist"))]
        )

        kb = _create_sample_data(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        result = orch.route("What is type 2 diabetes?")
        assert result == "endocrinologist"

    @patch("agents.cardiologist.YandexNativeEmbeddings")
    @patch("agents.endocrinologist.YandexNativeEmbeddings")
    @patch("agents.cardiologist.FAISS")
    @patch("agents.endocrinologist.FAISS")
    @patch("orchestrator.client")
    def test_route_unknown_specialist(self, mock_client, mock_endo_faiss, mock_cardio_faiss, mock_endo_emb, mock_cardio_emb, tmp_path):
        from orchestrator import MedicalOrchestrator

        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="neurologist"))]
        )

        kb = _create_sample_data(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        answer = orch.answer("What causes migraines?")
        assert "could not determine" in answer.lower()


class TestEndToEndAnswer:

    @patch("agents.cardiologist.YandexNativeEmbeddings")
    @patch("agents.endocrinologist.YandexNativeEmbeddings")
    @patch("agents.cardiologist.FAISS")
    @patch("agents.endocrinologist.FAISS")
    @patch("agents.cardiologist.client")
    @patch("orchestrator.client")
    def test_cardiologist_answer(
        self, mock_orch_client, mock_agent_client, mock_endo_faiss, mock_cardio_faiss, mock_endo_emb, mock_cardio_emb, tmp_path
    ):
        from orchestrator import MedicalOrchestrator

        mock_orch_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="cardiologist"))]
        )

        mock_agent_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Atrial fibrillation is an irregular heart rhythm originating in the atria."))]
        )

        mock_doc = MagicMock()
        mock_doc.page_content = "AFib is a common arrhythmia."
        mock_vs_instance = MagicMock()
        mock_vs_instance.similarity_search_with_score.return_value = [(mock_doc, 0.5)]
        mock_cardio_faiss.load_local.return_value = mock_vs_instance
        mock_cardio_faiss.from_documents.return_value = mock_vs_instance
        mock_endo_faiss.load_local.return_value = mock_vs_instance

        kb = _create_sample_data(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        answer = orch.answer("What is atrial fibrillation?")
        assert len(answer) > 0
        assert "atrial fibrillation" in answer.lower()

    @patch("agents.cardiologist.YandexNativeEmbeddings")
    @patch("agents.endocrinologist.YandexNativeEmbeddings")
    @patch("agents.cardiologist.FAISS")
    @patch("agents.endocrinologist.FAISS")
    @patch("agents.cardiologist.client")
    @patch("orchestrator.client")
    def test_empty_domain_edge_case(
        self, mock_orch_client, mock_agent_client, mock_endo_faiss, mock_cardio_faiss, mock_endo_emb, mock_cardio_emb, tmp_path
    ):
        from orchestrator import MedicalOrchestrator

        mock_orch_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="cardiologist"))]
        )

        mock_agent_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Insufficient evidence in the current knowledge base to address this query."))]
        )

        mock_doc = MagicMock()
        mock_doc.page_content = "Unrelated cake recipe."
        mock_vs_instance = MagicMock()
        # L2 distance 1.8 > 1.2 threshold → chunk gets filtered out
        mock_vs_instance.similarity_search_with_score.return_value = [(mock_doc, 1.8)]
        mock_cardio_faiss.load_local.return_value = mock_vs_instance
        mock_cardio_faiss.from_documents.return_value = mock_vs_instance
        mock_endo_faiss.load_local.return_value = mock_vs_instance

        kb = _create_sample_data(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        answer = orch.answer("How to bake a cake?")
        assert "insufficient evidence" in answer.lower()
