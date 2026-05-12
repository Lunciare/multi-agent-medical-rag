import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def _build_orchestrator(tmp_path):
    from orchestrator import MedicalOrchestrator

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "sample.txt").write_text("Sample cardiology text.", encoding="utf-8")
    return MedicalOrchestrator(knowledge_base_dir=str(kb))


class TestEmergencyDetection:

    EMERGENCY_QUERIES = [
        "I'm having a heart attack right now",
        "I can't breathe and my chest hurts",
        "My father lost consciousness, call 911",
        "She is choking and turning blue",
        "I think I'm having a stroke",
        "Patient in cardiac arrest",
        "There is severe bleeding from the wound",
    ]

    @pytest.mark.parametrize("query", EMERGENCY_QUERIES)
    @patch("agents.cardiologist.OpenAIEmbeddings")
    @patch("agents.cardiologist.FAISS")
    def test_emergency_query_is_blocked(self, mock_faiss, mock_emb, query, tmp_path):
        orch = _build_orchestrator(tmp_path)
        result = orch.safety_check(query)
        assert result is not None
        assert "emergency" in result.lower()

    @patch("agents.cardiologist.OpenAIEmbeddings")
    @patch("agents.cardiologist.FAISS")
    def test_emergency_does_not_reach_specialist(self, mock_faiss, mock_emb, tmp_path):
        orch = _build_orchestrator(tmp_path)
        answer = orch.answer("I'm having a heart attack right now")
        assert "emergency" in answer.lower()
        assert "911" in answer


class TestTreatmentDetection:

    TREATMENT_QUERIES = [
        "Prescribe me something for high blood pressure",
        "What medication should I take for my arrhythmia?",
        "Give me a dosage for metoprolol",
        "Write me a prescription for statins",
    ]

    @pytest.mark.parametrize("query", TREATMENT_QUERIES)
    @patch("agents.cardiologist.OpenAIEmbeddings")
    @patch("agents.cardiologist.FAISS")
    def test_treatment_query_is_blocked(self, mock_faiss, mock_emb, query, tmp_path):
        orch = _build_orchestrator(tmp_path)
        result = orch.safety_check(query)
        assert result is not None
        assert "prescribe" in result.lower() or "dosage" in result.lower()


class TestSafeQueryPassthrough:

    SAFE_QUERIES = [
        "What is atrial fibrillation?",
        "Explain the difference between systolic and diastolic pressure",
        "What are the symptoms of mitral valve prolapse?",
        "How does an ECG work?",
    ]

    @pytest.mark.parametrize("query", SAFE_QUERIES)
    @patch("agents.cardiologist.OpenAIEmbeddings")
    @patch("agents.cardiologist.FAISS")
    def test_safe_query_returns_none(self, mock_faiss, mock_emb, query, tmp_path):
        orch = _build_orchestrator(tmp_path)
        result = orch.safety_check(query)
        assert result is None
