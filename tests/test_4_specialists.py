"""Pytest coverage for the 4-specialist registry (Stage 36-39).

The pre-Stage-36 test suite (test_integration.py) only exercises the
historical cardiologist + endocrinologist pair. This file adds the missing
coverage for the two new specialists — gastroenterologist and infectionist —
and the refusal gate's behaviour on each.

All tests use the same MagicMock/patch strategy as test_integration.py:
the FAISS vectorstore and Yandex chat-completions client are mocked so the
tests run offline (no API key required).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _create_minimal_kb(tmp_path: Path) -> Path:
    """A throwaway knowledge-base dir; the FAISS mock means the directory's
    contents don't really matter, but the orchestrator's constructor scans
    the registry's `folder_path` so we set up one cardiology stub for the
    common `DEFAULT_KNOWLEDGE_BASE_DIR` argument."""
    kb_dir = tmp_path / "cardiology"
    kb_dir.mkdir()
    (kb_dir / "sample.txt").write_text(
        "Atrial Fibrillation\n\n"
        "Atrial fibrillation (AFib) is an irregular and often rapid heart rate.\n",
        encoding="utf-8",
    )
    return kb_dir


REGISTRY_KEYS = ("cardiologist", "endocrinologist",
                 "gastroenterologist", "infectionist")


class TestFourSpecialistConstruction:
    """The orchestrator must instantiate one SpecialistAgent per registry key."""

    @patch("agents.specialist.YandexNativeEmbeddings")
    @patch("agents.specialist.FAISS")
    def test_orchestrator_constructs_all_4_specialists(
        self, mock_faiss, mock_emb, tmp_path
    ):
        from orchestrator import MedicalOrchestrator

        kb = _create_minimal_kb(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        # Every registry key must have a constructed agent.
        for key in REGISTRY_KEYS:
            assert key in orch.agents, f"missing agent: {key!r}"
            assert orch.agents[key] is not None, f"agent {key!r} is None"

        # And no unexpected extras.
        assert set(orch.agents.keys()) == set(REGISTRY_KEYS), (
            f"unexpected agents in orchestrator: {set(orch.agents) - set(REGISTRY_KEYS)}"
        )

    @patch("agents.specialist.YandexNativeEmbeddings")
    @patch("agents.specialist.FAISS")
    def test_allowed_specialists_includes_all_4(self, mock_faiss, mock_emb, tmp_path):
        from orchestrator import MedicalOrchestrator

        kb = _create_minimal_kb(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        # `allowed_specialists` is the strict allow-list used by
        # `_parse_router_output` to reject any LLM-emitted specialty outside
        # the registry.
        assert set(orch.allowed_specialists) == set(REGISTRY_KEYS)


class TestFourSpecialistRouting:
    """Routing returns the correct specialist when the LLM emits it."""

    @patch("agents.specialist.YandexNativeEmbeddings")
    @patch("agents.specialist.FAISS")
    @patch("orchestrator.client")
    def test_route_returns_gastroenterologist(
        self, mock_client, mock_faiss, mock_emb, tmp_path
    ):
        from orchestrator import MedicalOrchestrator

        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"specialist": "gastroenterologist"}'
            ))]
        )

        kb = _create_minimal_kb(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        result = orch.route("A 32yo with bloody diarrhoea and oral ulcers — diagnosis?")
        assert result == "gastroenterologist"

    @patch("agents.specialist.YandexNativeEmbeddings")
    @patch("agents.specialist.FAISS")
    @patch("orchestrator.client")
    def test_route_returns_infectionist(
        self, mock_client, mock_faiss, mock_emb, tmp_path
    ):
        from orchestrator import MedicalOrchestrator

        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"specialist": "infectionist"}'
            ))]
        )

        kb = _create_minimal_kb(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        result = orch.route(
            "68yo with fever, cough, right lower lobe consolidation — empirical antibiotics?"
        )
        assert result == "infectionist"


class TestFourSpecialistAnswerFlow:
    """End-to-end mock answer flow for each of the two new specialists.

    The refusal gate is monkey-patched off so the mocked retrieval drives
    the agent's generation. Each test verifies that the orchestrator routes
    to the right specialist, the specialist answers, and the returned name
    matches the registry's `name` field.
    """

    @patch("agents.specialist.YandexNativeEmbeddings")
    @patch("agents.specialist.FAISS")
    @patch("agents.specialist.client")
    @patch("orchestrator.client")
    def test_gastro_agent_answer_with_mock(
        self, mock_orch_client, mock_agent_client, mock_faiss, mock_emb, tmp_path
    ):
        from orchestrator import MedicalOrchestrator

        # Router emits the gastro routing decision.
        mock_orch_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"specialist": "gastroenterologist"}'
            ))]
        )

        # Agent's generation call returns a faithful-looking answer.
        mock_agent_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="**Clinical Summary**\nUlcerative colitis flare.\n\n"
                        "**Evidence-Based Insights**\nThe context supports UC.\n\n"
                        "**Limitations**\nCorpus snippet only.\n\n"
                        "This output is for informational use…"
            ))]
        )

        # Mocked retrieval surfaces a single UC chunk inside MAX_L2_DISTANCE.
        mock_doc = MagicMock()
        mock_doc.page_content = "Ulcerative colitis (UC) is an idiopathic IBD."
        mock_doc.metadata = {"source_file": "uc.txt", "doc_name": "uc-stub",
                              "category": "Guidelines"}
        mock_vs_instance = MagicMock()
        mock_vs_instance.similarity_search_with_score.return_value = [(mock_doc, 0.5)]
        mock_faiss.load_local.return_value = mock_vs_instance
        mock_faiss.from_documents.return_value = mock_vs_instance

        kb = _create_minimal_kb(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        # Disable the refusal gate uniformly across all agents.
        for agent in orch.agents.values():
            agent._refusal_gate = type(
                "NoOpGate", (), {"refuse": lambda self, q: False}
            )()

        name, answer, _evidence = orch.answer(
            "A 32-year-old with UC flare and bloody diarrhoea — induction strategy?"
        )
        assert name == "Gastroenterologist"
        assert len(answer) > 0
        assert "ulcerative colitis" in answer.lower()

    @patch("agents.specialist.YandexNativeEmbeddings")
    @patch("agents.specialist.FAISS")
    @patch("agents.specialist.client")
    @patch("orchestrator.client")
    def test_infect_agent_answer_with_mock(
        self, mock_orch_client, mock_agent_client, mock_faiss, mock_emb, tmp_path
    ):
        from orchestrator import MedicalOrchestrator

        mock_orch_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"specialist": "infectionist"}'
            ))]
        )
        mock_agent_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="**Clinical Summary**\nCommunity-acquired pneumonia.\n\n"
                        "**Evidence-Based Insights**\nContext supports CAP.\n\n"
                        "**Limitations**\nCorpus snippet only.\n\n"
                        "This output is for informational use…"
            ))]
        )

        mock_doc = MagicMock()
        mock_doc.page_content = "Community-acquired pneumonia is treated empirically."
        mock_doc.metadata = {"source_file": "cap.txt", "doc_name": "cap-stub",
                              "category": "Guidelines"}
        mock_vs_instance = MagicMock()
        mock_vs_instance.similarity_search_with_score.return_value = [(mock_doc, 0.4)]
        mock_faiss.load_local.return_value = mock_vs_instance
        mock_faiss.from_documents.return_value = mock_vs_instance

        kb = _create_minimal_kb(tmp_path)
        orch = MedicalOrchestrator(knowledge_base_dir=str(kb))

        for agent in orch.agents.values():
            agent._refusal_gate = type(
                "NoOpGate", (), {"refuse": lambda self, q: False}
            )()

        name, answer, _evidence = orch.answer(
            "68yo with fever, cough, RLL consolidation — empirical antibiotics?"
        )
        assert name == "Infectionist"
        assert len(answer) > 0
        assert "pneumonia" in answer.lower()


class TestFourSpecialistRefusalGate:
    """The L2 refusal gate must engage on a clear out-of-scope query for each
    specialist. Strategy: stub the specialty's vectorstore so retrieval returns
    zero chunks within MAX_L2_DISTANCE (and the per-specialty L2 threshold).
    The agent's `answer()` then short-circuits to the REFUSAL_RESPONSE string.

    These tests guard against a regression where adding a new specialist breaks
    the gate by skipping the `refuse()` check on its agent.
    """

    @staticmethod
    def _setup_orchestrator_for_gate_test(mock_faiss, far_l2: float, tmp_path: Path):
        """Build an orchestrator whose every agent's retrieval is mocked to
        return chunks all at `far_l2` — above any reasonable threshold so
        the gate fires."""
        from orchestrator import MedicalOrchestrator

        # All four agents share the same mocked vectorstore — `far_l2` is
        # the L2 distance of every retrieved chunk.
        far_doc = MagicMock()
        far_doc.page_content = "irrelevant content"
        far_doc.metadata = {"source_file": "irrelevant.txt",
                             "doc_name": "irrelevant", "category": "Articles"}
        mock_vs_instance = MagicMock()
        mock_vs_instance.similarity_search_with_score.return_value = [
            (far_doc, far_l2)
        ] * 5
        mock_faiss.load_local.return_value = mock_vs_instance
        mock_faiss.from_documents.return_value = mock_vs_instance

        kb = _create_minimal_kb(tmp_path)
        return MedicalOrchestrator(knowledge_base_dir=str(kb))

    @patch("agents.specialist.YandexNativeEmbeddings")
    @patch("agents.specialist.FAISS")
    def test_refusal_gate_engages_for_cardiologist(self, mock_faiss, mock_emb, tmp_path):
        from agents.specialist import REFUSAL_RESPONSE

        # 1.5 is well above both MAX_L2_DISTANCE (1.2) and L2_REJECT_MIN
        # (currently 1.02 post-Stage-39 re-tune) — gate must fire.
        orch = self._setup_orchestrator_for_gate_test(mock_faiss, far_l2=1.5,
                                                      tmp_path=tmp_path)
        with patch("orchestrator.client") as mock_orch_client:
            mock_orch_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content='{"specialist": "cardiologist"}'
                ))]
            )
            _name, answer, _ev = orch.answer(
                "A philosophical question with no cardiac content whatsoever."
            )
            assert REFUSAL_RESPONSE in answer

    @patch("agents.specialist.YandexNativeEmbeddings")
    @patch("agents.specialist.FAISS")
    def test_refusal_gate_engages_for_endocrinologist(self, mock_faiss, mock_emb, tmp_path):
        from agents.specialist import REFUSAL_RESPONSE
        orch = self._setup_orchestrator_for_gate_test(mock_faiss, far_l2=1.5,
                                                      tmp_path=tmp_path)
        with patch("orchestrator.client") as mock_orch_client:
            mock_orch_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content='{"specialist": "endocrinologist"}'
                ))]
            )
            _name, answer, _ev = orch.answer(
                "A purely abstract question about epistemology."
            )
            assert REFUSAL_RESPONSE in answer

    @patch("agents.specialist.YandexNativeEmbeddings")
    @patch("agents.specialist.FAISS")
    def test_refusal_gate_engages_for_gastroenterologist(self, mock_faiss, mock_emb, tmp_path):
        from agents.specialist import REFUSAL_RESPONSE
        orch = self._setup_orchestrator_for_gate_test(mock_faiss, far_l2=1.5,
                                                      tmp_path=tmp_path)
        with patch("orchestrator.client") as mock_orch_client:
            mock_orch_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content='{"specialist": "gastroenterologist"}'
                ))]
            )
            _name, answer, _ev = orch.answer(
                "Tell me about the philosophy of language — no GI content here."
            )
            assert REFUSAL_RESPONSE in answer

    @patch("agents.specialist.YandexNativeEmbeddings")
    @patch("agents.specialist.FAISS")
    def test_refusal_gate_engages_for_infectionist(self, mock_faiss, mock_emb, tmp_path):
        from agents.specialist import REFUSAL_RESPONSE
        orch = self._setup_orchestrator_for_gate_test(mock_faiss, far_l2=1.5,
                                                      tmp_path=tmp_path)
        with patch("orchestrator.client") as mock_orch_client:
            mock_orch_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content='{"specialist": "infectionist"}'
                ))]
            )
            _name, answer, _ev = orch.answer(
                "Tell me about Shakespeare — completely outside infectious disease."
            )
            assert REFUSAL_RESPONSE in answer
