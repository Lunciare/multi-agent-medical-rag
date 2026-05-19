from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from langchain_core.documents import Document


class BaseMedicalAgent(ABC):
    """Abstract base class for all specialist medical RAG agents.

    Concrete subclasses must implement embed_query, retrieve, and answer.
    The answer() method must return (specialist_name: str, response: str, evidence: str).
    """

    # Concrete subclasses populate this from refusal_gate.RefusalGate. Optional —
    # if None, refuse() always returns False (backward-compat).
    refusal_gate: Optional[object] = None

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a clinical query using the domain embedding model."""
        ...

    @abstractmethod
    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve top-K relevant (chunk, L2 distance) pairs from the domain FAISS index.

        Returns pairs filtered by the domain MAX_L2_DISTANCE threshold so that
        callers can render confidence without re-running the search.
        """
        ...

    @abstractmethod
    def answer(self, question: str) -> Tuple[str, str, str]:
        """Return (specialist_name, clinical_response, evidence_string)."""
        ...

    def refuse(self, query: str) -> bool:
        """Return True if the agent's refusal gate marks the query out-of-scope.

        Default implementation delegates to `self.refusal_gate.refuse(query)` if
        one is configured; otherwise returns False. Callers in `answer()` should
        invoke this *before* the LLM call.
        """
        if self.refusal_gate is None:
            return False
        return bool(self.refusal_gate.refuse(query))
