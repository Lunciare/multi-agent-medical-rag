from abc import ABC, abstractmethod
from typing import List, Tuple
from langchain_core.documents import Document


class BaseMedicalAgent(ABC):
    """Abstract base class for all specialist medical RAG agents.

    Concrete subclasses must implement embed_query, retrieve, and answer.
    The answer() method must return (specialist_name: str, response: str, evidence: str).
    """

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
