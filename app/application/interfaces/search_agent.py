"""
Search agent interface.
"""

from abc import ABC, abstractmethod
from app.core import EvidencePack


class SearchAgent(ABC):
    """Abstract interface for search agents (offline and online)."""

    @abstractmethod
    def search(self, query: str, **kwargs) -> EvidencePack:
        """
        Search for information relevant to the query.

        Args:
            query: The question to search for
            **kwargs: Additional parameters specific to the implementation

        Returns:
            EvidencePack with the search results and metadata
        """
        pass