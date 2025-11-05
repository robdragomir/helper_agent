"""
Guardrail agent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class GuardrailAgent(ABC):
    """Abstract interface for query validation and safety checking."""

    @abstractmethod
    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate user query for safety and topic relevance before processing.

        Args:
            query: The user's input query

        Returns:
            Dict with:
            - is_safe: bool - whether the content is safe and appropriate
            - is_in_scope: bool - whether the query is relevant to LangChain/LangGraph/AI engineering
            - decision: str - "allow", "reject", or "block"
            - reason: str - short explanation
        """
        pass