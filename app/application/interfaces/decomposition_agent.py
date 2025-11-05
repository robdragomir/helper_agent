"""
Query decomposition agent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict


class DecompositionAgent(ABC):
    """Abstract interface for query decomposition."""

    @abstractmethod
    def decompose(self, query: str) -> Dict:
        """
        Decompose a user query into subquestions.

        Returns a dict with:
        - decomposed_questions: List of question objects with id, question, and requires fields
        - final_question_id: The ID of the final question to return to user
        """
        pass