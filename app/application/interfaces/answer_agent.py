"""
Answer generation agent interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from app.core import EvidencePack, FinalAnswer


class AnswerAgent(ABC):
    """Abstract interface for answer generation."""

    @abstractmethod
    def generate(
        self,
        query: str,
        evidence_packs: List[EvidencePack],
        conversation_history: Optional[List[Dict]] = None,
        dependent_question_context: Optional[str] = None,
    ) -> FinalAnswer:
        """
        Generate final answer from evidence.

        Args:
            query: The question to answer
            evidence_packs: Evidence from search results
            conversation_history: Previous conversation context
            dependent_question_context: Context from dependent subquestions

        Returns:
            FinalAnswer with the response and citations
        """
        pass