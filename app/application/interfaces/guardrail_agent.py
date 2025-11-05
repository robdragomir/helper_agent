"""
Guardrail agent interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from app.core import FinalAnswer


class GuardrailAgent(ABC):
    """Abstract interface for answer validation and safety checking."""

    @abstractmethod
    def validate(self, answer: FinalAnswer) -> Tuple[bool, str]:
        """
        Validate answer for safety and compliance.

        Args:
            answer: The answer to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        pass