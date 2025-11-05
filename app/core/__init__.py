"""
Core layer - contains domain models and configuration.
"""

from .models import (
    EvidencePack,
    FinalAnswer,
    KnowledgeBaseSnapshot,
    InferenceTrace,
)
from .config import settings

__all__ = [
    "EvidencePack",
    "FinalAnswer",
    "KnowledgeBaseSnapshot",
    "InferenceTrace",
    "settings",
]