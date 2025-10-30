"""
Core layer - contains domain models and configuration.
Follows onion architecture - innermost layer.
"""

from .models import (
    KnowledgeRouteDecision,
    EvidencePack,
    FinalAnswer,
    KnowledgeBaseSnapshot,
    InferenceTrace,
)
from .config import settings

__all__ = [
    "KnowledgeRouteDecision",
    "EvidencePack",
    "FinalAnswer",
    "KnowledgeBaseSnapshot",
    "InferenceTrace",
    "settings",
]