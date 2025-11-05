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
from .logging_config import configure_logging, get_logger

__all__ = [
    "EvidencePack",
    "FinalAnswer",
    "KnowledgeBaseSnapshot",
    "InferenceTrace",
    "settings",
    "configure_logging",
    "get_logger",
]