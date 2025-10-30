"""
Infrastructure layer - contains implementation details for external systems.
Follows onion architecture - manages KB, vector stores, and external API calls.
"""

from .kb_manager import KnowledgeBaseManager, AgenticChunker, VectorStoreManager
from .online_search import OnlineSearchManager

__all__ = [
    "KnowledgeBaseManager",
    "AgenticChunker",
    "VectorStoreManager",
    "OnlineSearchManager",
]