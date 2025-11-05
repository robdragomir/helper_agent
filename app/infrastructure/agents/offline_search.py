"""
Offline search agent implementation.
"""

from typing import Optional
import logging
import math

from app.core import settings, EvidencePack
from app.infrastructure import KnowledgeBaseManager
from app.application.interfaces import SearchAgent

logger = logging.getLogger(__name__)


class OfflineSearchAgent(SearchAgent):
    """Searches the local knowledge base."""

    def __init__(self):
        self.kb_manager = KnowledgeBaseManager()

    def search(self, query: str, top_k: Optional[int] = None, **kwargs) -> EvidencePack:
        """Search offline knowledge base."""
        logger.info(f"OfflineSearchAgent.search() called with query: '{query}', top_k={top_k}")
        results = self.kb_manager.search_offline(query, top_k=top_k)
        logger.info(f"KB search returned {len(results) if results else 0} results")

        if not results:
            logger.warning(f"No results found for query: '{query}'")
            return EvidencePack(
                mode="offline",
                context_text="",
                coverage_confidence=0.0,
                notes="No relevant chunks found in knowledge base",
            )

        # Combine results and extract sources
        context_text = "\n\n---\n\n".join([chunk for chunk, _, _ in results])
        coverage_confidence = self.compute_coverage_confidence(results, settings.similarity_threshold)
        logger.info(f"Coverage confidence: {coverage_confidence}")

        # Extract unique sources from results
        sources = []
        seen_sources = set()
        for _, _, metadata in results:
            source = metadata.get('source', 'unknown')
            if source not in seen_sources:
                sources.append({"source": source, "type": "offline"})
                seen_sources.add(source)
                logger.info(f"Added offline source: {source}")

        return EvidencePack(
            mode="offline",
            context_text=context_text,
            coverage_confidence=coverage_confidence,
            notes=f"Retrieved {len(results)} chunks from KB",
            sources=sources,
        )

    @staticmethod
    def compute_coverage_confidence(results, threshold: float):
        distances = sorted(dist for _, dist, _ in results)

        avg_distance = sum(distances) / len(distances)
        best_distance = distances[0]
        logger.info(f"Average distance: {avg_distance}, best distance: {best_distance}, similarity_threshold: {settings.similarity_threshold}")

        def dist_to_conf(dist):
            # exponential falloff; threshold ~ distance where confidence is ~0.37
            return math.exp(-dist / threshold)

        avg_conf = dist_to_conf(avg_distance)
        best_conf = dist_to_conf(best_distance)

        coverage_confidence = 0.7 * best_conf + 0.3 * avg_conf
        return max(0.0, min(1.0, coverage_confidence))