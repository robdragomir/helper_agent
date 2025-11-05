"""
Online search agent implementation.
"""

import logging

from app.core import EvidencePack
from app.infrastructure import OnlineSearchManager
from app.application.interfaces import SearchAgent

logger = logging.getLogger(__name__)


class OnlineSearchAgent(SearchAgent):
    """Searches the web for information."""

    def __init__(self):
        self.search_manager = OnlineSearchManager()

    def search(self, query: str, **kwargs) -> EvidencePack:
        """Search online for information."""
        logger.info(f"OnlineSearchAgent.search() called with query: '{query}'")
        results = self.search_manager.search(query)
        logger.info(f"OnlineSearchManager returned {len(results) if results else 0} results")

        if not results:
            logger.warning(f"No results found from Tavily search for query: '{query}'")
            return EvidencePack(
                mode="online",
                context_text="",
                coverage_confidence=0.0,
                notes="No web search results found",
            )

        logger.info(f"Before validation/reranking: {len(results)} results")
        # Validate and rerank results
        results = self.search_manager.validate_and_rerank(results, query)
        logger.info(f"After validation/reranking: {len(results)} results")

        # Combine top results (with more content for better answers)
        context_parts = []
        sources = []
        seen_urls = set()

        for i, result in enumerate(results[:5]):  # Top 5 results for more context
            # Use full content if available, otherwise show more characters
            content_preview = result.content if len(result.content) < 1500 else result.content[:1500] + "..."
            context_parts.append(
                f"[{i+1}] {result.title}\nSource: {result.url}\n"
                f"Content: {content_preview}\n"
                f"Relevance: {result.relevance_score:.2f}"
            )
            logger.debug(f"Added result {i+1}: {result.title} (score: {result.relevance_score:.2f})")

            # Track unique sources
            if result.url not in seen_urls:
                sources.append({"source": result.url, "title": result.title, "type": "online"})
                seen_urls.add(result.url)
                logger.info(f"Added online source: {result.url}")

        context_text = "\n\n---\n\n".join(context_parts)
        logger.info(f"Context text length: {len(context_text)} characters")

        # Average relevance score as confidence
        avg_confidence = (
            sum([r.relevance_score for r in results]) / len(results)
            if results
            else 0.0
        )
        logger.info(f"Average confidence: {avg_confidence:.2f}")

        return EvidencePack(
            mode="online",
            context_text=context_text,
            coverage_confidence=avg_confidence,
            notes=f"Retrieved {len(results)} web results",
            sources=sources,
        )