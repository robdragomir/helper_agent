"""
Online Search Manager - Infrastructure layer.
Handles web search and source validation for online mode.
"""

from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
import re
from langchain_tavily import TavilySearch

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core import settings

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a web search result."""

    title: str
    url: str
    content: str
    relevance_score: float
    recency_score: float  # 0.0 to 1.0, higher is more recent


class OnlineSearchManager:
    """Manages online search via Tavily and validates source credibility."""

    def __init__(self):
        logger.info(f"Initializing OnlineSearchManager with tavily_api_key={bool(settings.tavily_api_key)}")

        if not settings.tavily_api_key:
            logger.warning("TAVILY_API_KEY not set in settings. Online search will fail.")
        else:
            # Ensure Tavily API key is in environment
            os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

        self.tavily_search = TavilySearch(
            max_results=settings.max_online_search_results,
        )
        logger.info(f"TavilySearch initialized with max_results={settings.max_online_search_results}")

        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
        )
        logger.info(f"ChatGoogleGenerativeAI initialized with model={settings.llm_model}")

    def search(self, query: str) -> List[SearchResult]:
        """
        Search the web for relevant information.
        Searches for both LangGraph and LangChain documentation.
        """
        try:
            all_search_results = []

            # Search for LangGraph results
            langgraph_query = f"{query} python langgraph library"
            logger.info(f"Searching LangGraph with: '{langgraph_query}'")
            langgraph_results = self._search_tavily(langgraph_query)
            all_search_results.extend(langgraph_results)
            logger.info(f"LangGraph search returned {len(langgraph_results)} results")

            # Search for LangChain results
            langchain_query = f"{query} python langchain library"
            logger.info(f"Searching LangChain with: '{langchain_query}'")
            langchain_results = self._search_tavily(langchain_query)
            all_search_results.extend(langchain_results)
            logger.info(f"LangChain search returned {len(langchain_results)} results")

            # Remove duplicates (by URL)
            seen_urls = set()
            deduplicated_results = []
            for result in all_search_results:
                if result.url not in seen_urls:
                    deduplicated_results.append(result)
                    seen_urls.add(result.url)

            logger.info(f"OnlineSearchManager.search() returning {len(deduplicated_results)} deduplicated results")
            return deduplicated_results

        except Exception as e:
            logger.error(f"Error during online search: {e}", exc_info=True)
            return []

    def _search_tavily(self, query: str) -> List[SearchResult]:
        """
        Internal method to search Tavily for a specific query.
        Returns list of SearchResult objects.
        """
        try:
            results = self.tavily_search.invoke(query)
            logger.info(f"Tavily raw response type: {type(results)}")

            search_results = []
            if isinstance(results, str):
                # Parse the string response
                parsed_results = self._parse_tavily_response(results)
            elif isinstance(results, dict):
                # Tavily returns a dict with 'results' key
                parsed_results = results.get("results", [])
                logger.info(f"Extracted results from dict: {len(parsed_results)} items")
            elif isinstance(results, list):
                parsed_results = results
            else:
                parsed_results = []
                logger.warning(f"Unexpected Tavily response type: {type(results)}")

            logger.info(f"Parsed {len(parsed_results)} results from Tavily")

            for result in parsed_results:
                # Prefer raw_content if available, fall back to content
                content = result.get("raw_content") or result.get("content", "")

                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=content,
                    relevance_score=result.get("score", 0.5),
                    recency_score=self._calculate_recency_score(
                        result.get("published_date", "")
                    ),
                )
                search_results.append(search_result)
                logger.debug(f"Added search result: {search_result.title} (content length: {len(content)})")

            return search_results

        except Exception as e:
            logger.error(f"Error during Tavily search: {e}", exc_info=True)
            return []

    def _parse_tavily_response(self, response_str: str) -> List[Dict]:
        """Parse Tavily search response string."""
        try:
            # Try to parse as JSON first
            return json.loads(response_str)
        except json.JSONDecodeError:
            # Fallback: extract basic info from string format
            return self._extract_results_from_string(response_str)

    @staticmethod
    def _extract_results_from_string(response_str: str) -> List[Dict]:
        """Extract results from Tavily string response format."""
        results = []
        # This is a simple parser; Tavily response format may vary
        lines = response_str.split("\n")
        current_result = {}

        for line in lines:
            if "Title:" in line:
                if current_result:
                    results.append(current_result)
                current_result = {"title": line.split("Title:")[-1].strip()}
            elif "URL:" in line:
                current_result["url"] = line.split("URL:")[-1].strip()
            elif "Content:" in line:
                current_result["content"] = line.split("Content:")[-1].strip()

        if current_result:
            results.append(current_result)

        return results

    def _calculate_recency_score(self, published_date: str) -> float:
        """Calculate how recent a source is (0.0 to 1.0, higher is more recent)."""
        if not published_date:
            return 0.5  # Neutral score for unknown dates

        try:

            published = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            now = datetime.now(published.tzinfo) if published.tzinfo else datetime.now()

            # Calculate days old
            days_old = (now - published).days

            # Score: 1.0 for today, 0.5 for 30 days, 0.0 for 90+ days
            if days_old <= 0:
                return 1.0
            elif days_old >= 90:
                return 0.0
            else:
                return 1.0 - (days_old / 90.0)

        except Exception:
            return 0.5  # Default neutral score

    def validate_and_rerank(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """
        Use LLM to validate search results for accuracy and relevance.
        Reranks based on how well results answer the query.
        """
        if not results:
            return []

        # Prepare context for LLM validation
        results_context = "\n\n".join(
            [
                f"Source {i+1}: {result.title}\nURL: {result.url}\n"
                f"Content: {result.content[:500]}...\n"
                f"Relevance: {result.relevance_score}"
                for i, result in enumerate(results)
            ]
        )

        system_message = SystemMessage(
            content=(
                "You are an expert at evaluating the quality and relevance of web search results for Python documentation queries. "
                "\n\n"
                "Evaluate each source based on these criteria:\n"
                "1. ACCURACY: Does the content contain factually correct information about the topic? "
                "   - Official documentation (docs.langchain.com, github.com official repos) = high accuracy\n"
                "   - Peer-reviewed technical articles = high accuracy\n"
                "   - Blog posts or unofficial tutorials = may have inaccuracies\n"
                "2. RELEVANCE: How well does the content answer the specific query?\n"
                "   - Exact match to query topic = high relevance\n"
                "   - Related but tangential information = medium relevance\n"
                "   - General information only = low relevance\n"
                "3. AUTHORITY: Is the source trustworthy and authoritative?\n"
                "   - Official documentation sites = highest authority\n"
                "   - Major tech publications and established blogs = high authority\n"
                "   - Unknown or low-traffic sites = low authority\n"
                "\n"
                "Examples:\n"
                "- 'docs.langchain.com/langgraph/...' discussing StateGraph: HIGH (official, accurate, directly relevant)\n"
                "- 'github.com/langchain-ai/langgraph/...' discussing implementation: HIGH (official source)\n"
                "- 'Medium article on using LangGraph' discussing the same feature: MEDIUM (relevant but unofficial)\n"
                "- 'Random blog about Python async patterns': LOW (not specific to the query)\n"
                "\n"
                "Return a JSON object with scores for each source (0.0-1.0)."
            )
        )

        human_message = HumanMessage(
            content=(
                f"Query: {query}\n\n"
                f"Search Results:\n{results_context}\n\n"
                f"For each source, provide a validation score (0.0-1.0) based on relevance and authority. "
                f"Return as JSON: {{'source_0': 0.8, 'source_1': 0.6, ...}}"
            )
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            response_text = response.content if hasattr(response, "content") else str(response)

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())

                # Apply validation scores
                for i, result in enumerate(results):
                    key = f"source_{i}"
                    if key in scores:
                        result.relevance_score = (
                            result.relevance_score * 0.5 + scores[key] * 0.5
                        )

        except Exception as e:
            print(f"Error validating search results: {e}")

        # Sort by combined score
        for result in results:
            combined_score = (
                result.relevance_score * 0.7 + result.recency_score * 0.3
            )
            result.relevance_score = combined_score

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
