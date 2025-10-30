"""
Online Search Manager - Infrastructure layer.
Handles web search and source validation for online mode.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import logging

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
        import os
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
        """Search the web for relevant information."""
        try:
            results = self.tavily_search.invoke(query)

            search_results = []
            if isinstance(results, str):
                # Parse the string response
                parsed_results = self._parse_tavily_response(results)
            elif isinstance(results, list):
                parsed_results = results
            else:
                parsed_results = []

            for result in parsed_results:
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    relevance_score=result.get("score", 0.5),
                    recency_score=self._calculate_recency_score(
                        result.get("published_date", "")
                    ),
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            print(f"Error during online search: {e}")
            return []

    def _parse_tavily_response(self, response_str: str) -> List[Dict]:
        """Parse Tavily search response string."""
        try:
            # Try to parse as JSON first
            return json.loads(response_str)
        except json.JSONDecodeError:
            # Fallback: extract basic info from string format
            return self._extract_results_from_string(response_str)

    def _extract_results_from_string(self, response_str: str) -> List[Dict]:
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
            # Try to parse the date
            from datetime import datetime

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
                "You are an expert at evaluating the quality and relevance of web search results. "
                "Assess each source based on: accuracy, relevance to the query, and authority. "
                "Return a JSON object with scores for each source."
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

            # Extract JSON from response
            import json
            import re

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

    def check_source_agreement(
        self, results: List[SearchResult], key_claim: str
    ) -> float:
        """
        Check how many sources agree on a key claim.
        Returns agreement ratio (0.0 to 1.0).
        """
        if not results:
            return 0.0

        agreeing_count = 0
        system_message = SystemMessage(
            content="You are an expert at determining if text content supports a specific claim. "
            "Answer only with 'yes' or 'no'."
        )

        for result in results:
            human_message = HumanMessage(
                content=(
                    f"Does this content support the claim: '{key_claim}'?\n\n"
                    f"Content: {result.content[:1000]}"
                )
            )

            try:
                response = self.llm.invoke([system_message, human_message])
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )

                if "yes" in response_text.lower():
                    agreeing_count += 1

            except Exception:
                pass

        return agreeing_count / len(results) if results else 0.0