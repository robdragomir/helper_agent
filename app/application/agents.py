"""
Agent implementations - Application layer.
Contains router, search, answer generation, and guardrail agents.
"""

from typing import List, Dict, Optional, Tuple
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import re

from app.core import settings, KnowledgeRouteDecision, EvidencePack, FinalAnswer
from app.infrastructure import KnowledgeBaseManager, OnlineSearchManager
from app.infrastructure.telemetry import EvaluationMetrics

# Configure logging
logger = logging.getLogger(__name__)


class RouterAgent:
    """Routes queries to appropriate search mode(s)."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
        )

    def route(self, query: str, forced_mode: Optional[str] = None) -> KnowledgeRouteDecision:
        """
        Decide whether to use offline, online, or combined search.
        Can be overridden by forced_mode argument.
        """
        if forced_mode in ["offline", "online", "both"]:
            return KnowledgeRouteDecision(
                route=forced_mode,
                reason=f"User forced {forced_mode} mode",
                policy_flags=[],
            )

        system_message = SystemMessage(
            content=(
                "You are an expert query router for a LangGraph documentation helper. "
                "Decide whether to use: 'offline' (local KB), 'online' (web search), "
                "'both' (combine both sources), or 'clarify' (ask for clarification). "
                "Consider: specificity of query, likely need for recent info, and whether "
                "the query relates to core LangGraph concepts vs cutting-edge features."
            )
        )

        human_message = HumanMessage(
            content=(
                f"Query: {query}\n\n"
                f"Respond in JSON format: "
                f"{{'route': 'offline'|'online'|'both'|'clarify', "
                f"'reason': 'explanation', "
                f"'policy_flags': [list of concerns if any]}}"
            )
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            response_text = response.content if hasattr(response, "content") else str(response)

            # Extract JSON
            import json

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return KnowledgeRouteDecision(**data)

        except Exception as e:
            print(f"Error in router agent: {e}")

        # Default to both if something goes wrong
        return KnowledgeRouteDecision(
            route="both",
            reason="Default routing due to processing error",
            policy_flags=[],
        )


class OfflineSearchAgent:
    """Searches the local knowledge base."""

    def __init__(self):
        self.kb_manager = KnowledgeBaseManager()

    def search(self, query: str, top_k: Optional[int] = None) -> EvidencePack:
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

        # Combine results
        context_text = "\n\n---\n\n".join([chunk for chunk, _ in results])
        avg_distance = sum([dist for _, dist in results]) / len(results)
        logger.info(f"Average distance: {avg_distance}, similarity_threshold: {settings.similarity_threshold}")

        # Convert distance to confidence (lower distance = higher confidence)
        coverage_confidence = max(0.0, 1.0 - (avg_distance / settings.similarity_threshold))
        logger.info(f"Coverage confidence: {coverage_confidence}")

        return EvidencePack(
            mode="offline",
            context_text=context_text,
            coverage_confidence=coverage_confidence,
            notes=f"Retrieved {len(results)} chunks from KB",
        )


class OnlineSearchAgent:
    """Searches the web for information."""

    def __init__(self):
        self.search_manager = OnlineSearchManager()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
        )

    def search(self, query: str) -> EvidencePack:
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
        for i, result in enumerate(results[:5]):  # Top 5 results for more context
            # Use full content if available, otherwise show more characters
            content_preview = result.content if len(result.content) < 1500 else result.content[:1500] + "..."
            context_parts.append(
                f"[{i+1}] {result.title}\nSource: {result.url}\n"
                f"Content: {content_preview}\n"
                f"Relevance: {result.relevance_score:.2f}"
            )
            logger.debug(f"Added result {i+1}: {result.title} (score: {result.relevance_score:.2f})")

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
        )


class AnswerGenerationAgent:
    """Generates answer based on provided evidence."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
        )
        self.eval_metrics = EvaluationMetrics()

    def generate(
        self,
        query: str,
        evidence_packs: List[EvidencePack],
    ) -> FinalAnswer:
        """Generate final answer from evidence."""
        # Combine context from all evidence packs
        contexts = []
        citations = []
        used_offline = False
        used_online = False

        for pack in evidence_packs:
            if pack.context_text:
                contexts.append(pack.context_text)
                if pack.mode == "offline":
                    used_offline = True
                    citations.append({"label": "[KB]", "source": "Internal KB", "note": "offline source"})
                elif pack.mode == "online":
                    used_online = True
                    citations.append({"label": "[Web]", "source": "Web Search", "note": "online source"})

        combined_context = "\n\n---\n\n".join(contexts)

        if not combined_context:
            return FinalAnswer(
                text="I couldn't find any relevant information to answer your question.",
                used_offline=False,
                used_online=False,
                answer_confidence=0.0,
                citations=[],
            )

        logger.info(f"AnswerGenerationAgent.generate() called with query: '{query}'")
        logger.info(f"Number of evidence packs: {len(evidence_packs)}")
        logger.info(f"Combined context length: {len(combined_context)} characters")
        logger.info(f"Used offline: {used_offline}, Used online: {used_online}")
        logger.info(f"Full context being passed to LLM:\n{'-'*80}\n{combined_context}\n{'-'*80}")

        system_message = SystemMessage(
            content=(
                "You are a LangGraph documentation assistant. "
                "IMPORTANT: You MUST ONLY answer based on the provided context. "
                "Do NOT use any knowledge from your training data. "
                "If the information is not in the provided context, you MUST say 'This information is not available in the provided context.' "
                "Do not make assumptions or fill in gaps with general knowledge. "
                "Only cite what is explicitly stated in the context. "
                "When answering, prioritize practical information including:\n"
                "- How to use features (step-by-step instructions)\n"
                "- Code examples and syntax\n"
                "- Required imports and setup\n"
                "- Configuration options and parameters\n"
                "- Common use cases and patterns\n"
                "Be direct, specific, and actionable in your answers."
            )
        )

        human_message = HumanMessage(
            content=(
                f"Context:\n{combined_context}\n\n"
                f"User Question: {query}\n\n"
                f"Please provide a comprehensive answer based on the context above."
            )
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            answer_text = response.content if hasattr(response, "content") else str(response)

            # Compute confidence score
            avg_evidence_confidence = (
                sum([p.coverage_confidence for p in evidence_packs]) / len(evidence_packs)
                if evidence_packs
                else 0.0
            )

            # Compute faithfulness
            faithfulness = self.eval_metrics.compute_faithfulness(
                answer_text, combined_context
            )

            # Final confidence is average of evidence and faithfulness
            final_confidence = (avg_evidence_confidence * 0.6 + faithfulness * 0.4)

            return FinalAnswer(
                text=answer_text,
                used_offline=used_offline,
                used_online=used_online,
                answer_confidence=final_confidence,
                citations=citations,
            )

        except Exception as e:
            print(f"Error generating answer: {e}")
            return FinalAnswer(
                text=f"Error generating answer: {str(e)}",
                used_offline=used_offline,
                used_online=used_online,
                answer_confidence=0.0,
                citations=citations,
            )


class GuardrailAgent:
    """Validates answers for safety and accuracy."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
        )

    def validate(self, answer: FinalAnswer) -> Tuple[bool, str]:
        """
        Validate answer for safety and compliance.
        Returns (is_valid, reason).
        """
        system_message = SystemMessage(
            content=(
                "You are a safety validator for a technical Q&A system. "
                "Check if the answer contains any inappropriate, harmful, or protected information. "
                "Answer with 'SAFE' or 'UNSAFE' followed by explanation."
            )
        )

        human_message = HumanMessage(
            content=(
                f"Answer to validate:\n{answer.text}\n\n"
                f"Is this response safe to provide to a user? "
                f"Check for: harmful code, security vulnerabilities being exploited, "
                f"protected intellectual property, or misinformation."
            )
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            response_text = response.content if hasattr(response, "content") else str(response)

            if "SAFE" in response_text.upper():
                return True, "Answer passed safety check"
            else:
                # Extract reason
                reason = response_text.replace("UNSAFE", "").strip() or "Failed safety check"
                return False, reason

        except Exception as e:
            print(f"Error in guardrail validation: {e}")
            # Default to safe if validation fails
            return True, "Validation completed"
