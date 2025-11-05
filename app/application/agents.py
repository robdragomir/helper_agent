"""
Agent implementations - Application layer.
Contains query decomposition, search, answer generation, and guardrail agents.
"""

from typing import List, Dict, Optional, Tuple
import logging
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core import settings, EvidencePack, FinalAnswer
from app.infrastructure import KnowledgeBaseManager, OnlineSearchManager
from app.infrastructure.telemetry import EvaluationMetrics
import math


# Configure logging
logger = logging.getLogger(__name__)


class QueryDecompositionAgent:
    """Breaks down complex user queries into simpler subquestions."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
        )

    def decompose(self, query: str) -> Dict:
        """
        Decompose a user query into subquestions.

        Returns a dict with:
        - decomposed_questions: List of question objects with id, question, and requires fields
        - final_question_id: The ID of the final question to return to user
        """
        logger.info(f"QueryDecompositionAgent.decompose() called with query: '{query}'")

        system_message = SystemMessage(
            content=(
                "You are a query decomposition planner. "
                "Your job is to break a user question into a small set of simpler subquestions and to specify dependencies between them.\n\n"
                "Each subquestion:\n"
                "- Must be answerable from documentation or web search.\n"
                "- Should be clear, explicit, and avoid pronouns like \"it\" or \"they\".\n"
                "- Should focus on a single topic or step.\n\n"
                "Dependencies:\n"
                "- Some subquestions may require the answers to earlier subquestions.\n"
                "- Use a requires field containing a list of previous subquestion IDs whose answers are needed.\n"
                "- If a subquestion does not depend on any previous answer, use an empty list [].\n"
                "- Subquestions must be ordered so that any ID in requires appears earlier in the list.\n\n"
                "Final question:\n"
                "- The final subquestion should correspond to the user's original goal.\n"
                "- You must specify final_question_id as the ID of the subquestion whose answer should be returned to the user as the final answer.\n\n"
                "Output only valid JSON with this exact schema:\n"
                "{\n"
                "  \"decomposed_questions\": [\n"
                "    {\"id\": \"q1\", \"question\": \"string\", \"requires\": [\"q_id_1\", \"q_id_2\"]}\n"
                "  ],\n"
                "  \"final_question_id\": \"qX\"\n"
                "}\n\n"
                "Do not include any comments or extra fields."
            )
        )

        human_message = HumanMessage(
            content=f"User query: {query}\n\nDecompose this query into subquestions and provide the JSON response."
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            response_text = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response
            logger.info(f"Raw LLM response: {response_text[:500]}")

            # Try to parse JSON directly
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block or other wrapping
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: treat as single question
                    logger.warning(f"Could not parse decomposition response, treating query as single question")
                    result = {
                        "decomposed_questions": [
                            {"id": "q1", "question": query, "requires": []}
                        ],
                        "final_question_id": "q1"
                    }

            # Validate result structure
            if "decomposed_questions" not in result or "final_question_id" not in result:
                logger.warning("Invalid decomposition response structure, treating query as single question")
                result = {
                    "decomposed_questions": [
                        {"id": "q1", "question": query, "requires": []}
                    ],
                    "final_question_id": "q1"
                }

            logger.info(f"Decomposed into {len(result['decomposed_questions'])} subquestions")
            for q in result['decomposed_questions']:
                logger.info(f"  {q['id']}: {q['question']}")
                if q.get('requires'):
                    logger.info(f"    requires: {q['requires']}")

            return result

        except Exception as e:
            logger.error(f"Error decomposing query: {e}", exc_info=True)
            # Fallback: treat as single question
            return {
                "decomposed_questions": [
                    {"id": "q1", "question": query, "requires": []}
                ],
                "final_question_id": "q1"
            }


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
        conversation_history: Optional[List[Dict]] = None,
        dependent_question_context: Optional[str] = None,
    ) -> FinalAnswer:
        """
        Generate final answer from evidence.

        Args:
            query: The question to answer
            evidence_packs: Evidence from search results
            conversation_history: Previous conversation context
            dependent_question_context: Context from dependent subquestions (takes priority)
        """
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
                    # Use actual sources from pack
                    for i, source_info in enumerate(pack.sources):
                        label = f"[{i+1}]"
                        citations.append({
                            "label": label,
                            "source": source_info.get("source", "Unknown KB file"),
                            "note": "offline source"
                        })
                        logger.info(f"Added offline citation: {label} -> {source_info.get('source', 'Unknown KB file')}")
                    # Fallback if no sources
                    if not pack.sources:
                        citations.append({"label": "[KB]", "source": "Internal KB", "note": "offline source"})
                elif pack.mode == "online":
                    used_online = True
                    # Use actual URLs from pack
                    for i, source_info in enumerate(pack.sources):
                        label = f"[{i+1}]"
                        title = source_info.get("title", "")
                        url = source_info.get("source", "")
                        citations.append({
                            "label": label,
                            "source": url,
                            "title": title if title else url,
                            "note": "online source"
                        })
                        logger.info(f"Added online citation: {label} -> {title or url}")
                    # Fallback if no sources
                    if not pack.sources:
                        citations.append({"label": "[Web]", "source": "Web Search", "note": "online source"})

        combined_context = "\n\n---\n\n".join(contexts)

        # If we have dependent question context, prioritize it
        if dependent_question_context:
            if combined_context:
                combined_context = dependent_question_context + "\n\n---\n\n" + combined_context
            else:
                combined_context = dependent_question_context

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
                "Be direct, specific, and actionable in your answers.\n"
                "\n"
                "IMPORTANT FOR SYNTHESIS QUESTIONS:\n"
                "If you are answering a comparison or synthesis question (e.g., 'What is the difference between X and Y?'),\n"
                "and the context includes definitions or explanations of X and Y from previous answers,\n"
                "you SHOULD synthesize a comparison even if there is no direct search result about the comparison.\n"
                "Use logical reasoning to compare the features, characteristics, and use cases based on the provided definitions."
            )
        )

        # Build message list with conversation history
        messages = [system_message]

        # Add previous conversation turns if available
        if conversation_history:
            logger.info(f"Including conversation history with {len(conversation_history)} messages")
            for msg in conversation_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                    logger.debug(f"Added human message from history: {content[:100]}...")
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                    logger.debug(f"Added assistant message from history: {content[:100]}...")

        # Add current query with context
        current_query_message = HumanMessage(
            content=(
                f"Context:\n{combined_context}\n\n"
                f"User Question: {query}\n\n"
                f"Please provide a comprehensive answer based on the context above."
            )
        )
        messages.append(current_query_message)

        try:
            response = self.llm.invoke(messages)
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
