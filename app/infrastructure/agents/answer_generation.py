"""
Answer generation agent implementation.
"""

from typing import List, Dict, Optional
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core import settings, EvidencePack, FinalAnswer
from app.application.interfaces import AnswerAgent, EvaluationMetrics

logger = logging.getLogger(__name__)


class AnswerGenerationAgent(AnswerAgent):
    """Generates answer based on provided evidence."""

    def __init__(self, eval_metrics: EvaluationMetrics):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
        )
        self.eval_metrics = eval_metrics

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