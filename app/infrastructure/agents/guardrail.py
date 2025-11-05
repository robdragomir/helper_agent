"""
Guardrail agent implementation - validates user queries before processing.
"""

import json
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core import settings
from app.application.interfaces import GuardrailAgent as GuardrailAgentInterface


class GuardrailAgent(GuardrailAgentInterface):
    """Validates user queries for safety and topic relevance at the beginning of processing."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
        )

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate user query for safety and relevance before processing.

        Args:
            query: The user's input query

        Returns:
            Dict with:
            - is_safe: bool - whether the content is safe and appropriate
            - is_in_scope: bool - whether the query is relevant to LangChain/LangGraph/AI engineering
            - decision: str - "allow", "reject", or "block"
            - reason: str - short explanation
        """
        system_message = SystemMessage(
            content=(
                "You are the Guardrail Agent for an AI engineering assistant that specializes in helping users with:\n"
                "- LangChain and LangGraph frameworks,\n"
                "- RAG (Retrieval-Augmented Generation) systems,\n"
                "- vector databases and embeddings,\n"
                "- agentic architectures and orchestration,\n"
                "- LLM application development in Python,\n"
                "- and closely related AI engineering topics.\n\n"
                "Your job is to evaluate the user's input before it reaches the main assistant.\n\n"
                "🧭 Your responsibilities:\n\n"
                "Check topic relevance: Determine whether the user's query is related to:\n"
                "- LangChain, LangGraph, or other LLM orchestration frameworks (CrewAI, Semantic Kernel, etc.)\n"
                "- Retrieval-Augmented Generation (RAG)\n"
                "- vector stores, embeddings, or retrievers\n"
                "- LLM agent design, routing, or planning\n"
                "- testing, monitoring, or evaluation of LLM systems\n"
                "- related MLOps or deployment practices (FastAPI, LangServe, etc.)\n\n"
                "Check appropriateness: If the query contains hate speech, explicit sexual content, requests for illegal activities, "
                "self-harm, or personal data collection — mark it as unsafe.\n\n"
                "Respond only in valid JSON with this exact schema:\n"
                "{\n"
                '  "is_safe": boolean,\n'
                '  "is_in_scope": boolean,\n'
                '  "decision": "allow" | "reject" | "block",\n'
                '  "reason": "one short sentence"\n'
                "}\n\n"
                "Only produce this JSON. Do not generate any explanation or text outside of it."
            )
        )

        human_message = HumanMessage(
            content=f"User query: {query}\n\nEvaluate this query for safety and relevance to LangChain/LangGraph/AI engineering topics."
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            response_text = response.content if hasattr(response, "content") else str(response)

            # Parse JSON response
            result = json.loads(response_text)

            # Validate response structure
            required_keys = {"is_safe", "is_in_scope", "decision", "reason"}
            if not all(key in result for key in required_keys):
                raise ValueError(f"Missing required keys in response. Got: {result.keys()}")

            return result

        except json.JSONDecodeError:
            # If parsing fails, default to allowing (fail-open for safety)
            return {
                "is_safe": True,
                "is_in_scope": True,
                "decision": "allow",
                "reason": "Could not validate query, allowing to proceed."
            }
        except Exception as e:
            print(f"Error in query guardrail validation: {e}")
            return {
                "is_safe": True,
                "is_in_scope": True,
                "decision": "allow",
                "reason": "Validation error, allowing to proceed."
            }