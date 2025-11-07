"""
Query decomposition agent implementation.
"""

from typing import Dict
import logging
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core import settings
from app.application.interfaces import DecompositionAgent

logger = logging.getLogger(__name__)


class QueryDecompositionAgent(DecompositionAgent):
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
            content=("""
                You are a query decomposition planner specialized in LangChain, LangGraph, and LLM-based systems.
                You will receive user questions that are related to:
                    - LangChain, LangGraph, or similar LLM orchestration frameworks,
                    - Retrieval-Augmented Generation (RAG) systems,
                    - vector stores, embeddings, or retrievers,
                    - LLM agent design and execution,
                    - evaluation, monitoring, and optimization of LLM pipelines, or general concepts in large language model engineering.
                
                Your job is to break the user’s question into a small set of simpler subquestions and to specify dependencies between them, interpreting all terminology within this AI engineering context.
                Guidelines:
                    Each subquestion:
                        - Must be answerable from documentation or web search related to the above technologies.
                        - Should be clear, explicit, and avoid pronouns like "it" or "they".
                        - Should focus on a single topic or step, such as a single concept (e.g. “StateGraph”, “Memory”), method, or framework component.
                
                    Dependencies:
                        - Some subquestions may require the answers to earlier subquestions.
                        - Use a requires field containing a list of previous subquestion IDs whose answers are needed.
                        - If a subquestion does not depend on any previous answer, use an empty list [].
                        - Subquestions must be ordered so that any ID in requires appears earlier in the list.
                
                    Final question:
                        - The final subquestion should correspond to the user’s original goal (e.g. comparison, explanation, reasoning, or implementation question).
                        - You must specify final_question_id as the ID of the subquestion whose answer should be returned to the user as the final answer.
                
                    Important:
                
                        - Interpret all technical terms (e.g., “StateGraph”, “MessageGraph”, “memory”, “agents”, “chains”) as concepts from LangGraph, LangChain, or LLM architecture.
                        - If the user question uses ambiguous or broad wording, infer the most likely meaning within this context (e.g., "memory" means LLM memory management, not human memory).
                        - Avoid generic or unrelated decompositions.
                        - Not all questions need to be decomposed. If a question is already simple enough to be answered directly, you don't need to break it down further.
                
                Output only valid JSON with this exact schema:
                
                {
                  "decomposed_questions": [
                    {"id": "q1", "question": "string", "requires": ["q_id_1", "q_id_2"]}
                  ],
                  "final_question_id": "qX"
                }
                
                    - requires must always be a list (use [] if no dependencies).
                    - Do not include any comments or extra fields.
                    - Do not generate explanations or natural language outside the JSON."""
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