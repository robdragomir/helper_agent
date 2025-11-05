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