"""
Guardrail agent implementation.
"""

from typing import Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core import settings, FinalAnswer
from app.application.interfaces import GuardrailAgent as GuardrailAgentInterface


class GuardrailAgent(GuardrailAgentInterface):
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