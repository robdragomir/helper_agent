"""
LangGraph Workflow Orchestration - Application layer.
Defines the multi-agent workflow using LangGraph.
"""

import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.core import (
    settings,
    EvidencePack,
    FinalAnswer,
    InferenceTrace,
)
from app.application.agents import (
    OfflineSearchAgent,
    OnlineSearchAgent,
    AnswerGenerationAgent,
    GuardrailAgent,
)
from app.infrastructure.telemetry import TelemetryLogger, EvaluationMetrics


# Workflow State
class WorkflowState(dict):
    """State management for the workflow."""

    def __init__(self, query: str, mode: Optional[str] = None):
        super().__init__()
        if mode not in ["offline", "online", "both"]:
            raise ValueError(f"Mode must be 'offline', 'online', or 'both', got '{mode}'")

        self["trace_id"] = str(uuid.uuid4())
        self["query"] = query
        self["mode"] = mode
        self["offline_evidence"] = None
        self["online_evidence"] = None
        self["final_answer"] = None
        self["guardrail_passed"] = None
        self["start_time"] = datetime.now()
        self["offline_context_preview"] = None
        self["online_context_preview"] = None


class WorkflowOrchestrator:
    """Orchestrates the multi-agent workflow using LangGraph."""

    def __init__(self):
        self.offline_agent = OfflineSearchAgent()
        self.online_agent = OnlineSearchAgent()
        self.answer_agent = AnswerGenerationAgent()
        self.guardrail_agent = GuardrailAgent()
        self.telemetry = TelemetryLogger()
        self.eval_metrics = EvaluationMetrics()

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(dict)

        # Add nodes
        graph.add_node("search_offline", self._search_offline_node)
        graph.add_node("search_online", self._search_online_node)
        graph.add_node("generate_answer", self._generate_answer_node)
        graph.add_node("validate_answer", self._validate_answer_node)
        graph.add_node("finalize", self._finalize_node)

        # Set entry point with conditional routing based on mode
        graph.add_conditional_edges(
            START,
            self._mode_condition,
        )

        # Edges after search
        graph.add_edge("search_offline", "generate_answer")
        graph.add_edge("search_online", "generate_answer")

        # Answer generation
        graph.add_edge("generate_answer", "validate_answer")

        # Validation
        graph.add_edge("validate_answer", "finalize")

        # End
        graph.add_edge("finalize", END)

        return graph.compile()

    def _mode_condition(self, state: WorkflowState):
        """Route based on the requested mode."""
        mode = state.get("mode")

        if mode == "both":
            return [Send("search_offline", state), Send("search_online", state)]
        elif mode == "offline":
            return "search_offline"
        else:  # online
            return "search_online"

    def _search_offline_node(self, state: WorkflowState) -> WorkflowState:
        """Search offline knowledge base."""
        evidence = self.offline_agent.search(state["query"])
        state["offline_evidence"] = evidence
        state["offline_context_preview"] = evidence.context_text[:200] if evidence.context_text else None
        return state

    def _search_online_node(self, state: WorkflowState) -> WorkflowState:
        """Search online sources."""
        evidence = self.online_agent.search(state["query"])
        state["online_evidence"] = evidence
        state["online_context_preview"] = evidence.context_text[:200] if evidence.context_text else None
        return state

    def _generate_answer_node(self, state: WorkflowState) -> WorkflowState:
        """Generate answer from evidence."""
        evidence_packs = []

        if state.get("offline_evidence"):
            evidence_packs.append(state["offline_evidence"])
        if state.get("online_evidence"):
            evidence_packs.append(state["online_evidence"])

        if not evidence_packs:
            state["final_answer"] = FinalAnswer(
                text="Unable to find information to answer your question.",
                used_offline=False,
                used_online=False,
                answer_confidence=0.0,
                citations=[],
            )
        else:
            state["final_answer"] = self.answer_agent.generate(
                state["query"], evidence_packs
            )

        return state

    def _validate_answer_node(self, state: WorkflowState) -> WorkflowState:
        """Validate answer for safety."""
        answer = state["final_answer"]
        is_valid, reason = self.guardrail_agent.validate(answer)
        state["guardrail_passed"] = is_valid

        if not is_valid:
            # Replace answer with safe message
            answer.text = f"I cannot provide that answer due to safety concerns: {reason}"
            answer.answer_confidence = 0.0

        return state

    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize and log the inference."""
        # Create trace
        end_time = datetime.now()
        latency_ms = int((end_time - state["start_time"]).total_seconds() * 1000)

        trace = InferenceTrace(
            query_text=state["query"],
            route=state.get("mode"),
            offline_context_preview=state.get("offline_context_preview"),
            online_context_preview=state.get("online_context_preview"),
            final_answer=state["final_answer"],
            latency_ms_total=latency_ms,
            token_usage={},
        )

        # Log trace
        self.telemetry.log_inference(trace)

        # Compute and save evaluation metrics
        if state["final_answer"]:
            metrics = {}
            combined_context = ""

            if state.get("offline_evidence"):
                combined_context += state["offline_evidence"].context_text

            if state.get("online_evidence"):
                if combined_context:
                    combined_context += "\n\n---\n\n"
                combined_context += state["online_evidence"].context_text

            if combined_context:
                metrics["faithfulness"] = self.eval_metrics.compute_faithfulness(
                    state["final_answer"].text, combined_context
                )
                metrics["unsupported_claims"] = self.eval_metrics.compute_unsupported_claims(
                    state["final_answer"].text, combined_context
                )

            if state.get("offline_evidence"):
                chunks = state["offline_evidence"].context_text.split("\n\n---\n\n")
                metrics["retrieval_coverage"] = (
                    self.eval_metrics.compute_retrieval_coverage(state["query"], chunks)
                )

            if state.get("online_evidence"):
                metrics["source_freshness"] = self.eval_metrics.compute_source_freshness(
                    state["final_answer"].citations
                )

            self.eval_metrics.save_metrics(state["trace_id"], metrics)

        return state

    def run(self, query: str, mode: str) -> FinalAnswer:
        """
        Run the complete workflow.

        Args:
            query: The user's question
            mode: The search mode - must be 'offline', 'online', or 'both'

        Returns:
            FinalAnswer with the response

        Raises:
            ValueError: If mode is not 'offline', 'online', or 'both'
        """
        state = WorkflowState(query, mode)

        # Execute the graph
        result_state = self.graph.invoke(state)

        return result_state.get("final_answer")


# Singleton workflow instance
_workflow_instance = None


def get_workflow() -> WorkflowOrchestrator:
    """Get or create the workflow instance."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = WorkflowOrchestrator()
    return _workflow_instance