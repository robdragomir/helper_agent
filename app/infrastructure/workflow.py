"""
LangGraph Workflow Orchestration - Infrastructure layer.
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
from app.application.interfaces import (
    DecompositionAgent,
    SearchAgent,
    AnswerAgent,
    GuardrailAgent,
    TelemetryLogger,
    EvaluationMetrics,
)
from app.infrastructure.agents import (
    QueryDecompositionAgent,
    OfflineSearchAgent,
    OnlineSearchAgent,
    AnswerGenerationAgent as AnswerGenerationAgentImpl,
    GuardrailAgentImpl,
)
from app.infrastructure.services import TelemetryLogger as TelemetryLoggerImpl
from app.infrastructure.services import EvaluationMetrics as EvaluationMetricsImpl


# Workflow State
class WorkflowState(dict):
    """State management for the workflow."""

    def __init__(self, query: str, mode: Optional[str] = None):
        super().__init__()
        if mode not in ["offline", "online"]:
            raise ValueError(f"Mode must be 'offline' or 'online', got '{mode}'")

        self["trace_id"] = str(uuid.uuid4())
        self["query"] = query
        self["mode"] = mode
        self["decomposition"] = None  # Decomposed questions
        self["current_question_idx"] = 0  # Current question being processed
        self["question_answers"] = {}  # Map of question_id -> (query, answer, evidence_packs)
        self["offline_evidence"] = None
        self["online_evidence"] = None
        self["final_answer"] = None
        self["guardrail_passed"] = None
        self["start_time"] = datetime.now()
        self["offline_context_preview"] = None
        self["online_context_preview"] = None
        self["all_sources"] = []  # Aggregate all sources from all subquestions


class WorkflowOrchestrator:
    """Orchestrates the multi-agent workflow using LangGraph."""

    def __init__(
        self,
        decomposition_agent: Optional[DecompositionAgent] = None,
        offline_agent: Optional[SearchAgent] = None,
        online_agent: Optional[SearchAgent] = None,
        answer_agent: Optional[AnswerAgent] = None,
        guardrail_agent: Optional[GuardrailAgent] = None,
        telemetry: Optional[TelemetryLogger] = None,
        eval_metrics: Optional[EvaluationMetrics] = None,
    ):
        """
        Initialize the workflow orchestrator with optional dependency injection.
        If not provided, default implementations are created.
        """
        self.decomposition_agent: DecompositionAgent = decomposition_agent or QueryDecompositionAgent()
        self.offline_agent: SearchAgent = offline_agent or OfflineSearchAgent()
        self.online_agent: SearchAgent = online_agent or OnlineSearchAgent()
        self.eval_metrics: EvaluationMetrics = eval_metrics or EvaluationMetricsImpl()
        self.answer_agent: AnswerAgent = answer_agent or AnswerGenerationAgentImpl(self.eval_metrics)
        self.guardrail_agent: GuardrailAgent = guardrail_agent or GuardrailAgentImpl()
        self.telemetry: TelemetryLogger = telemetry or TelemetryLoggerImpl()

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for query validation, decomposition, and answering."""
        import logging
        logger = logging.getLogger(__name__)
        graph = StateGraph(dict)

        # Add nodes
        graph.add_node("guardrail", self._guardrail_node)
        graph.add_node("decompose", self._decompose_node)
        graph.add_node("process_subquestion", self._process_subquestion_node)
        graph.add_node("validate_answer", self._validate_answer_node)
        graph.add_node("finalize", self._finalize_node)

        # Start with guardrail check
        graph.add_edge(START, "guardrail")

        # Guardrail -> decomposition (or reject if not safe/in-scope)
        graph.add_conditional_edges(
            "guardrail",
            self._check_guardrail_decision,
            {
                "allow": "decompose",
                "reject": "finalize",
                "block": "finalize"
            }
        )

        # Decompose -> process subquestions (loop)
        graph.add_edge("decompose", "process_subquestion")

        # Process subquestion -> check if more questions or finalize
        graph.add_conditional_edges(
            "process_subquestion",
            self._has_more_questions,
            {
                "process_more": "process_subquestion",
                "finalize": "validate_answer"
            }
        )

        # Validate -> finalize
        graph.add_edge("validate_answer", "finalize")

        # End
        graph.add_edge("finalize", END)

        return graph.compile()

    def _guardrail_node(self, state: WorkflowState) -> WorkflowState:
        """Check query for safety and relevance using the guardrail agent."""
        import logging
        logger = logging.getLogger(__name__)

        query = state["query"]
        logger.info(f"_guardrail_node: Validating query for safety and relevance")

        # Validate query using guardrail agent
        guardrail_result = self.guardrail_agent.validate(query)
        state["guardrail_result"] = guardrail_result

        logger.info(f"_guardrail_node: Guardrail result: {guardrail_result['decision']}")
        logger.info(f"  is_safe: {guardrail_result['is_safe']}")
        logger.info(f"  is_in_scope: {guardrail_result['is_in_scope']}")
        logger.info(f"  reason: {guardrail_result['reason']}")

        return state

    def _check_guardrail_decision(self, state: WorkflowState) -> str:
        """Route based on guardrail decision."""
        guardrail_result = state.get("guardrail_result", {})
        decision = guardrail_result.get("decision", "allow")

        if decision == "allow":
            return "allow"
        elif decision == "reject":
            # Create an answer indicating the query is out of scope
            state["final_answer"] = FinalAnswer(
                text=f"I can only help with LangChain, LangGraph, and related AI engineering topics. {guardrail_result.get('reason', 'Your query appears to be out of scope.')}",
                used_offline=False,
                used_online=False,
                answer_confidence=0.0,
                citations=[],
            )
            return "reject"
        else:  # block
            # Create an answer indicating the query is blocked for safety
            state["final_answer"] = FinalAnswer(
                text=f"I cannot process this request for safety reasons. {guardrail_result.get('reason', 'Your query contains inappropriate content.')}",
                used_offline=False,
                used_online=False,
                answer_confidence=0.0,
                citations=[],
            )
            return "block"

    def _decompose_node(self, state: WorkflowState) -> WorkflowState:
        """Decompose the user query into subquestions."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("_decompose_node: Starting query decomposition")

        decomposition = self.decomposition_agent.decompose(state["query"])
        state["decomposition"] = decomposition
        state["current_question_idx"] = 0

        logger.info(f"_decompose_node: Decomposed into {len(decomposition['decomposed_questions'])} questions")
        return state

    def _process_subquestion_node(self, state: WorkflowState) -> WorkflowState:
        """Process the current subquestion through search and answer generation."""
        import logging
        logger = logging.getLogger(__name__)

        decomposition = state["decomposition"]
        current_idx = state["current_question_idx"]
        questions = decomposition["decomposed_questions"]

        if current_idx >= len(questions):
            logger.info("_process_subquestion_node: No more questions to process")
            return state

        current_q = questions[current_idx]
        question_id = current_q["id"]
        question_text = current_q["question"]
        requires = current_q.get("requires", [])

        logger.info(f"_process_subquestion_node: Processing {question_id}: {question_text}")
        if requires:
            logger.info(f"  Requires: {requires}")

        # Prepare context from required questions
        required_answers_context = ""
        if requires:
            required_answers_context = "Previous answers for context:\n"
            for req_id in requires:
                if req_id in state["question_answers"]:
                    req_query, req_answer, _ = state["question_answers"][req_id]
                    required_answers_context += f"\n[{req_id}] Q: {req_query}\nA: {req_answer.text}\n"

        # Search for this question based on mode
        mode = state["mode"]
        evidence = None

        if mode == "offline":
            evidence = self.offline_agent.search(question_text)
        elif mode == "online":
            evidence = self.online_agent.search(question_text)

        # Track sources
        evidence_packs = []
        if evidence and evidence.context_text:
            evidence_packs.append(evidence)
            state["all_sources"].extend(evidence.sources)

        # Generate answer
        if not evidence_packs and not requires:
            # No evidence and no dependencies - return empty answer
            answer = FinalAnswer(
                text=f"No information found for: {question_text}",
                used_offline=mode == "offline",
                used_online=mode == "online",
                answer_confidence=0.0,
                citations=[],
            )
        else:
            # Include required answers as dependent context
            conv_history = state.get("conversation_history", []).copy() if state.get("conversation_history") else []

            answer = self.answer_agent.generate(
                question_text,
                evidence_packs,
                conversation_history=conv_history,
                dependent_question_context=required_answers_context if required_answers_context else None
            )

        # Store the answer
        state["question_answers"][question_id] = (question_text, answer, evidence_packs)
        state["current_question_idx"] += 1

        # Log the intermediate answer
        logger.info(f"_process_subquestion_node: INTERMEDIATE ANSWER FOR {question_id}")
        logger.info(f"  Question: {question_text}")
        logger.info(f"  Answer: {answer.text[:500]}")
        if answer.text and len(answer.text) > 500:
            logger.info(f"  ... (truncated)")
        logger.info(f"  Confidence: {answer.answer_confidence:.2f}")
        logger.info(f"  Sources: {answer.citations}")
        logger.info(f"  Has required context: {bool(requires)}")
        logger.info(f"_process_subquestion_node: Stored answer for {question_id}, moved to index {state['current_question_idx']}")
        return state

    def _has_more_questions(self, state: WorkflowState) -> str:
        """Check if there are more subquestions to process."""
        import logging
        logger = logging.getLogger(__name__)

        decomposition = state.get("decomposition")
        if not decomposition:
            logger.info("_has_more_questions: No decomposition, finalizing")
            return "finalize"

        current_idx = state["current_question_idx"]
        total_questions = len(decomposition["decomposed_questions"])

        if current_idx < total_questions:
            logger.info(f"_has_more_questions: More questions ({current_idx}/{total_questions}), processing more")
            return "process_more"
        else:
            logger.info(f"_has_more_questions: All questions processed ({current_idx}/{total_questions}), finalizing")
            return "finalize"

    def _validate_answer_node(self, state: WorkflowState) -> WorkflowState:
        """Get the final answer from question_answers (guardrail check already happened at query level)."""
        import logging
        logger = logging.getLogger(__name__)

        decomposition = state.get("decomposition")
        if not decomposition:
            logger.warning("_validate_answer_node: No decomposition found")
            return state

        final_question_id = decomposition["final_question_id"]
        if final_question_id not in state["question_answers"]:
            logger.warning(f"_validate_answer_node: Final question {final_question_id} not found in answers")
            return state

        _, final_answer, _ = state["question_answers"][final_question_id]

        # Store the final answer (guardrail check already happened at the query level)
        state["final_answer"] = final_answer
        logger.info(f"_validate_answer_node: Retrieved final answer for question {final_question_id}")
        return state

    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize by logging traces and metrics."""
        import logging
        logger = logging.getLogger(__name__)

        end_time = datetime.now()
        latency_ms = int((end_time - state["start_time"]).total_seconds() * 1000)

        # Create trace for the overall query
        trace = InferenceTrace(
            query_text=state["query"],
            route=state.get("mode"),
            offline_context_preview=None,
            online_context_preview=None,
            final_answer=state.get("final_answer"),
            latency_ms_total=latency_ms,
            token_usage={},
        )

        # Log trace
        self.telemetry.log_inference(trace)

        # Aggregate metrics from all subquestions
        if state["question_answers"]:
            metrics = {}
            combined_context = ""

            for question_id, (query_text, answer, evidence_packs) in state["question_answers"].items():
                for pack in evidence_packs:
                    if pack.context_text:
                        combined_context += pack.context_text + "\n\n---\n\n"

            combined_context = combined_context.rstrip("\n\n---\n\n")

            if combined_context and state.get("final_answer"):
                metrics["faithfulness"] = self.eval_metrics.compute_faithfulness(
                    state["final_answer"].text, combined_context
                )
                metrics["unsupported_claims"] = self.eval_metrics.compute_unsupported_claims(
                    state["final_answer"].text, combined_context
                )
                self.eval_metrics.save_metrics(state["trace_id"], metrics)

        logger.info(f"_finalize_node: Finalized trace {state['trace_id']}")
        return state

    def run(self, query: str, mode: str, conversation_history: Optional[list] = None) -> FinalAnswer:
        """
        Run the complete workflow with query decomposition.

        Args:
            query: The user's question
            mode: The search mode - must be 'offline' or 'online'
            conversation_history: Optional list of conversation messages for context

        Returns:
            FinalAnswer with the response and aggregated sources from all subquestions

        Raises:
            ValueError: If mode is not 'offline' or 'online'
        """
        import logging
        logger = logging.getLogger(__name__)

        state = WorkflowState(query, mode)
        state["conversation_history"] = conversation_history or []

        logger.info(f"WorkflowOrchestrator.run() called with query: '{query}', mode: {mode}")

        # Execute the graph
        result_state = self.graph.invoke(state)

        final_answer = result_state.get("final_answer")
        if final_answer:
            # Add aggregated sources from all subquestions
            all_sources = result_state.get("all_sources", [])
            # Deduplicate sources
            seen_sources = set()
            unique_sources = []
            for source in all_sources:
                source_str = source.get("source", "")
                if source_str and source_str not in seen_sources:
                    unique_sources.append(source)
                    seen_sources.add(source_str)

            if unique_sources and not final_answer.citations:
                # If we don't have citations, build them from aggregated sources
                for i, source_info in enumerate(unique_sources):
                    label = f"[{i+1}]"
                    if source_info.get("type") == "online":
                        final_answer.citations.append({
                            "label": label,
                            "source": source_info.get("source", ""),
                            "title": source_info.get("title", ""),
                            "note": "online source"
                        })
                    else:
                        final_answer.citations.append({
                            "label": label,
                            "source": source_info.get("source", ""),
                            "note": "offline source"
                        })

        logger.info(f"WorkflowOrchestrator.run() completed")
        return final_answer


# Singleton workflow instance
_workflow_instance = None


def get_workflow() -> WorkflowOrchestrator:
    """Get or create the workflow instance."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = WorkflowOrchestrator()
    return _workflow_instance