"""
Telemetry and Evaluation Layer - Infrastructure layer.
Logs requests, metrics, and computes quality evaluations.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core import settings
from app.core.models import InferenceTrace, FinalAnswer


class TelemetryLogger:
    """Logs requests, decisions, and performance metrics."""

    def __init__(self):
        self.log_path = settings.telemetry_log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_inference(self, trace: InferenceTrace) -> None:
        """Log a complete inference trace."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(trace.model_dump(mode="json"), default=str) + "\n")

    def get_recent_traces(self, limit: int = 100) -> List[InferenceTrace]:
        """Get recent inference traces."""
        traces = []
        if not os.path.exists(self.log_path):
            return traces

        with open(self.log_path, "r") as f:
            lines = f.readlines()[-limit:]
            for line in lines:
                try:
                    data = json.loads(line)
                    # Reconstruct InferenceTrace from dict
                    traces.append(data)
                except json.JSONDecodeError:
                    pass

        return traces


class EvaluationMetrics:
    """Computes quality metrics for answers."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
        )
        self.metrics_path = settings.eval_metrics_path
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

    def compute_faithfulness(self, answer: str, context: str) -> float:
        """
        Compute faithfulness score (0.0-1.0).
        Measures how well the answer is grounded in the provided context.
        """
        system_message = SystemMessage(
            content=(
                "You are an expert at evaluating answer quality. "
                "Assess whether the answer is faithful to the provided context. "
                "Return a score from 0.0 to 1.0 where 1.0 means the answer is entirely grounded in context."
            )
        )

        human_message = HumanMessage(
            content=(
                f"Context:\n{context}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Return only a number between 0.0 and 1.0 representing faithfulness."
            )
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            response_text = response.content if hasattr(response, "content") else str(response)

            # Extract score from response
            import re

            score_match = re.search(r"0\.\d+|1\.0", response_text)
            if score_match:
                return float(score_match.group())
        except Exception as e:
            print(f"Error computing faithfulness: {e}")

        return 0.5  # Default neutral score

    def compute_unsupported_claims(self, answer: str, context: str) -> int:
        """
        Count unsupported claims in the answer.
        Returns the number of claims not grounded in context.
        """
        system_message = SystemMessage(
            content=(
                "You are an expert at identifying unsupported claims. "
                "Identify claims in the answer that are NOT supported by the context. "
                "List each unsupported claim on a new line starting with '- '."
            )
        )

        human_message = HumanMessage(
            content=(
                f"Context:\n{context}\n\n"
                f"Answer:\n{answer}\n\n"
                f"List any claims in the answer not supported by the context."
            )
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            response_text = response.content if hasattr(response, "content") else str(response)

            # Count lines starting with '- '
            unsupported_claims = len([line for line in response_text.split("\n") if line.strip().startswith("-")])
            return unsupported_claims

        except Exception as e:
            print(f"Error computing unsupported claims: {e}")

        return 0

    def compute_retrieval_coverage(
        self, query: str, retrieved_chunks: List[str]
    ) -> float:
        """
        Compute retrieval coverage (0.0-1.0).
        Measures whether the retrieved chunks contain information relevant to the query.
        """
        if not retrieved_chunks:
            return 0.0

        chunks_context = "\n\n".join(retrieved_chunks[:3])  # Use top 3 chunks

        system_message = SystemMessage(
            content=(
                "You are an expert at evaluating information retrieval quality. "
                "Assess whether the retrieved chunks contain relevant information to answer the query. "
                "Return a score from 0.0 to 1.0."
            )
        )

        human_message = HumanMessage(
            content=(
                f"Query: {query}\n\n"
                f"Retrieved Chunks:\n{chunks_context}\n\n"
                f"Return only a number between 0.0 and 1.0 representing coverage."
            )
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            response_text = response.content if hasattr(response, "content") else str(response)

            import re

            score_match = re.search(r"0\.\d+|1\.0", response_text)
            if score_match:
                return float(score_match.group())

        except Exception as e:
            print(f"Error computing retrieval coverage: {e}")

        return 0.5

    def compute_source_freshness(self, sources: List[Dict[str, str]]) -> float:
        """
        Compute source freshness (0.0-1.0).
        Based on how recent the sources are.
        """
        if not sources:
            return 0.5

        freshness_scores = []

        for source in sources:
            # Parse date if available
            date_str = source.get("date", "")
            if date_str:
                try:
                    from datetime import datetime

                    source_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    now = datetime.now(source_date.tzinfo) if source_date.tzinfo else datetime.now()

                    days_old = (now - source_date).days

                    # Score: 1.0 for today, 0.0 for 365+ days
                    if days_old <= 0:
                        freshness_scores.append(1.0)
                    elif days_old >= 365:
                        freshness_scores.append(0.0)
                    else:
                        freshness_scores.append(1.0 - (days_old / 365.0))

                except Exception:
                    freshness_scores.append(0.5)
            else:
                freshness_scores.append(0.5)

        return sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0.5

    def save_metrics(self, trace_id: str, metrics: Dict[str, float]) -> None:
        """Save evaluation metrics."""
        metrics_entry = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }

        # Append to metrics file
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics_entry) + "\n")

    def get_average_metrics(self, limit: int = 100) -> Dict[str, float]:
        """Get average metrics over recent traces."""
        if not os.path.exists(self.metrics_path):
            return {}

        metrics_data = []
        with open(self.metrics_path, "r") as f:
            lines = f.readlines()[-limit:]
            for line in lines:
                try:
                    data = json.loads(line)
                    metrics_data.append(data.get("metrics", {}))
                except json.JSONDecodeError:
                    pass

        if not metrics_data:
            return {}

        # Compute averages
        all_keys = set()
        for m in metrics_data:
            all_keys.update(m.keys())

        averages = {}
        for key in all_keys:
            values = [m[key] for m in metrics_data if key in m]
            if values:
                averages[key] = sum(values) / len(values)

        return averages