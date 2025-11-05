"""
Evaluation metrics interface.
"""

from abc import ABC, abstractmethod


class EvaluationMetrics(ABC):
    """Abstract interface for evaluation metrics computation."""

    @abstractmethod
    def compute_faithfulness(self, answer: str, context: str) -> float:
        """
        Compute faithfulness score (0.0-1.0) for an answer given context.

        Args:
            answer: The generated answer
            context: The context used to generate the answer

        Returns:
            Faithfulness score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def compute_unsupported_claims(self, answer: str, context: str) -> float:
        """
        Compute ratio of unsupported claims in the answer.

        Args:
            answer: The generated answer
            context: The context used to generate the answer

        Returns:
            Score between 0.0 and 1.0 (lower is better)
        """
        pass

    @abstractmethod
    def compute_retrieval_coverage(self, query: str, chunks: list) -> float:
        """
        Compute how well retrieved chunks cover the query.

        Args:
            query: The original query
            chunks: The retrieved chunks

        Returns:
            Coverage score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def compute_source_freshness(self, citations: list) -> float:
        """
        Compute freshness of sources used in citations.

        Args:
            citations: List of citations from the answer

        Returns:
            Freshness score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def save_metrics(self, trace_id: str, metrics: dict) -> None:
        """
        Save computed metrics for a trace.

        Args:
            trace_id: Unique identifier for the trace
            metrics: Dictionary of computed metrics
        """
        pass

    @abstractmethod
    def get_average_metrics(self, limit: int = 100) -> dict:
        """
        Get average metrics across recent traces.

        Args:
            limit: Number of recent traces to average

        Returns:
            Dictionary of averaged metrics
        """
        pass