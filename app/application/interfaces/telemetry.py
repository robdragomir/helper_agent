"""
Telemetry logger interface.
"""

from abc import ABC, abstractmethod
from app.core import InferenceTrace


class TelemetryLogger(ABC):
    """Abstract interface for telemetry logging."""

    @abstractmethod
    def log_inference(self, trace: InferenceTrace) -> None:
        """
        Log an inference trace.

        Args:
            trace: The inference trace to log
        """
        pass

    @abstractmethod
    def get_recent_traces(self, limit: int = 100) -> list:
        """
        Get recent inference traces.

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of recent traces
        """
        pass