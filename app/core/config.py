"""
Configuration management for the LangGraph Helper Agent.
Follows onion architecture - core configuration layer.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # API Keys
    google_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None

    # LLM Configuration
    llm_model: str = "gemini-1.5-flash"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Knowledge Base Configuration
    langgraph_docs_url: str = "https://langchain-ai.github.io/langgraph/reference/llms-full.txt"
    langchain_docs_url: str = "https://docs.langchain.com/llms.txt"
    local_docs_path: str = "data/langgraph-docs.txt"
    kb_chunks_path: str = "data/kb_chunks.json"
    vector_store_path: str = "data/faiss_index"

    # RAG Configuration
    top_k_chunks: int = 5
    similarity_threshold: float = 1.5  # L2 distance threshold (lower = more similar, typical range 0.5-2.0)

    # Search Configuration
    max_online_search_results: int = 5
    min_source_agreement_ratio: float = 0.6

    # Telemetry Configuration
    telemetry_log_path: str = "logs/telemetry.jsonl"
    eval_metrics_path: str = "logs/eval_metrics.json"

    # Model Configuration
    chunk_size_for_agentic_chunking: int = 2000
    agentic_chunking_threshold_kb: int = 30  # File size threshold in KB for agentic chunking

    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env


# Global settings instance
settings = Settings()