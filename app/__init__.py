"""
LangGraph Helper Agent - Multi-agent system for answering LangGraph questions.
Follows onion architecture pattern.

Structure:
- core: Domain models and configuration
- infrastructure: External system integrations (KB, vector store, web search)
- application: Business logic and agent orchestration
- presentation: User interface (CLI)
"""

__version__ = "0.1.0"