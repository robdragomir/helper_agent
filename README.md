# LangGraph Helper Agent

A sophisticated multi-agent system for answering LangGraph documentation questions using offline RAG, online search, and agentic orchestration with LangGraph.

## Architecture

This project follows **onion architecture** pattern with clear separation of concerns:

```
┌──────────────────────────────────────────────────────────┐
│   Core Layer (Models & Config)                           │
│   - Pydantic domain models (EvidencePack, FinalAnswer)   │
│   - Configuration management                             │
│   - Logging configuration                                │
└──────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────┐
│   Application Layer (Interfaces + Orchestration)         │
│   - Abstract interfaces for all agents                   │
│   - WorkflowOrchestrator with LangGraph                  │
│   - Query decomposition into subquestions                │
│   - Dependency injection for testability                 │
└──────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────┐
│   Infrastructure Layer (Implementations)                 │
│   - Query Decomposition Agent (LLM-based)                │
│   - Offline Search Agent (FAISS vector search)           │
│   - Online Search Agent (Tavily + reranking)             │
│   - Answer Generation Agent (context-aware)              │
│   - Guardrail Agent (safety validation)                  │
│   - Telemetry Logger & Evaluation Metrics                │
│   - KB Manager (agentic chunking + FAISS)               │
│   - Online Search Manager (Tavily integration)          │
└──────────────────────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────────────────────┐
│   Presentation Layer (CLI)                               │
│   - User interface with typer                            │
│   - Minimal output: answer + sources only                │
└──────────────────────────────────────────────────────────┘
```

## Operating Modes

### Offline Mode
- **Command**: `python -m main ask "question" --mode offline`
- **Data Source**: Local knowledge base built from:
  - LangGraph documentation (latest release)
  - LangChain documentation (latest release)
- **Search Method**: Semantic similarity using FAISS vector store
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dim vectors)
- **Processing**: Instant responses (no API calls, no latency)
- **Data Freshness**: Updated when you run `build-kb` (detects changes via SHA256 hashing)
- **Cost**: Free (no API costs)

### Online Mode
- **Command**: `python -m main ask "question" --mode online`
- **Data Source**: Web search via Tavily API
- **Search Method**: Full-text web search with relevance scoring
- **Result Validation**: LLM-based credibility assessment
- **Processing**: Network latency + LLM inference time
- **Data Freshness**: Real-time (always latest information)
- **Cost**: Requires Tavily API key (credits-based pricing)

## Key Features

### Intelligent Query Processing
- **Query Decomposition**: Complex questions broken into simpler subquestions
- **Dependency Tracking**: Subquestions can depend on answers from previous questions
- **Context Synthesis**: Answers combined intelligently for final response
- **Conversation History**: Multi-turn conversations maintained in interactive mode

### Advanced RAG System
- **Agentic Chunking**: LLM-driven semantic document chunking (configurable size: default 2000 chars)
- **Vector Store**: FAISS with `all-MiniLM-L6-v2` embeddings for efficient similarity search
- **Change Detection**: SHA256-based detection of documentation updates
- **Source Tracking**: Original file sources preserved through chunking to final citations
- **Metadata Management**: Chunks include file source and context information

### Smart Answer Generation
- **Offline Search Agent**: Semantic search retrieving top-K chunks (default: 5)
- **Online Search Agent**: Web search with result validation and reranking
- **Answer Generation**: Context-aware answers using conversation history
- **Safety Validation**: Guardrail agent checks for harmful/unsafe content

### Quality & Observability
- **Confidence Scoring**: 60% evidence quality + 40% answer faithfulness
- **Telemetry Logging**: Full inference traces to `data/telemetry.jsonl`
- **Quality Metrics**:
  - Faithfulness: How well answer is grounded in context (0.0-1.0)
  - Unsupported Claims: Count of claims not in context
  - Retrieval Coverage: Whether retrieved content is relevant (0.0-1.0)
  - Source Freshness: Recency of online sources (0.0-1.0)

## Setup

### Prerequisites
- Python 3.11+
- Google Gemini API Key
- Tavily API Key (optional, for online search)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone and setup environment**
```bash
cd helper_agent
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys
export $(cat .env | xargs)
```

4. **Build knowledge base**
```bash
python -m main build-kb
```

## Usage

### CLI Commands

**Ask a question (offline mode)**
```bash
python -m main ask "How do I create a basic graph?" --mode offline
```

**Ask a question (online mode)**
```bash
python -m main ask "What are the latest LangGraph features?" --mode online
```

**Interactive mode (follow-up questions)**
```bash
# Start with offline mode, stay in interactive for follow-ups
python -m main ask "How does state work?" --mode offline --interactive

# Or use the dedicated interactive command
python -m main interactive --mode offline
```

**Build knowledge base**
```bash
# Build KB from remote docs (LangGraph + LangChain)
python -m main build-kb

# Force rebuild even if up to date
python -m main build-kb --force-rebuild
```

**View statistics**
```bash
python -m main stats
```

### Docker Deployment

**Build and run with Docker Compose**
```bash
# Create .env file with API keys
cat > .env << EOF
GOOGLE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
EOF

# Build image
docker-compose build

# Run commands
docker-compose run langgraph-helper ask "How do I use graphs?"

# Interactive mode
docker-compose run -it langgraph-helper interactive

# Build KB
docker-compose run langgraph-helper build-kb
```

**Build and run standalone**
```bash
docker build -t langgraph-helper .
docker run --env-file .env -it langgraph-helper ask "Your question here"
```

## Project Structure

```
helper_agent/
├── app/
│   ├── core/
│   │   ├── models.py              # Pydantic domain models (EvidencePack, FinalAnswer, etc.)
│   │   ├── config.py              # Configuration & settings management
│   │   └── logging_config.py      # Logging configuration (file-based)
│   │
│   ├── application/
│   │   ├── interfaces/            # Abstract interfaces
│   │   │   ├── decomposition_agent.py    # DecompositionAgent interface
│   │   │   ├── search_agent.py           # SearchAgent interface
│   │   │   ├── answer_agent.py           # AnswerAgent interface
│   │   │   ├── guardrail_agent.py        # GuardrailAgent interface
│   │   │   ├── telemetry.py              # TelemetryLogger interface
│   │   │   └── evaluation.py             # EvaluationMetrics interface
│   │   └── workflow.py            # WorkflowOrchestrator with LangGraph
│   │
│   ├── infrastructure/
│   │   ├── agents/                # Agent implementations
│   │   │   ├── decomposition_agent.py    # Query decomposition
│   │   │   ├── offline_search.py         # FAISS-based semantic search
│   │   │   ├── online_search.py          # Tavily web search
│   │   │   ├── answer_generation.py      # LLM answer generation
│   │   │   └── guardrail.py              # Safety validation
│   │   ├── kb_manager.py          # Knowledge base & agentic chunking
│   │   ├── online_search.py       # Tavily API integration
│   │   ├── document_fetcher.py    # Download & manage documentation
│   │   └── telemetry.py           # TelemetryLogger & EvaluationMetrics implementations
│   │
│   └── presentation/
│       └── cli.py                 # CLI interface (typer + rich)
│
├── data/
│   ├── sources.json               # Source file hashes for change detection
│   ├── langgraph-docs/            # Downloaded LangGraph documentation
│   ├── langchain-docs/            # Downloaded LangChain documentation
│   ├── kb_chunks.json             # Agentic chunks with metadata (generated)
│   └── faiss_index/               # FAISS vector store (generated)
│
├── logs/
│   ├── app.log                    # Application logs (rotating file handler)
│   ├── telemetry.jsonl            # Inference traces (one JSON per line)
│   └── eval_metrics.json          # Quality metrics
│
├── main.py                        # Entry point
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
├── docker-compose.yml             # Docker Compose setup
├── .env.example                   # Environment variable template
└── README.md                      # This file
```

## Configuration

Edit `.env` file to configure the system:

```env
# API Keys (Required)
GOOGLE_API_KEY=your_gemini_api_key              # Google Gemini API key for LLM
TAVILY_API_KEY=your_tavily_api_key              # Tavily API key for online search

# LLM Configuration
LLM_MODEL=gemini-2.5-flash-lite                 # Model to use for generation & validation
EMBEDDING_MODEL=all-MiniLM-L6-v2                # Model for semantic embeddings

# RAG Parameters (Offline Mode)
TOP_K_CHUNKS=5                                  # Number of knowledge base chunks to retrieve
SIMILARITY_THRESHOLD=0.3                        # Minimum similarity score to return results
CHUNK_SIZE_FOR_AGENTIC_CHUNKING=2000            # Character size for semantic chunks

# Search Parameters (Online Mode)
MAX_ONLINE_SEARCH_RESULTS=5                     # Number of web results to retrieve
MIN_SOURCE_AGREEMENT_RATIO=0.6                  # Minimum relevance threshold for sources

# Paths
TELEMETRY_LOG_PATH=data/telemetry.jsonl         # Where to save inference traces
EVAL_METRICS_PATH=data/eval_metrics.json        # Where to save evaluation metrics
```

**Important Version Information:**
- **Python**: 3.11+ required
- **LLM Model**: Currently using `gemini-2.5-flash-lite` (highly optimized for cost/speed)
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional, fast & accurate)
- **Vector Database**: FAISS (CPU-based, no external dependencies)
- **Web Search**: Tavily API v1

## Workflow & State Management

The system uses **LangGraph** for orchestration with a stateful graph that processes queries:

```
User Query (+ mode: offline/online)
  ↓
[Query Decomposition] → LLM breaks complex question into subquestions
  ├─ Identifies dependencies between subquestions
  └─ Orders subquestions for logical execution
  ↓
[Process Subquestions] → Loop through each subquestion
  │
  ├─ If Offline Mode:
  │   ├→ [Offline Search Agent] → FAISS semantic similarity search
  │   └→ Retrieve top-K chunks with metadata & sources
  │
  └─ If Online Mode:
      ├→ [Online Search Agent] → Tavily web search
      ├→ [Result Validation] → LLM credibility assessment
      └→ Retrieve top-5 results with relevance scores
  ↓
[Answer Generation Agent]
  ├─ Takes question + retrieved evidence
  ├─ Uses conversation history for context
  ├─ Includes dependent answers for synthesis questions
  ├─ Computes confidence: 60% evidence quality + 40% faithfulness
  └─ Returns answer with citations
  ↓
[Guardrail Agent] → Safety validation
  ├─ Checks for harmful/unsafe content
  └─ Rejects if unsafe, keeps answer if safe
  ↓
[Telemetry & Metrics]
  ├─ Logs full inference trace
  ├─ Computes quality metrics (faithfulness, coverage, etc.)
  └─ Saves to data/telemetry.jsonl
  ↓
[Format & Return]
  ├─ Answer (displayed to user)
  ├─ Sources (displayed to user)
  └─ Metadata (logged only, not shown to user)
```

### State Management with LangGraph

The workflow maintains a `WorkflowState` dict that flows through the graph:

- `query`: Original user question
- `mode`: "offline" or "online"
- `decomposition`: Parsed subquestions with dependencies
- `question_answers`: Map of subquestion_id → (query, answer, evidence)
- `final_answer`: The answer to return to user
- `all_sources`: Aggregated citations from all subquestions
- `conversation_history`: Previous messages for multi-turn conversations

## Evaluation Metrics

The system automatically computes quality metrics:

- **Faithfulness**: How well answer is grounded in provided context (0.0-1.0)
- **Unsupported Claims**: Count of claims not in context
- **Retrieval Coverage**: Whether retrieved chunks contain relevant info (0.0-1.0)
- **Source Freshness**: Recency of online sources (0.0-1.0)

View metrics:
```bash
python -m main stats
```

## Development

### Architecture: Onion Pattern with Dependency Injection

The system follows strict onion architecture:
- **Core**: Domain models (Pydantic), no dependencies
- **Application**: Abstract interfaces, business logic, orchestration
- **Infrastructure**: Concrete implementations, external API calls
- **Presentation**: CLI interface

### Adding Custom Agents

To add a new custom agent:

1. **Define interface** in `app/application/interfaces/my_agent.py`:
```python
from abc import ABC, abstractmethod
from typing import Any

class MyAgent(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass
```

2. **Create implementation** in `app/infrastructure/agents/my_agent.py`:
```python
from app.application.interfaces import MyAgent as MyAgentInterface

class MyAgent(MyAgentInterface):
    def process(self, data: Any) -> Any:
        # Your implementation
        return result
```

3. **Use dependency injection** in workflow:
```python
from app.application import WorkflowOrchestrator

workflow = WorkflowOrchestrator(
    my_agent=MyAgent()  # Pass custom agent
)
```

### Extending RAG

**Modify Chunking Strategy**:
- Edit `app/infrastructure/kb_manager.py` → `AgenticChunker._chunk_documents()`
- Adjust `CHUNK_SIZE_FOR_AGENTIC_CHUNKING` in `.env`

**Change Embedding Model**:
- Edit `app/infrastructure/kb_manager.py` → `VectorStoreManager.__init__()`
- Update `EMBEDDING_MODEL` in `.env`

**Adjust Search Parameters**:
- `TOP_K_CHUNKS`: Number of chunks to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.3)

### Custom Quality Metrics

Add new metrics to `app/infrastructure/telemetry.py` → `EvaluationMetrics` class:

```python
def compute_custom_metric(self, answer: str, context: str) -> float:
    """Your custom metric implementation."""
    # LLM evaluation or heuristic-based
    return score  # 0.0-1.0
```

### Testing with Dependency Injection

Example: Testing with a mock agent:

```python
from app.application import WorkflowOrchestrator
from unittest.mock import Mock

mock_agent = Mock(spec=SearchAgent)
mock_agent.search.return_value = EvidencePack(...)

workflow = WorkflowOrchestrator(offline_agent=mock_agent)
result = workflow.run("test query", mode="offline")
```

## Troubleshooting

**KB Build Fails**
- Verify internet connection
- Check API key in .env
- Try with `--force` flag

**No Results Found**
- Increase `TOP_K_CHUNKS` in .env
- Lower `SIMILARITY_THRESHOLD`
- Try online mode to verify working system

**Slow Response**
- LLM inference takes time (30-60s typical)
- Online search adds network latency
- Consider offline-only mode for speed

**Docker Issues**
- Ensure .env file exists before running
- Check volume permissions: `docker-compose run langgraph-helper build-kb`
- Verify API keys are in .env file

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- Code follows onion architecture pattern
- Type hints on all functions
- Docstrings for public APIs
- Tests for new features