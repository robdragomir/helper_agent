# LangGraph Helper Agent

A sophisticated multi-agent system for answering LangGraph documentation questions using offline RAG, online search, and agentic orchestration with LangGraph.

## Architecture

This project follows **onion architecture** pattern:

```
┌─────────────────────────────────────────────┐
│   Presentation Layer (CLI)                  │
│   - User interface with typer               │
│   - Output formatting with rich             │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│   Application Layer                         │
│   - Router Agent (decide search mode)       │
│   - Offline Search Agent                    │
│   - Online Search Agent                     │
│   - Answer Generation Agent                 │
│   - Guardrail Agent (validation)            │
│   - LangGraph Workflow Orchestration        │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│   Infrastructure Layer                      │
│   - KB Manager (agentic chunking)           │
│   - Vector Store (FAISS)                    │
│   - Online Search (Tavily)                  │
│   - Telemetry & Evaluation                  │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│   Core Layer                                │
│   - Pydantic Models                         │
│   - Configuration                           │
└─────────────────────────────────────────────┘
```

## Features

### Multi-Mode Search
- **Offline Mode**: Uses local RAG with agentic chunking
- **Online Mode**: Web search with source validation and recency scoring
- **Combined Mode**: Merges results from both with intelligent reranking

### Intelligent Agents
- **Router Agent**: Automatically decides optimal search strategy
- **Offline Search Agent**: Semantic search on LangGraph documentation
- **Online Search Agent**: Web search with source credibility validation
- **Answer Generation Agent**: Creates coherent answers from evidence
- **Guardrail Agent**: Validates answers for safety and accuracy

### Advanced RAG
- **Agentic Chunking**: LLM-driven document chunking based on semantic boundaries
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: Sentence transformers for semantic understanding
- **Knowledge Base Versioning**: Automatic detection and update of documentation

### Telemetry & Evaluation
- **Request Logging**: Full inference traces with routing decisions
- **Quality Metrics**:
  - Faithfulness (groundedness in context)
  - Unsupported claims detection
  - Retrieval coverage
  - Source freshness

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

**Ask a question (auto-mode)**
```bash
python -m main ask "How do I create a basic graph?"
```

**Ask with specific mode**
```bash
# Offline only
python -m main ask "How does state work?" --mode offline

# Online only
python -m main ask "What are the latest LangGraph features?" --mode online

# Combined
python -m main ask "How do graphs compare to other frameworks?" --mode combined
```

**Interactive mode**
```bash
python -m main interactive
```

**Build knowledge base**
```bash
# Build KB from remote docs
python -m main build-kb

# Force rebuild even if up to date
python -m main build-kb --force
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
│   │   ├── models.py          # Pydantic domain models
│   │   └── config.py          # Configuration management
│   ├── infrastructure/
│   │   ├── kb_manager.py      # Knowledge base & agentic chunking
│   │   ├── online_search.py   # Web search & source validation
│   │   └── telemetry.py       # Logging & evaluation metrics
│   ├── application/
│   │   ├── agents.py          # Multi-agent implementations
│   │   └── workflow.py        # LangGraph orchestration
│   └── presentation/
│       └── cli.py             # CLI interface
├── data/
│   ├── langgraph-docs.txt     # Downloaded LangGraph documentation
│   ├── kb_chunks.json         # Agentic chunks (generated)
│   └── faiss_index/           # Vector store (generated)
├── logs/
│   ├── telemetry.jsonl        # Inference traces
│   └── eval_metrics.json      # Quality metrics
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Docker Compose setup
└── .env.example              # Environment variable template
```

## Configuration

Edit `.env` file to configure:

```env
# API Keys
GOOGLE_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key

# LLM Configuration
LLM_MODEL=gemini-pro
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG Parameters
TOP_K_CHUNKS=5                    # Chunks to retrieve
SIMILARITY_THRESHOLD=0.3          # Filter threshold
CHUNK_SIZE_FOR_AGENTIC_CHUNKING=2000

# Search Parameters
MAX_ONLINE_SEARCH_RESULTS=5
MIN_SOURCE_AGREEMENT_RATIO=0.6
```

## Workflow

```
Query
  ↓
[Router Agent] → Decides: offline | online | both
  ↓
  ├→ [Offline Search] → Semantic search on KB
  │   ↓
  │   [Vector Search] → Top-K chunks from FAISS
  │
  └→ [Online Search] → Web search + validation
      ↓
      [Source Validation] → LLM-based credibility check
      ↓
      [Result Reranking] → Combined score ranking
  ↓
[Answer Generation Agent] → Creates answer from evidence
  ↓
[Guardrail Agent] → Validates for safety
  ↓
[Telemetry] → Logs trace + computes metrics
  ↓
Answer to User
```

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

### Adding Custom Agents

1. Create agent class in `app/application/agents.py`
2. Inherit from base agent pattern
3. Implement search/process methods
4. Add to workflow in `app/application/workflow.py`

### Extending RAG

- Modify chunking in `AgenticChunker.chunk_document()`
- Adjust similarity threshold in `config.py`
- Add custom embedding models to `VectorStoreManager`

### Custom Evaluations

Add metrics in `EvaluationMetrics` class in `app/infrastructure/telemetry.py`

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