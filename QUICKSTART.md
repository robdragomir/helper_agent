# Quick Start Guide

## 1. Setup (5 minutes)

### Prerequisites
- Python 3.11+
- API Keys:
  - Google Gemini API Key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
  - Tavily API Key (get from [Tavily](https://tavily.com/)) - optional for online search

### Install

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env and add your API keys
```

## 2. Build Knowledge Base (2-3 minutes)

```bash
python -m main build-kb
```

This will:
- Download latest LangGraph documentation
- Intelligently chunk it using LLM
- Create vector embeddings
- Store in `data/faiss_index/`

## 3. Ask Questions

### Single Question
```bash
python -m main ask "How do I create a graph?"
```

### With Mode Selection
```bash
# Offline only (fast, knowledge base)
python -m main ask "How does state work?" --mode offline

# Online only (current info)
python -m main ask "Latest features?" --mode online

# Combined (best of both)
python -m main ask "How do graphs compare?" --mode combined
```

### Interactive Mode
```bash
python -m main interactive
```

## 4. View Statistics

```bash
python -m main stats
```

Shows:
- Query metrics
- Average confidence scores
- Source freshness
- Quality metrics

## Docker (Optional)

### Quick Setup
```bash
# Build
docker-compose build

# Ask a question
docker-compose run langgraph-helper ask "Your question"

# Interactive mode
docker-compose run -it langgraph-helper interactive

# Build KB
docker-compose run langgraph-helper build-kb
```

## Common Issues

**No results found**
- Increase `TOP_K_CHUNKS=10` in `.env`
- Lower `SIMILARITY_THRESHOLD=0.2`
- Try online mode

**KB build fails**
- Check internet connection
- Verify `GOOGLE_API_KEY` in `.env`
- Try: `python -m main build-kb --force`

**Slow responses**
- LLM inference takes 30-60s normally
- Use offline mode for faster responses
- First query is slowest (loading models)

## Architecture at a Glance

```
Your Query
    ↓
Router Agent (decides: offline/online/both)
    ↓
    ├→ Offline: Vector search on KB
    └→ Online: Web search + validation
    ↓
Answer Generation (combines evidence)
    ↓
Safety Check (guardrails)
    ↓
Your Answer (with citations & confidence)
```

## Key Concepts

**Offline Mode**: Uses local knowledge base
- Fast responses
- No internet needed
- Limited to existing docs

**Online Mode**: Web search
- Latest information
- Slower responses
- Validates sources

**Combined**: Both sources
- Best coverage
- Slower
- Highest confidence

## Next Steps

1. **Customize**: Edit prompts in `app/application/agents.py`
2. **Extend**: Add new agents to workflow
3. **Deploy**: Use Docker for production
4. **Monitor**: Check `logs/` for telemetry

## Full Documentation

See `README.md` for comprehensive documentation.