# 🤖 Agentic Post Generator

A production-ready, multi-agent system for generating platform-specific social media posts using **LangGraph**, **Azure OpenAI**, **FAISS retrieval**, and **Langfuse observability**.

## 🎯 What & Why

This project implements a **multi-agent workflow** that:
- **Plans** the structure and key points for a social media post
- **Writes** a draft using retrieval-augmented generation (RAG) with platform-specific templates
- **Fact-checks** the content using DeepEval metrics (faithfulness & answer relevancy)
- **Refines** the post if quality scores fall below thresholds
- **Traces** every step with Langfuse for full observability

**Why this architecture?**
- ✅ **Modular**: Swap FAISS → Azure AI Search with minimal code changes
- ✅ **Persistent**: Multi-session support via LangGraph checkpointer + Cosmos DB
- ✅ **Observable**: Full tracing with Langfuse, including custom metrics

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          FastAPI Service                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ POST /sessions│  │POST /posts:  │  │ GET /health        │   │
│  │ (create/resume)│  │   generate   │  │                    │   │
│  └──────┬───────┘  └──────┬───────┘  └────────────────────┘   │
│         │                  │                                     │
│         └──────────────────┼─────────────────────────────────┐  │
│                            ▼                                  │  │
│              ┌─────────────────────────────┐                 │  │
│              │   LangGraph Workflow        │                 │  │
│              │                             │                 │  │
│              │  ┌──────┐    ┌────────┐    │                 │  │
│              │  │Planner├───►│ Writer │    │                 │  │
│              │  └──────┘    └───┬────┘    │                 │  │
│              │                  │          │                 │  │
│              │            ┌─────▼─────┐   │                 │  │
│              │            │Fact-Checker│   │                 │  │
│              │            └─────┬─────┘   │                 │  │
│              │                  │          │                 │  │
│              │              ┌───▼────┐    │                 │  │
│              │              │ Router │    │                 │  │
│              │              └───┬────┘    │                 │  │
│              │                  │          │                 │  │
│              │          ┌───────▼──────┐  │                 │  │
│              │          │ Score >= 0.8?│  │                 │  │
│              │          └┬─────────────┬┘  │                 │  │
│              │           │Yes        No│   │                 │  │
│              │         ┌─▼──┐      ┌──▼┐  │                 │  │
│              │         │END │      │Loop│  │                 │  │
│              │         └────┘      └──┬┘  │                 │  │
│              │                        │   │                 │  │
│              │         ┌──────────────┘   │                 │  │
│              │         │  (max 1 time)    │                 │  │
│              │         └─────────────────►│                 │  │
│              └─────────────────────────────┘                 │  │
│                                                              │  │
└──────────────────────────────────────────────────────────────┘  │
                                                                  │
┌─────────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  Azure OpenAI       │  │  FAISS Index     │  │  Cosmos DB   │ │
│  (Chat + Embeddings)│  │  (Retrieval)     │  │  (Memory)    │ │
│                     │  │                  │  │              │ │
│  • Planner LLM     │  │  • 20-30 papers  │  │  • LTM       │ │
│  • Writer LLM      │  │  • Save/Load     │  │  • Checkpts  │ │
│  • Checker LLM     │  │  • k=5 search    │  │  • TTL       │ │
└─────────────────────┘  └──────────────────┘  └──────────────┘ │
                                                                  │
┌─────────────────────────────────────────────────────────────────┘
│  Langfuse (Observability)
│  • Trace every graph run
│  • Attach scores (faithfulness, answer_relevancy)
│  • View spans: planner, writer, fact_checker, router
└─────────────────────────────────────────────────────────────────
```

---

## 📊 Agent Flow

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 1. PLANNER                                          │
│    • Load user LTM (tone, platform prefs)           │
│    • Retrieve relevant context from FAISS           │
│    • Create structured outline                      │
└──────────────────┬──────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│ 2. WRITER                                           │
│    • Load platform template (LinkedIn/Twitter)      │
│    • Generate draft using outline + context        │
│    • Apply tone from LTM                            │
└──────────────────┬──────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│ 3. FACT-CHECKER (DeepEval)                          │
│    • Compute Faithfulness (vs retrieved context)    │
│    • Compute Answer Relevancy (vs user request)     │
│    • Attach scores to Langfuse trace                │
└──────────────────┬──────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│ 4. ROUTER (Conditional Logic)                       │
│    • If min(scores) >= 0.80 → END                   │
│    • Else → back to WRITER (max 1 loop)             │
└──────────────────┬──────────────────────────────────┘
                   ▼
              Final Post
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker** & **Docker Compose** (for local dev with Langfuse)
- **Azure OpenAI** deployment (chat + embeddings models)
- **Azure Cosmos DB** account (optional for production; defaults to in-memory)

### 1. Clone & Setup

```bash
# Clone the repository
cd agentic-post-gen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
# Or for dev dependencies:
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash

# Edit .env with your credentials:
- AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
- AZURE_OPENAI_CHAT_DEPLOYMENT
- AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT
- (If embeddings are on a different resource) AZURE_OPENAI_EMBEDDINGS_ENDPOINT, AZURE_OPENAI_EMBEDDINGS_API_KEY, AZURE_OPENAI_EMBEDDINGS_API_VERSION
- (Optional) DEEPEVAL_MODEL (e.g., gpt-4o)
- (Optional) COSMOS_ENDPOINT, COSMOS_KEY
- (Optional) LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

### 3. Build FAISS Index

```bash
# Place 20-30 arXiv papers (PDFs or text files) in data/papers/
# Then build the FAISS index:
python scripts/build_faiss_index.py
```

This will create embeddings and save the index to `data/faiss_index/`.

### 4. Run Locally with Docker Compose

```bash
# Start Langfuse (observability) locally
docker compose -f infra/docker-compose.langfuse.yml up -d

# Langfuse UI: http://localhost:3000

# Start API separately
uvicorn api.main:app --reload --port 8000
```

**Or run API directly:**

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Test the API


**Create a session:**
```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-123", "platform": "linkedin"}'
```

**Generate a post:**
```bash
curl -X POST http://localhost:8000/posts:generate \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<session-id-from-above>",
    "topic": "The impact of transformer models on NLP",
    "platform": "linkedin",
    "tone": "insightful"
  }'
```

---

## 📁 Project Structure

```
agentic-post-gen/
├── agents/                    # Single-task agent implementations
│   ├── planner.py            # Outline & research planning
│   ├── writer.py             # Platform-specific content generation
│   └── fact_checker.py       # DeepEval scoring & validation
├── api/                       # FastAPI application
│   ├── main.py               # Routes & endpoints
│   └── schemas.py            # Request/response models
├── lib/                       # Reusable adapters & utilities
│   ├── retriever/            # Retrieval interfaces
│   │   ├── base.py           # Abstract base class
│   │   ├── faiss_store.py    # FAISS implementation
│   │   └── azure_ai_search.py # Azure AI Search stub (TODO)
│   ├── memory/               # Persistence layer
│   │   ├── ltm_cosmos.py     # Long-term memory (user prefs)
│   │   └── checkpointer_cosmos.py # LangGraph checkpointer
│   └── observability/        # Tracing & metrics
│       └── langfuse_cb.py    # Langfuse callback helper
├── workflows/                 # LangGraph orchestration
│   ├── state.py              # Graph state schema
│   ├── build_graph.py        # Graph construction & compilation
│   └── run.py                # Graph execution helpers
├── prompts/                   # LLM prompt templates
│   ├── planner_system.txt
│   ├── writer_linkedin.txt
│   ├── writer_twitter.txt
│   └── checker_instructions.txt
├── data/                      # Data storage
│   ├── papers/               # Source documents (PDFs/text)
│   └── faiss_index/          # Saved FAISS index
├── scripts/                   # Utilities
│   ├── build_faiss_index.py  # Index examples
├── tests/                     # Test suite
│   ├── test_retriever.py
│   ├── test_graph_smoke.py
│   └── test_api_smoke.py
├── infra/                     # Deployment & infrastructure
│   ├── docker-compose.langfuse.yml
│   ├── docker-compose.local.yml
│   ├── Dockerfile
│   └── azure-deploy.md
├── docs/                      # Module documentation
│   ├── retriever.md
│   ├── memory.md
│   ├── graph.md
│   └── api.md
├── .env.example               # Environment template
├── pyproject.toml             # Python project config
├── README.md                  # This file
└── SECURITY.md                # Security guidelines
```

---

## 🔧 Configuration

All configuration is via environment variables :

| Variable | Description | Default |
|----------|-------------|---------|
| `CHECKPOINTER` | Checkpointer type: `memory` or `cosmos` | `memory` |
| `FAITHFULNESS_THRESHOLD` | Minimum faithfulness score (0.0-1.0) | `0.80` |
| `ANSWER_RELEVANCY_THRESHOLD` | Minimum relevancy score (0.0-1.0) | `0.80` |
| `MAX_REFINEMENT_LOOPS` | Max refinement attempts | `1` |
| `FAISS_INDEX_PATH` | Path to FAISS index directory | `./data/faiss_index` |
| `SESSION_TTL_SECONDS` | Session expiration time | `5184000` (60 days) |
| `DEEPEVAL_MODEL` | Model id used by DeepEval metrics (falls back to chat deployment) | `gpt-4o` |

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_retriever.py -v
```

## 📚 Documentation

Detailed module documentation is available in the `docs/` folder:

- **[Retrieval System](docs/retriever.md)** - FAISS & Azure AI Search adapters
- **[Memory Management](docs/memory.md)** - LTM & checkpointer implementation
- **[Graph Workflow](docs/graph.md)** - LangGraph state & orchestration
- **[API Reference](docs/api.md)** - Endpoints & schemas


## 📊 Observability

### Langfuse Integration

Every graph execution is traced in Langfuse with:
- **Spans**: One per node (planner, writer, fact_checker, router)
- **Scores**: Faithfulness & answer relevancy metrics
- **Metadata**: User ID, session ID, platform, topic

Access traces at `http://localhost:3000` (local) or your Langfuse Cloud URL.

### Example Trace Structure

```
Trace: generate_post_linkedin_user123
  ├─ Span: planner (duration: 1.2s)
  ├─ Span: writer (duration: 2.5s)
  ├─ Span: fact_checker (duration: 0.8s)
  │   ├─ Score: faithfulness = 0.92
  │   └─ Score: answer_relevancy = 0.87
  └─ Span: router (duration: 0.1s)
```

---



## 📄 License

See LICENSE file for details

---


