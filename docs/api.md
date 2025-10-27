# API Reference

FastAPI service exposing endpoints to create sessions and generate platform-specific posts.

Base URL (local): http://localhost:8000

## Endpoints

- GET /health
  - Returns component status and version.
  - 200 OK body:
    {
      "status": "healthy|degraded",
      "message": "...",
      "version": "0.1.0",
      "components": {
        "retriever": "ok|not_initialized|error",
        "ltm": "ok|error",
        "checkpointer": "ok (memory|cosmos)",
        "tracer": "ok|disabled|error"
      }
    }

- POST /sessions
  - Body:
    {
      "user_id": "user-123",
      "platform": "linkedin"
    }
  - 201 Created body:
    {
      "session_id": "session-user-123-<id>",
      "user_id": "user-123",
      "platform": "linkedin",
      "message": "Session created successfully"
    }

- POST /posts:generate
  - Body:
    {
      "session_id": "session-user-123-<id>",
      "topic": "The impact of transformer models on NLP",
      "platform": "linkedin",
      "tone": "insightful"
    }
  - 200 OK body:
    {
      "post_markdown": "...",
      "scores": {"faithfulness": 0.91, "answer_relevancy": 0.88},
      "trace_url": null | "http://localhost:3000/...",
      "session_id": "...",
      "refinement_count": 0
    }
  - 424 Failed Dependency: FAISS index missing or retriever error
  - 500 Internal Server Error: Unhandled exceptions

## Quick start

1) Activate env and set .env
2) Build FAISS index:
   python scripts/build_faiss_index.py
3) Run API:
   uvicorn api.main:app --reload --port 8000
4) Open http://localhost:8000/docs

## Curl examples

# Create session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-123", "platform": "linkedin"}'

# Generate post (replace SESSION_ID)
curl -X POST http://localhost:8000/posts:generate \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "topic": "Transformers in NLP", "platform": "linkedin", "tone": "insightful"}'

## Environment variables (minimum)

- AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
- AZURE_OPENAI_CHAT_DEPLOYMENT
- AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT
- If embeddings are on a separate resource:
  - AZURE_OPENAI_EMBEDDINGS_ENDPOINT
  - AZURE_OPENAI_EMBEDDINGS_API_KEY
  - AZURE_OPENAI_EMBEDDINGS_API_VERSION

## Notes

- Requires a built FAISS index at data/faiss_index/ (see build script).
- Tracing to Langfuse enabled when LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY are set.
