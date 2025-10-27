# Memory & Checkpointing

This project supports both in-memory and Azure Cosmos DB-backed persistence for:
- Long-term memory (user preferences, LTM)
- Graph checkpointing (LangGraph thread state)

## Modules
- `lib/memory/ltm_cosmos.py` — Cosmos-backed LTM
- `lib/memory/checkpointer_cosmos.py` — Cosmos-backed LangGraph checkpointer

## Modes

- Default (no Cosmos):
  - LTM: In-memory fallback (created on demand inside app)
  - Checkpointer: In-memory (ephemeral)
  - Set `CHECKPOINTER=memory` (default)

- Cosmos-backed:
  - LTM: Cosmos container (recommended for multi-session persistence)
  - Checkpointer: Cosmos container for thread state
  - Set `CHECKPOINTER=cosmos`

## Required Environment (Cosmos mode)

- `COSMOS_ENDPOINT` — e.g., `https://<account>.documents.azure.com:443/`
- `COSMOS_KEY` — primary key
- `COSMOS_DATABASE_NAME` — database name (e.g., `master`)
- `COSMOS_LTM_CONTAINER` — user prefs container (e.g., `chats`)
- `COSMOS_CHECKPOINTS_CONTAINER` — checkpoints container (e.g., `graph_checkpoints`)
- `COSMOS_PARTITION_KEY` — partition key for LTM (e.g., `/user_id`)

## Containers

- LTM container schema (typical):
  - Partition key: `/user_id`
  - Document fields:
    - `user_id`: string
    - `preferred_tone`: string
    - `platform_defaults`: object

- Checkpoints container schema:
  - Partition key: `/thread_id`
  - TTL enabled (optional) → controls session expiry

## Switching Modes

- Use in-memory (dev):
  - `CHECKPOINTER=memory`

- Use Cosmos (prod-like):
  - `CHECKPOINTER=cosmos`
  - Set all Cosmos env vars above

