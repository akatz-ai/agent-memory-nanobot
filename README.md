# agent-memory-nanobot

Adapter layer that integrates [agent-memory](https://github.com/akatz-ai/agent-memory) (the standalone graph memory library) with [nanobot](https://github.com/akatz-ai/nanobot) (the AI agent framework).

## What This Does

`agent-memory` is a framework-agnostic memory core — it handles storage, retrieval, consolidation, and ingestion but doesn't know about any specific agent framework.

This package bridges the gap by:

- **Wiring nanobot's LLM providers** into agent-memory's interfaces (so memory operations use the same Claude/OpenAI connection the agent already has)
- **Exposing memory operations as nanobot tools** that agents can call during conversations
- **Managing hybrid memory** — coordinating file-based memory (MEMORY.md / HISTORY.md) with the graph store so both stay in sync

## Architecture

```
┌──────────────────────────────────────────────┐
│  nanobot (agent framework)                   │
│  ├── Agent loop, sessions, Discord/Telegram  │
│  └── LLM providers (Anthropic, OpenAI, etc.) │
└──────────────┬───────────────────────────────┘
               │
┌──────────────▼───────────────────────────────┐
│  agent-memory-nanobot (this package)         │
│  ├── NanobotMemoryModule  — facade/entry     │
│  ├── NanobotLLMAdapter    — provider bridge  │
│  ├── HybridMemoryManager  — file + graph     │
│  └── Tools                — recall, save,    │
│                             forget, ingest,  │
│                             graph, stats     │
└──────────────┬───────────────────────────────┘
               │
┌──────────────▼───────────────────────────────┐
│  agent-memory (core library)                 │
│  ├── MemoryGraphStore     — LanceDB storage  │
│  ├── PreTurnRetriever     — hybrid retrieval │
│  ├── MemoryConsolidator   — extraction/dedup │
│  └── MemoryIngestionAgent — bulk markdown    │
└──────────────────────────────────────────────┘
```

## Components

### `NanobotMemoryModule`

The main entry point. Initializes all agent-memory components using nanobot's runtime config and providers.

```python
from agent_memory_nanobot import NanobotMemoryModule

module = NanobotMemoryModule(
    provider=nanobot_llm_provider,
    workspace=Path("~/.nanobot/workspace/agents/general"),
    config={
        "db_path": "~/.agent-memory/nanobot",
        "embedding": {
            "model": "openai/text-embedding-3-small",
            "dimensions": 512,
        },
    },
)
await module.initialize()
tools = module.get_tools()  # Register these with the agent
```

### `NanobotLLMAdapter`

Translates nanobot's `LLMProvider` interface to agent-memory's `LLM` protocol. Handles model routing — if the main provider is Anthropic Direct but a memory operation needs an OpenAI model, it falls back to litellm automatically.

### `HybridMemoryManager`

Coordinates the dual-storage approach:
- **File-based** (MEMORY.md for long-term facts, HISTORY.md for event logs) — human-readable, git-friendly, the source of truth
- **Graph-based** (LanceDB with vector embeddings + associations) — fast retrieval, semantic search, relationship traversal

Extraction writes to markdown first, then indexes into the graph.

### Tools

Six tools exposed to agents:

| Tool | Description |
|------|-------------|
| `memory_recall` | Semantic/keyword/recency search across the memory graph |
| `memory_save` | Store a new memory with metadata, entities, and associations |
| `memory_forget` | Soft-delete a memory by ID |
| `memory_graph` | Traverse associations from seed memory IDs |
| `memory_ingest` | Bulk ingest raw markdown/text into the graph |
| `memory_stats` | Return graph statistics (node count, edge count, type distribution) |

## Install

```bash
pip install agent-memory-nanobot
```

Or from source:

```bash
pip install -e .
```

Requires both `agent-memory` and `nanobot-ai` as dependencies.

## Configuration

Memory config lives in nanobot's `config.json` under the `memoryGraph` key:

```json
{
  "memoryGraph": {
    "db_path": "~/.agent-memory/nanobot",
    "embedding": {
      "model": "openai/text-embedding-3-small",
      "dimensions": 512,
      "api_key": "sk-..."
    },
    "retrieval": {
      "enabled": true,
      "max_results": 12,
      "max_context_words": 320
    },
    "consolidation": {
      "enabled": true,
      "engine": "hybrid",
      "dedup_threshold": 0.3
    }
  }
}
```

## License

[AGPL-3.0](LICENSE)
