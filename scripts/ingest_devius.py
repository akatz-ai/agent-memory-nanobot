#!/usr/bin/env python3
"""Ingest Devius memory files into agent-memory graph storage."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure local repos are importable when running this script directly.
sys.path.insert(0, "/data/projects/agent-memory")
sys.path.insert(0, "/data/projects/nanobot")
sys.path.insert(0, "/data/projects/agent-memory-nanobot")

from agent_memory.embeddings import LiteLLMEmbedding
from agent_memory.ingestion import MemoryIngestionAgent
from agent_memory.store import MemoryGraphStore
from agent_memory_nanobot.providers import NanobotLLMAdapter
from nanobot.providers.litellm_provider import LiteLLMProvider


MEMORY_ROOT = os.path.expanduser("~/.clawbro/agents/devius")
DB_PATH = "/data/projects/nanobot-deploy-test/.nanobot/memory_graph"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "openai/gpt-4o-mini"
FILES: list[tuple[str, str]] = [
    ("SOUL.md", "identity"),
    ("IDENTITY.md", "identity"),
    ("USER.md", "identity"),
    ("memory/MEMORY.md", "fact"),
    ("memory/codex-tmux-workflow.md", "fact"),
    ("memory/comfygit-registry-data.md", "fact"),
    ("memory/release-process.md", "fact"),
    ("memory/structured/agent-platform-strategy.md", "decision"),
    ("memory/daily/2026-02-15.md", "event"),
    ("memory/daily/2026-02-16.md", "event"),
    ("memory/daily/2026-02-17.md", "event"),
    ("memory/daily/2026-02-18.md", "event"),
    ("memory/daily/2026-02-19.md", "event"),
    ("memory/daily/2026-02-20.md", "event"),
    ("memory/daily/2026-02-21.md", "event"),
    ("memory/daily/2026-02-22.md", "event"),
]


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


async def main() -> int:
    _require_env("OPENAI_API_KEY")

    provider = LiteLLMProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        default_model=LLM_MODEL,
        provider_name="openai",
    )
    llm = NanobotLLMAdapter(provider=provider, default_model=LLM_MODEL)

    embedding = LiteLLMEmbedding(model=EMBEDDING_MODEL, dimensions=512)
    store = MemoryGraphStore(db_path=DB_PATH, embedding=embedding)
    ingestion = MemoryIngestionAgent(
        store=store,
        llm=llm,
        model=LLM_MODEL,
        chunk_size_chars=3500,
    )

    await store.initialize()

    totals = {
        "files_attempted": 0,
        "files_processed": 0,
        "chunks_processed": 0,
        "memories_added": 0,
        "edges_created": 0,
        "files_missing": 0,
    }

    root = Path(MEMORY_ROOT)
    print(f"Memory root: {root}")
    print(f"DB path: {DB_PATH}")

    try:
        for rel_path, memory_type in FILES:
            totals["files_attempted"] += 1
            file_path = root / rel_path

            if not file_path.exists():
                totals["files_missing"] += 1
                print(f"SKIP {rel_path} (missing)")
                continue

            result = await ingestion.ingest_file(
                file_path=file_path,
                source=f"devius:{memory_type}:{rel_path}",
                peer_key="devius",
            )

            totals["files_processed"] += 1
            totals["chunks_processed"] += result.get("chunks_processed", 0)
            totals["memories_added"] += result.get("memories_added", 0)
            totals["edges_created"] += result.get("edges_created", 0)

            print(
                "OK "
                f"{rel_path} "
                f"chunks={result.get('chunks_processed', 0)} "
                f"added={result.get('memories_added', 0)} "
                f"edges={result.get('edges_created', 0)}"
            )

        stats = await store.stats()

        print("\nIngestion totals")
        print(f"  files_attempted: {totals['files_attempted']}")
        print(f"  files_processed: {totals['files_processed']}")
        print(f"  files_missing:   {totals['files_missing']}")
        print(f"  chunks:          {totals['chunks_processed']}")
        print(f"  memories_added:  {totals['memories_added']}")
        print(f"  edges_created:   {totals['edges_created']}")

        print("\nStore stats")
        print(f"  total_memories: {stats.get('total_memories', 0)}")
        print(f"  total_edges:    {stats.get('total_edges', 0)}")
        print(f"  forgotten:      {stats.get('forgotten', 0)}")
        print(f"  by_type:        {stats.get('by_type', {})}")
        return 0
    finally:
        await store.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
