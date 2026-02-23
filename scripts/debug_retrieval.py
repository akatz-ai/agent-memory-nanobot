#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, "/data/projects/agent-memory")

from agent_memory.embeddings import LiteLLMEmbedding
from agent_memory.store import MemoryGraphStore


DB_PATH = "/data/projects/nanobot-deploy-test/.nanobot/memory_graph"


async def main() -> int:
    key = os.environ.get("OPENAI_API_KEY")
    print("OPENAI_API_KEY:", "set" if key else "missing")

    embedding = LiteLLMEmbedding(
        model="openai/text-embedding-3-small",
        dimensions=512,
    )
    store = MemoryGraphStore(db_path=DB_PATH, embedding=embedding)
    await store.initialize()

    queries = [
        "Alex timezone",
        "Codex patterns",
        "ComfyGit version",
        "core values Devius",
        "scheduled report times",
    ]

    try:
        for q in queries:
            print(f"\nQuery: {q}")
            try:
                results = await store.recall(
                    query=q,
                    mode="hybrid",
                    max_results=5,
                    graph_depth=1,
                )
            except Exception as exc:
                print(f"  ERROR: {exc}")
                continue

            print(f"  Results: {len(results)}")
            for row in results[:5]:
                print(
                    f"  - type={row.get('memory_type')} id={row.get('id')} "
                    f"content={str(row.get('content', ''))[:120].replace(chr(10), ' ')}"
                )
    finally:
        await store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
