#!/usr/bin/env python3
"""E2E test for nanobot + agent-memory integration."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure local repos are importable when running this script directly.
sys.path.insert(0, "/data/projects/agent-memory")
sys.path.insert(0, "/data/projects/nanobot")
sys.path.insert(0, "/data/projects/agent-memory-nanobot")

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.cli.commands import _make_provider
from nanobot.config.loader import load_config


CONFIG_PATH = Path("/data/projects/nanobot-deploy-test/.nanobot/config.json")
WORKSPACE_PATH = Path("/data/projects/nanobot-deploy-test/.nanobot/workspace")
DB_PATH = "/data/projects/nanobot-deploy-test/.nanobot/memory_graph"

TESTS = [
    ("What timezone is Alex in?", ["PST", "Pacific", "Los Angeles"]),
    ("What's the release process for ComfyGit?", ["version", "changelog", "PR", "main"]),
    ("What Codex patterns should I follow?", ["monitor", "scheduler", "tmux", "never nest"]),
    ("What happened on February 18?", ["Oracle", "CBW", "hybrid", "memory"]),
    ("What's the agent platform strategy?", ["Zeroclaw", "MCP", "portable"]),
    ("What are the scheduled report times?", ["10:00", "10:30", "10:55", "standup"]),
    ("How do I handle unknown node mappings in ComfyGit?", ["community_mapping", "registry"]),
    ("What is Alex's communication style preference?", ["concise", "professional", "warm"]),
    ("What's the current ComfyGit version?", ["v0.3.18"]),
    ("What are my core values as Devius?", ["helpful", "opinion", "resourceful"]),
]


def _memory_graph_config() -> dict:
    return {
        "enabled": True,
        "db_path": DB_PATH,
        "embedding": {
            "model": "text-embedding-3-small",
            "dimensions": 512,
        },
        "retrieval": {
            "enabled": True,
            "model": "openai/gpt-4o-mini",
            "max_context_words": 45,
        },
        "consolidation": {
            "enabled": True,
            "model": "openai/gpt-4o-mini",
            "dedup_threshold": 0.3,
        },
        "ingestion": {
            "model": "openai/gpt-4o-mini",
            "chunk_size_chars": 3500,
        },
    }


async def main() -> int:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Config not found: {CONFIG_PATH}")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required for memory retrieval embeddings")

    config = load_config(config_path=CONFIG_PATH)
    config.agents.defaults.workspace = str(WORKSPACE_PATH)

    memory_graph_config = config.memory_graph or _memory_graph_config()
    memory_graph_config.setdefault("enabled", True)
    memory_graph_config.setdefault("db_path", DB_PATH)

    bus = MessageBus()
    provider = _make_provider(config)

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        brave_api_key=config.tools.web.search.api_key or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        memory_graph_config=memory_graph_config,
    )

    passed = 0
    total = len(TESTS)

    try:
        for idx, (question, expected_keywords) in enumerate(TESTS, start=1):
            response = await agent.process_direct(
                question,
                session_key="cli:e2e-memory",
                channel="cli",
                chat_id="e2e-memory",
            )

            lower_response = response.lower()
            matched = [kw for kw in expected_keywords if kw.lower() in lower_response]
            ok = bool(matched)
            if ok:
                passed += 1

            excerpt = " ".join(response.strip().split())[:220]
            status = "PASS" if ok else "FAIL"
            print(f"[{idx:02d}] {status} | Q: {question}")
            print(f"     matched: {matched if matched else 'none'}")
            print(f"     response: {excerpt}\n")

        print(f"Summary: {passed}/{total} passed")
        return 0 if passed >= 8 else 1
    finally:
        await agent.close_mcp()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
