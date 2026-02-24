from __future__ import annotations

import pytest

from agent_memory_nanobot.hybrid import HybridMemoryManager


class _FakeLLM:
    async def complete(self, messages, model=None, temperature=0.0, max_tokens=500):
        return "# MEMORY\n\n- Keep release decisions and project facts."


class _FakeConsolidator:
    def __init__(self):
        self.llm = _FakeLLM()
        self.model = "claude-haiku-4-5"
        self._extracted = [
            {
                "content": "Nanobot switched to hybrid memory pipeline.",
                "entry_type": "decision",
                "context_tag": "memory redesign",
                "importance": 0.9,
                "entities": ["Nanobot", "hybrid memory"],
            },
            {
                "content": "Daily history files are now the source audit log.",
                "entry_type": "fact",
                "context_tag": "history format",
                "importance": 0.8,
                "entities": ["daily history"],
            },
        ]

    async def extract_from_messages(self, messages, peer_key=None):
        return list(self._extracted)

    async def index_extracted(self, extracted, peer_key=None, source_session=None, source="consolidation"):
        return {"added": len(extracted), "updated": 0, "edges_created": 1}


@pytest.mark.asyncio
async def test_compact_writes_daily_history_and_indexes(tmp_path):
    manager = HybridMemoryManager(
        workspace=tmp_path,
        store=object(),  # store usage is delegated to consolidator
        consolidator=_FakeConsolidator(),
        config={"consolidation": {"model": "claude-haiku-4-5"}},
    )
    messages = [
        {"role": "user", "content": "message 1", "timestamp": "2026-02-23T14:30:01"},
        {"role": "assistant", "content": "message 2", "timestamp": "2026-02-23T14:30:05"},
    ]

    result = await manager.compact(
        session_key="telegram:8554401569",
        messages=messages,
        start_index=0,
        end_index=2,
    )

    assert result.messages_processed == 2
    assert result.memories_indexed == 2
    assert result.edges_created == 1
    assert result.history_file.name == "2026-02-23.md"

    text = result.history_file.read_text(encoding="utf-8")
    assert "## 14:30 — Compaction (telegram:8554401569)" in text
    assert "- [decision] Nanobot switched to hybrid memory pipeline. (context: memory redesign)" in text
    assert "- [fact] Daily history files are now the source audit log. (context: history format)" in text

    memory_md = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "Keep release decisions" in memory_md


@pytest.mark.asyncio
async def test_compact_idempotent_history_section(tmp_path):
    manager = HybridMemoryManager(
        workspace=tmp_path,
        store=object(),
        consolidator=_FakeConsolidator(),
        config={},
    )
    messages = [
        {"role": "user", "content": "message 1", "timestamp": "2026-02-23T18:45:01"},
        {"role": "assistant", "content": "message 2", "timestamp": "2026-02-23T18:45:05"},
    ]

    await manager.compact("telegram:8554401569", messages, 0, 2)
    await manager.compact("telegram:8554401569", messages, 0, 2)

    history_file = tmp_path / "memory" / "history" / "2026-02-23.md"
    text = history_file.read_text(encoding="utf-8")
    assert text.count("## 18:45 — Compaction (telegram:8554401569)") == 1
