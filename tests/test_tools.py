from __future__ import annotations

import json

import pytest

from agent_memory_nanobot import NanobotMemoryModule
from agent_memory_nanobot.tools import MemoryRecallTool, MemorySaveTool


class _StoreWithAssertion:
    def __init__(self, memory_file):
        self.memory_file = memory_file
        self.saved = False

    async def save(self, **kwargs):
        text = self.memory_file.read_text(encoding="utf-8")
        assert "Hybrid mode keeps files as source of truth." in text
        self.saved = True
        return "mem-1"


@pytest.mark.asyncio
async def test_memory_save_writes_to_memory_md_before_graph_index(tmp_path):
    memory_file = tmp_path / "memory" / "MEMORY.md"
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text("", encoding="utf-8")

    store = _StoreWithAssertion(memory_file=memory_file)
    tool = MemorySaveTool(store=store, workspace=tmp_path)

    raw = await tool.execute(
        content="Hybrid mode keeps files as source of truth.",
        memory_type="decision",
        context_tag="memory architecture",
        source_session="telegram:8554401569",
    )
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["memory_id"] == "mem-1"
    assert store.saved is True


class _RecallStore:
    def __init__(self):
        self.calls: list[dict] = []

    async def recall(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("peer_key") is not None:
            return []
        return [
            {
                "id": "global-1",
                "content": "global row",
                "memory_type": "fact",
                "source_message_ids": ["msg-7"],
                "source_turn_ids": ["turn-3"],
                "extraction_run_id": "run-xyz",
            }
        ]


@pytest.mark.asyncio
async def test_memory_recall_does_not_fallback_without_explicit_flag():
    store = _RecallStore()
    tool = MemoryRecallTool(store=store)

    raw = await tool.execute(query="hello", mode="hybrid", peer_key="peer:alpha")
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["count"] == 0
    assert payload["fallback_used"] is False
    assert len(store.calls) == 1


@pytest.mark.asyncio
async def test_memory_recall_can_fallback_when_explicitly_requested():
    store = _RecallStore()
    tool = MemoryRecallTool(store=store)

    raw = await tool.execute(
        query="hello",
        mode="hybrid",
        peer_key="peer:alpha",
        allow_global_fallback=True,
    )
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["fallback_used"] is True
    assert payload["peer_key"] is None
    assert payload["results"][0]["source_message_ids"] == ["msg-7"]
    assert payload["results"][0]["source_turn_ids"] == ["turn-3"]
    assert payload["results"][0]["extraction_run_id"] == "run-xyz"
    assert len(store.calls) == 2


class _Provider:
    def __init__(self, default_model: str):
        self._default_model = default_model

    def get_default_model(self) -> str:
        return self._default_model

    async def chat(self, **kwargs):
        return None


def test_module_model_resolution_prefers_llm_model(tmp_path):
    module = NanobotMemoryModule(
        provider=_Provider(default_model="provider-default"),
        workspace=tmp_path,
        config={
            "llm_model": "graph-llm-model",
            "background_model": "background-model",
        },
    )

    assert module.retriever is not None
    assert module.consolidator is not None
    assert module.ingestion is not None
    assert module.retriever.model == "graph-llm-model"
    assert module.consolidator.model == "graph-llm-model"
    assert module.ingestion.model == "graph-llm-model"


def test_module_model_resolution_uses_background_model_before_provider_default(tmp_path):
    module = NanobotMemoryModule(
        provider=_Provider(default_model="provider-default"),
        workspace=tmp_path,
        config={"background_model": "background-model"},
    )

    assert module.retriever is not None
    assert module.consolidator is not None
    assert module.ingestion is not None
    assert module.retriever.model == "background-model"
    assert module.consolidator.model == "background-model"
    assert module.ingestion.model == "background-model"


def test_module_model_resolution_falls_back_to_provider_default(tmp_path):
    module = NanobotMemoryModule(
        provider=_Provider(default_model="provider-default"),
        workspace=tmp_path,
        config={},
    )

    assert module.retriever is not None
    assert module.consolidator is not None
    assert module.ingestion is not None
    assert module.retriever.model == "provider-default"
    assert module.consolidator.model == "provider-default"
    assert module.ingestion.model == "provider-default"
