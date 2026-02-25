"""Nanobot adapter module for agent-memory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_memory_nanobot.hybrid import HybridMemoryManager
from agent_memory_nanobot.providers import NanobotLLMAdapter
from agent_memory_nanobot.tools import (
    MemoryForgetTool,
    MemoryGraphTool,
    MemoryIngestTool,
    MemoryRecallTool,
    MemorySaveTool,
    MemoryStatsTool,
)

if TYPE_CHECKING:
    from agent_memory import (
        LiteLLMEmbedding,
        MemoryConsolidator,
        MemoryGraphStore,
        MemoryIngestionAgent,
        PreTurnRetriever,
    )
    from nanobot.agent.tools.base import Tool
    from nanobot.providers.base import LLMProvider as NanobotProvider


class NanobotMemoryModule:
    """Facade that wires nanobot runtime providers into agent-memory components."""

    def __init__(
        self,
        provider: "NanobotProvider",
        workspace: Path,
        config: dict[str, Any] | None = None,
    ):
        self._provider = provider
        self._workspace = workspace
        self._config = config or {}
        self.initialized = False

        self.store: "MemoryGraphStore"
        self.retriever: "PreTurnRetriever" | None = None
        self.consolidator: "MemoryConsolidator" | None = None
        self.ingestion: "MemoryIngestionAgent" | None = None
        self.hybrid: HybridMemoryManager | None = None

        self._tools: list["Tool"] = []
        self._build_components()

    def _build_components(self) -> None:
        from agent_memory import (
            LiteLLMEmbedding,
            MemoryConsolidator,
            MemoryGraphStore,
            MemoryIngestionAgent,
            PreTurnRetriever,
        )

        db_path = str(self._config.get("db_path") or "~/.agent-memory/nanobot")
        embedding_cfg = self._config.get("embedding") or {}
        embedding_model = str(embedding_cfg.get("model") or "openai/text-embedding-3-small")
        if "/" not in embedding_model:
            # Force provider routing for common raw model names.
            embedding_model = f"openai/{embedding_model}"
        embedding_api_key = embedding_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        embedding_api_base = embedding_cfg.get("api_base") or embedding_cfg.get("base_url")
        embedding = LiteLLMEmbedding(
            model=embedding_model,
            dimensions=int(embedding_cfg.get("dimensions") or 512),
            api_key=str(embedding_api_key) if embedding_api_key else None,
            api_base=str(embedding_api_base) if embedding_api_base else None,
        )

        self.store = MemoryGraphStore(db_path=db_path, embedding=embedding)

        default_model = str(
            self._config.get("llm_model")
            or self._config.get("model")
            or self._provider.get_default_model()
        )
        llm = NanobotLLMAdapter(provider=self._provider, default_model=default_model)

        retrieval_cfg = self._config.get("retrieval") or {}
        if retrieval_cfg.get("enabled", True):
            self.retriever = PreTurnRetriever(
                store=self.store,
                llm=llm,
                model=str(retrieval_cfg.get("model") or default_model),
                max_context_words=int(retrieval_cfg.get("max_context_words") or 320),
                min_message_words=int(retrieval_cfg.get("min_message_words") or 3),
                max_results=int(retrieval_cfg.get("max_results") or 12),
                max_context_items=int(retrieval_cfg.get("max_context_items") or 12),
            )

        consolidation_cfg = self._config.get("consolidation") or {}
        if consolidation_cfg.get("enabled", True):
            self.consolidator = MemoryConsolidator(
                store=self.store,
                llm=llm,
                model=str(consolidation_cfg.get("model") or default_model),
                dedup_threshold=float(consolidation_cfg.get("dedup_threshold") or 0.3),
            )
            if str(consolidation_cfg.get("engine") or "legacy").lower() == "hybrid":
                self.hybrid = HybridMemoryManager(
                    workspace=self._workspace,
                    store=self.store,
                    consolidator=self.consolidator,
                    config=self._config,
                )

        ingestion_cfg = self._config.get("ingestion") or {}
        ingestion_chunk_chars = ingestion_cfg.get("chunk_size_chars")
        ingestion_chunk_words = ingestion_cfg.get("chunk_size_words")
        ingestion_overlap_chars = ingestion_cfg.get("chunk_overlap_chars")
        ingestion_dedup = ingestion_cfg.get("dedup_threshold")
        self.ingestion = MemoryIngestionAgent(
            store=self.store,
            llm=llm,
            model=str(ingestion_cfg.get("model") or default_model),
            chunk_size_chars=int(ingestion_chunk_chars) if ingestion_chunk_chars is not None else 3500,
            chunk_overlap_chars=(
                int(ingestion_overlap_chars) if ingestion_overlap_chars is not None else 300
            ),
            chunk_size_words=(
                int(ingestion_chunk_words) if ingestion_chunk_words is not None else None
            ),
            dedup_threshold=float(ingestion_dedup) if ingestion_dedup is not None else 0.15,
        )

        self._tools = [
            MemoryRecallTool(self.store),
            MemorySaveTool(self.store, workspace=self._workspace),
            MemoryForgetTool(self.store),
            MemoryGraphTool(self.store),
            MemoryIngestTool(self.ingestion),
            MemoryStatsTool(self.store),
        ]

    async def initialize(self) -> None:
        """Initialize persistent storage tables."""
        if self.initialized:
            return
        await self.store.initialize()
        self.initialized = True

    def get_tools(self) -> list["Tool"]:
        """Return memory tools ready to register in nanobot."""
        return list(self._tools)


__all__ = ["HybridMemoryManager", "NanobotMemoryModule", "NanobotLLMAdapter"]
