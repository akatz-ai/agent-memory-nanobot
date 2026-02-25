"""Nanobot tool adapters for agent-memory operations."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from nanobot.agent.tools.base import Tool
except Exception:
    from abc import ABC, abstractmethod

    class Tool(ABC):  # pragma: no cover - runtime fallback for import-time dependency gaps
        @property
        @abstractmethod
        def name(self) -> str:
            pass

        @property
        @abstractmethod
        def description(self) -> str:
            pass

        @property
        @abstractmethod
        def parameters(self) -> dict[str, Any]:
            pass

        @abstractmethod
        async def execute(self, **kwargs: Any) -> str:
            pass

if TYPE_CHECKING:
    from agent_memory.ingestion import MemoryIngestionAgent
    from agent_memory.store import MemoryGraphStore


def _json_ok(payload: dict[str, Any]) -> str:
    body = {"ok": True}
    body.update(payload)
    return json.dumps(body, ensure_ascii=False)


def _json_error(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


def _compact_memory_row(row: dict[str, Any], content_max: int = 280) -> dict[str, Any]:
    content = str(row.get("content", ""))
    if len(content) > content_max:
        content = content[: content_max - 1] + "…"
    out = {
        "id": row.get("id"),
        "memory_type": row.get("memory_type"),
        "importance": row.get("importance"),
        "content": content,
        "entities": row.get("entities", []),
        "source": row.get("source"),
        "created_at_ms": row.get("created_at_ms"),
    }
    if row.get("source_session"):
        out["source_session"] = row.get("source_session")
    if row.get("context_tag"):
        out["context_tag"] = row.get("context_tag")
    if "_distance" in row:
        out["distance"] = row.get("_distance")
    if "_relevance_score" in row:
        out["relevance_score"] = row.get("_relevance_score")
    if "similarity_score" in row:
        out["similarity_score"] = row.get("similarity_score")
    return out


class MemoryRecallTool(Tool):
    """Search memory using hybrid/vector/keyword/recent/important modes."""

    def __init__(self, store: "MemoryGraphStore"):
        self._store = store

    @property
    def name(self) -> str:
        return "memory_recall"

    @property
    def description(self) -> str:
        return "Recall relevant memories by semantic, keyword, recency, importance, and graph expansion."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "vector", "keyword", "recent", "important"],
                    "description": "Recall mode",
                },
                "memory_type": {"type": "string", "description": "Optional memory type filter"},
                "peer_key": {"type": "string", "description": "Optional peer/session scope"},
                "allow_global_fallback": {
                    "type": "boolean",
                    "description": "If true, retry without peer_key when peer-scoped recall is empty.",
                },
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max rows"},
                "graph_depth": {"type": "integer", "minimum": 0, "maximum": 3, "description": "Graph expansion depth"},
            },
        }

    async def execute(
        self,
        query: str | None = None,
        mode: str = "hybrid",
        memory_type: str | None = None,
        peer_key: str | None = None,
        allow_global_fallback: bool = False,
        max_results: int = 10,
        graph_depth: int = 1,
        **kwargs: Any,
    ) -> str:
        if mode in {"hybrid", "vector", "keyword"} and not (query or "").strip():
            return _json_error("query is required for hybrid/vector/keyword modes")

        try:
            results = await self._store.recall(
                query=query,
                mode=mode,
                memory_type=memory_type,
                peer_key=peer_key,
                max_results=max_results,
                graph_depth=graph_depth,
            )
            used_peer_key = peer_key
            fallback_used = False
            if not results and peer_key and allow_global_fallback:
                results = await self._store.recall(
                    query=query,
                    mode=mode,
                    memory_type=memory_type,
                    peer_key=None,
                    max_results=max_results,
                    graph_depth=graph_depth,
                )
                if results:
                    fallback_used = True
                    used_peer_key = None

            compact = [_compact_memory_row(row) for row in results[:max_results]]
            return _json_ok(
                {
                    "mode": mode,
                    "count": len(compact),
                    "peer_key": used_peer_key,
                    "fallback_used": fallback_used,
                    "allow_global_fallback": bool(allow_global_fallback),
                    "results": compact,
                }
            )
        except Exception as exc:
            return _json_error(str(exc))


class MemorySaveTool(Tool):
    """Save a memory node to the graph store."""

    def __init__(self, store: "MemoryGraphStore", workspace: Path | None = None):
        self._store = store
        self._memory_file = workspace / "memory" / "MEMORY.md" if workspace else None

    @property
    def name(self) -> str:
        return "memory_save"

    @property
    def description(self) -> str:
        return "Save a memory item with metadata such as type, importance, entities, and associations."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory text content"},
                "memory_type": {"type": "string", "description": "Memory type, e.g. fact/event/decision"},
                "importance": {"type": "number", "minimum": 0, "maximum": 1, "description": "Importance score"},
                "source": {"type": "string", "description": "Source label"},
                "source_session": {"type": "string", "description": "Source session key"},
                "context_tag": {"type": "string", "description": "Optional context label (2-5 words)"},
                "peer_key": {"type": "string", "description": "Peer/session scope"},
                "entities": {"type": "array", "items": {"type": "string"}, "description": "Named entities"},
                "associations": {
                    "type": "array",
                    "description": "Optional association edges to existing memory IDs",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_id": {"type": "string"},
                            "relation_type": {"type": "string"},
                            "weight": {"type": "number"},
                            "provenance": {"type": "string"},
                        },
                        "required": ["target_id"],
                    },
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        source: str = "manual",
        source_session: str | None = None,
        context_tag: str | None = None,
        peer_key: str | None = None,
        entities: list[str] | None = None,
        associations: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            self._write_through_memory_file(
                content=content,
                memory_type=memory_type,
                context_tag=context_tag,
                source_session=source_session,
            )
            memory_id = await self._store.save(
                content=content,
                memory_type=memory_type,
                importance=importance,
                source=source,
                source_session=source_session,
                context_tag=context_tag,
                peer_key=peer_key,
                entities=entities,
                associations=associations,
            )
            return _json_ok({"memory_id": memory_id})
        except Exception as exc:
            return _json_error(str(exc))

    def _write_through_memory_file(
        self,
        content: str,
        memory_type: str,
        context_tag: str | None,
        source_session: str | None,
    ) -> None:
        if self._memory_file is None:
            return

        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._memory_file.exists():
            self._memory_file.write_text("", encoding="utf-8")

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        line = f"- [{stamp}] [{memory_type}] {content.strip()}"
        if context_tag:
            line += f" (context: {context_tag.strip()})"
        if source_session:
            line += f" (session: {source_session.strip()})"

        existing = self._memory_file.read_text(encoding="utf-8")
        if line in existing:
            return

        prefix = "" if (not existing or existing.endswith("\n")) else "\n"
        self._memory_file.write_text(f"{existing}{prefix}{line}\n", encoding="utf-8")


class MemoryForgetTool(Tool):
    """Soft-delete a memory node."""

    def __init__(self, store: "MemoryGraphStore"):
        self._store = store

    @property
    def name(self) -> str:
        return "memory_forget"

    @property
    def description(self) -> str:
        return "Soft-delete (forget) a memory by memory ID."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to forget"},
                "reason": {"type": "string", "description": "Optional reason for forgetting"},
            },
            "required": ["memory_id"],
        }

    async def execute(self, memory_id: str, reason: str | None = None, **kwargs: Any) -> str:
        try:
            forgotten = await self._store.forget(memory_id=memory_id, reason=reason)
            return _json_ok({"memory_id": memory_id, "forgotten": forgotten})
        except Exception as exc:
            return _json_error(str(exc))


class MemoryGraphTool(Tool):
    """Traverse graph relationships from seed memory IDs."""

    def __init__(self, store: "MemoryGraphStore"):
        self._store = store

    @property
    def name(self) -> str:
        return "memory_graph"

    @property
    def description(self) -> str:
        return "Traverse associated memories starting from one or more seed IDs."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "seed_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Seed memory IDs to traverse from",
                },
                "depth": {"type": "integer", "minimum": 0, "maximum": 3, "description": "Traversal depth"},
                "relation_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional relation type filter",
                },
                "max_nodes": {"type": "integer", "minimum": 1, "maximum": 500, "description": "Node limit"},
            },
            "required": ["seed_ids"],
        }

    async def execute(
        self,
        seed_ids: list[str],
        depth: int = 1,
        relation_types: list[str] | None = None,
        max_nodes: int = 50,
        **kwargs: Any,
    ) -> str:
        if not seed_ids:
            return _json_error("seed_ids must not be empty")

        try:
            nodes: dict[str, dict[str, Any]] = {}
            edges: dict[str, dict[str, Any]] = {}

            for seed_id in seed_ids:
                graph = await self._store.get_neighbors(
                    memory_id=seed_id,
                    depth=depth,
                    relation_types=relation_types,
                    max_nodes=max_nodes,
                )
                for node in graph.get("nodes", []):
                    node_id = node.get("id")
                    if node_id and node_id not in nodes:
                        nodes[node_id] = node
                for edge in graph.get("edges", []):
                    edge_id = edge.get("id")
                    if edge_id and edge_id not in edges:
                        edges[edge_id] = edge

            return _json_ok(
                {
                    "seed_ids": seed_ids,
                    "depth": depth,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "nodes": list(nodes.values()),
                    "edges": list(edges.values()),
                }
            )
        except Exception as exc:
            return _json_error(str(exc))


class MemoryIngestTool(Tool):
    """Ingest raw markdown/text into structured graph memories."""

    def __init__(self, ingestion: "MemoryIngestionAgent" | None):
        self._ingestion = ingestion

    @property
    def name(self) -> str:
        return "memory_ingest"

    @property
    def description(self) -> str:
        return "Ingest raw markdown or text and extract memory records into the graph."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Raw markdown or plain text to ingest"},
                "source": {"type": "string", "description": "Source label"},
                "peer_key": {"type": "string", "description": "Optional peer/session scope"},
                "file_name": {"type": "string", "description": "Virtual filename hint for ingestion"},
            },
            "required": ["text"],
        }

    async def execute(
        self,
        text: str,
        source: str = "tool_ingestion",
        peer_key: str | None = None,
        file_name: str = "ingested.md",
        **kwargs: Any,
    ) -> str:
        if self._ingestion is None:
            return _json_error("ingestion agent is not configured")

        temp_path: Path | None = None
        try:
            suffix = Path(file_name).suffix or ".md"
            with tempfile.NamedTemporaryFile("w", suffix=suffix, encoding="utf-8", delete=False) as tmp:
                tmp.write(text)
                temp_path = Path(tmp.name)

            result = await self._ingestion.ingest_file(
                file_path=temp_path,
                source=source,
                peer_key=peer_key,
            )
            return _json_ok(result)
        except Exception as exc:
            return _json_error(str(exc))
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)


class MemoryStatsTool(Tool):
    """Return store-level statistics."""

    def __init__(self, store: "MemoryGraphStore"):
        self._store = store

    @property
    def name(self) -> str:
        return "memory_stats"

    @property
    def description(self) -> str:
        return "Return memory graph statistics such as total nodes, edges, and type distribution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            stats = await self._store.stats()
            return _json_ok(stats)
        except Exception as exc:
            return _json_error(str(exc))
