from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_memory.consolidator import MemoryConsolidator
    from agent_memory.store import MemoryGraphStore

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    entry_type: str
    content: str
    context: str
    entities: list[str]
    importance: float = 0.5
    source_message_ids: list[str] | None = None
    source_turn_ids: list[str] | None = None
    extraction_run_id: str | None = None


@dataclass
class CompactionResult:
    entries: list[HistoryEntry]
    history_file: Path
    messages_processed: int
    memories_indexed: int
    edges_created: int


class HybridMemoryManager:
    """Unified file-first memory with graph indexing."""
    _CANONICAL_SECTIONS = (
        "Identity & Preferences",
        "Active Projects",
        "Decisions",
        "Reference Facts",
        "Recent Context",
    )
    _DEFAULT_MEMORY_MD_MAX_TOKENS = 4000
    _DEFAULT_MEMORY_MD_MAX_CHARS = 16000

    def __init__(
        self,
        workspace: Path,
        store: "MemoryGraphStore",
        consolidator: "MemoryConsolidator",
        config: dict[str, Any] | None = None,
    ):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.history_dir = self.memory_dir / "history"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.store = store
        self.consolidator = consolidator
        self.config = config or {}
        consolidation_cfg = self.config.get("consolidation") or {}
        self._memory_md_max_tokens = max(
            100,
            int(consolidation_cfg.get("memory_md_max_tokens") or self._DEFAULT_MEMORY_MD_MAX_TOKENS),
        )
        self._memory_md_max_chars = max(
            1000,
            int(consolidation_cfg.get("memory_md_max_chars") or self._DEFAULT_MEMORY_MD_MAX_CHARS),
        )

    async def compact(
        self,
        session_key: str,
        messages: list[dict[str, Any]],
        start_index: int,
        end_index: int,
    ) -> CompactionResult:
        """Single-pass extraction + file write + graph index."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        start = max(0, start_index)
        end = max(start, min(end_index, len(messages)))
        message_window = messages[start:end]
        compaction_time = self._resolve_compaction_time(message_window)

        extracted = await self.consolidator.extract_from_messages(
            messages=message_window,
            peer_key=session_key,
        )
        entries = [self._to_history_entry(item) for item in extracted if item.get("content")]

        history_file = self._write_history_entries(entries, session_key=session_key, timestamp=compaction_time)
        memories_indexed, edges_created = await self._index_entries(entries, session_key=session_key)
        await self._rewrite_memory_md(entries, session_key=session_key, timestamp=compaction_time)

        return CompactionResult(
            entries=entries,
            history_file=history_file,
            messages_processed=len(message_window),
            memories_indexed=memories_indexed,
            edges_created=edges_created,
        )

    def _write_history_entries(
        self,
        entries: list[HistoryEntry],
        session_key: str,
        timestamp: datetime,
    ) -> Path:
        """Append entries to today's history file. Returns file path."""
        history_file = self.history_dir / f"{timestamp.date().isoformat()}.md"
        heading = f"# {timestamp.date().isoformat()}\n\n"
        if not history_file.exists():
            history_file.write_text(heading, encoding="utf-8")

        if not entries:
            return history_file

        section = self._format_history_section(entries, session_key=session_key, timestamp=timestamp)
        marker = section.splitlines()[0]
        existing = history_file.read_text(encoding="utf-8")
        if marker in existing:
            return history_file

        separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        history_file.write_text(f"{existing}{separator}{section}\n", encoding="utf-8")
        return history_file

    def _format_history_section(
        self,
        entries: list[HistoryEntry],
        session_key: str,
        timestamp: datetime,
    ) -> str:
        """Format entries as markdown section for daily file."""
        section_id = self._section_id(entries, session_key=session_key, timestamp=timestamp)
        lines = [
            f"<!-- compact_id: {section_id} -->",
            f"## {timestamp.strftime('%H:%M')} — Compaction ({session_key})",
            "",
        ]
        for entry in entries:
            context = entry.context.strip() or "general"
            content = entry.content.strip().replace("\n", " ")
            lines.append(f"- [{entry.entry_type}] {content} (context: {context})")
        return "\n".join(lines).rstrip()

    async def _index_entries(
        self,
        entries: list[HistoryEntry],
        session_key: str,
    ) -> tuple[int, int]:
        """Index entries into graph. Returns (memories_indexed, edges_created)."""
        if not entries:
            return 0, 0

        extracted_payload = [
            {
                "content": entry.content,
                "entry_type": entry.entry_type,
                "memory_type": entry.entry_type,
                "context_tag": entry.context,
                "importance": entry.importance,
                "entities": entry.entities,
                "source_message_ids": list(entry.source_message_ids or []),
                "source_turn_ids": list(entry.source_turn_ids or []),
                "extraction_run_id": entry.extraction_run_id,
            }
            for entry in entries
        ]
        stats = await self.consolidator.index_extracted(
            extracted=extracted_payload,
            peer_key=session_key,
            source_session=session_key,
            source="hybrid_history",
        )
        return int(stats.get("added", 0)) + int(stats.get("updated", 0)), int(
            stats.get("edges_created", 0)
        )

    def read_memory_md(self) -> str:
        """Read MEMORY.md contents."""
        if not self.memory_file.exists():
            return ""
        return self.memory_file.read_text(encoding="utf-8")

    def read_daily_history(self, date: str | None = None) -> str:
        """Read a daily history file. Defaults to today."""
        day = date or datetime.now().date().isoformat()
        file_path = self.history_dir / f"{day}.md"
        if not file_path.exists():
            return ""
        return file_path.read_text(encoding="utf-8")

    def list_history_files(self) -> list[Path]:
        """List all daily history files, sorted by date."""
        if not self.history_dir.exists():
            return []
        return sorted(self.history_dir.glob("*.md"))

    def _to_history_entry(self, extracted: dict[str, Any]) -> HistoryEntry:
        entry_type = self._coerce_entry_type(extracted.get("entry_type", extracted.get("memory_type")))
        content = str(extracted.get("content", "")).strip()
        context = str(extracted.get("context_tag") or "general").strip()
        entities = [str(e).strip() for e in extracted.get("entities", []) if str(e).strip()]
        importance = self._coerce_importance(extracted.get("importance", 0.5))
        return HistoryEntry(
            entry_type=entry_type,
            content=content,
            context=context or "general",
            entities=entities,
            importance=importance,
            source_message_ids=self._normalize_strings(extracted.get("source_message_ids")),
            source_turn_ids=self._normalize_strings(extracted.get("source_turn_ids")),
            extraction_run_id=self._coerce_optional_str(extracted.get("extraction_run_id")),
        )

    def _section_id(self, entries: list[HistoryEntry], session_key: str, timestamp: datetime) -> str:
        payload = [
            {
                "entry_type": entry.entry_type,
                "content": entry.content,
                "context": entry.context,
                "entities": entry.entities,
            }
            for entry in entries
        ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest_src = f"{session_key}|{timestamp.isoformat()}|{raw}".encode("utf-8")
        return hashlib.sha256(digest_src).hexdigest()[:16]

    def _resolve_compaction_time(self, messages: list[dict[str, Any]]) -> datetime:
        for msg in messages:
            raw = msg.get("timestamp")
            if not isinstance(raw, str):
                continue
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
        return datetime.now()

    async def _rewrite_memory_md(
        self,
        entries: list[HistoryEntry],
        session_key: str,
        timestamp: datetime,
    ) -> None:
        if not entries:
            return

        existing = self._coerce_to_canonical(self.read_memory_md())
        section = self._format_history_section(entries, session_key=session_key, timestamp=timestamp)
        model = str(
            (self.config.get("consolidation") or {}).get("model")
            or getattr(self.consolidator, "model", "")
        )

        prompt = (
            "Update MEMORY.md as a curated long-term memory document.\n"
            f"Use this exact canonical structure with markdown headings: {', '.join(self._CANONICAL_SECTIONS)}.\n"
            "Rules:\n"
            "- Keep stable facts, preferences, decisions, goals, and active projects.\n"
            "- Remove chatter, transient details, and instructions.\n"
            f"- Hard budget: <= {self._memory_md_max_tokens} tokens and <= {self._memory_md_max_chars} chars.\n"
            "- Return markdown only."
        )
        user_message = (
            "## Existing MEMORY.md\n"
            f"{existing or '(empty)'}\n\n"
            "## New Daily History Entries\n"
            f"{section}\n\n"
            "Return the full updated MEMORY.md using the canonical headings."
        )

        try:
            updated = await self.consolidator.llm.complete(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message},
                ],
                model=model or None,
                temperature=0.0,
                max_tokens=min(self._memory_md_max_tokens, 3500),
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Hybrid MEMORY.md rewrite failed: %s", exc)
            return

        normalized = self._strip_code_fence(updated)
        if not normalized.strip():
            logger.warning("Hybrid MEMORY.md rewrite rejected: empty output")
            return
        if not self._is_valid_canonical_structure(normalized):
            logger.warning("Hybrid MEMORY.md rewrite rejected: malformed canonical structure")
            return

        bounded, overflow_sections = self._enforce_memory_budget(
            self._coerce_to_canonical(normalized)
        )
        if not bounded.strip():
            logger.warning("Hybrid MEMORY.md rewrite rejected: empty after budget enforcement")
            return
        if not self._fits_memory_budget(bounded):
            logger.warning("Hybrid MEMORY.md rewrite rejected: exceeded memory budget")
            return

        self._archive_memory_overflow(
            overflow_sections,
            session_key=session_key,
            timestamp=timestamp,
        )
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(bounded.strip() + "\n", encoding="utf-8")

    def _coerce_to_canonical(self, content: str) -> str:
        text = self._strip_code_fence(content or "").strip()
        if not text:
            return self._empty_canonical_memory()

        if not self._is_valid_canonical_structure(text):
            # Preserve prior memory by moving legacy/unstructured content into facts.
            facts = [line.strip() for line in text.splitlines() if line.strip()]
            sections = {name: [] for name in self._CANONICAL_SECTIONS}
            sections["Reference Facts"] = [f"- {line}" for line in facts]
            return self._render_canonical(sections)

        parsed = self._parse_canonical_sections(text)
        return self._render_canonical(parsed)

    def _empty_canonical_memory(self) -> str:
        sections = {name: [] for name in self._CANONICAL_SECTIONS}
        return self._render_canonical(sections)

    def _parse_canonical_sections(self, content: str) -> dict[str, list[str]]:
        sections = {name: [] for name in self._CANONICAL_SECTIONS}
        current: str | None = None
        for raw_line in content.strip().splitlines():
            line = raw_line.rstrip()
            if line.startswith("# "):
                continue
            if line.startswith("## "):
                header = line[3:].strip()
                current = header if header in sections else None
                continue
            if current is None:
                continue
            sections[current].append(line)
        return sections

    def _render_canonical(self, sections: dict[str, list[str]]) -> str:
        lines = ["# MEMORY", ""]
        for header in self._CANONICAL_SECTIONS:
            body = sections.get(header, [])
            if not body or not any(line.strip() for line in body):
                body = ["- (none)"]
            lines.append(f"## {header}")
            lines.extend(body)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _is_valid_canonical_structure(self, content: str) -> bool:
        stripped = content.strip()
        if not stripped:
            return False
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines or lines[0] != "# MEMORY":
            return False
        headers = [line[3:].strip() for line in lines if line.startswith("## ")]
        if len(headers) != len(set(headers)):
            return False
        return all(section in headers for section in self._CANONICAL_SECTIONS)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def _fits_memory_budget(self, content: str) -> bool:
        return (
            len(content) <= self._memory_md_max_chars
            and self._estimate_tokens(content) <= self._memory_md_max_tokens
        )

    def _enforce_memory_budget(self, content: str) -> tuple[str, list[tuple[str, list[str]]]]:
        if self._fits_memory_budget(content):
            return content, []

        sections = self._parse_canonical_sections(content)
        overflow: list[tuple[str, list[str]]] = []
        section_to_overflow: dict[str, list[str]] = {}

        def _move_whole_section(header: str) -> None:
            body = [line for line in sections.get(header, []) if line.strip()]
            if not body:
                return
            overflow.append((header, body))
            section_to_overflow.setdefault(header, []).extend(body)
            sections[header] = ["- (archived due to size budget)"]

        # Demote low-priority sections first.
        for header in reversed(self._CANONICAL_SECTIONS):
            candidate = self._render_canonical(sections)
            if self._fits_memory_budget(candidate):
                return candidate, overflow
            _move_whole_section(header)

        # If still too large, trim lines from each section tail.
        for header in self._CANONICAL_SECTIONS:
            body = sections.get(header, [])
            while len(body) > 1 and not self._fits_memory_budget(self._render_canonical(sections)):
                removed = body.pop()
                if removed.strip():
                    section_to_overflow.setdefault(header, []).insert(0, removed)
                if not any(line.strip() for line in body):
                    body[:] = ["- (archived due to size budget)"]
            sections[header] = body

        bounded = self._render_canonical(sections)
        if section_to_overflow:
            overflow = [(header, lines) for header, lines in section_to_overflow.items() if lines]
        return bounded, overflow

    def _archive_memory_overflow(
        self,
        overflow_sections: list[tuple[str, list[str]]],
        *,
        session_key: str,
        timestamp: datetime,
    ) -> None:
        if not overflow_sections:
            return

        self.history_dir.mkdir(parents=True, exist_ok=True)
        history_file = self.history_dir / f"{timestamp.date().isoformat()}.md"
        heading = f"# {timestamp.date().isoformat()}\n\n"
        if not history_file.exists():
            history_file.write_text(heading, encoding="utf-8")

        payload = json.dumps(overflow_sections, ensure_ascii=False, sort_keys=True)
        marker_id = hashlib.sha256(
            f"{session_key}|{timestamp.isoformat()}|{payload}".encode("utf-8")
        ).hexdigest()[:16]
        marker = f"<!-- memory_overflow_id: {marker_id} -->"

        existing = history_file.read_text(encoding="utf-8")
        if marker in existing:
            return

        lines = [
            marker,
            f"## {timestamp.strftime('%H:%M')} — MEMORY.md overflow archive ({session_key})",
            "",
            "Moved sections from MEMORY.md to enforce compaction budget:",
            "",
        ]
        for header, body in overflow_sections:
            if not body:
                continue
            lines.append(f"### {header}")
            lines.extend(body)
            lines.append("")

        section = "\n".join(lines).rstrip()
        separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        history_file.write_text(f"{existing}{separator}{section}\n", encoding="utf-8")

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        candidate = (text or "").strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
        return candidate

    @staticmethod
    def _coerce_entry_type(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"fact", "decision", "event", "goal"}:
            return text
        mapped = {
            "preference": "fact",
            "identity": "fact",
            "observation": "event",
            "todo": "goal",
        }
        return mapped.get(text, "fact")

    @staticmethod
    def _coerce_importance(value: Any) -> float:
        try:
            importance = float(value)
        except (TypeError, ValueError):
            importance = 0.5
        return max(0.0, min(1.0, importance))

    @staticmethod
    def _normalize_strings(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        out: list[str] = []
        for value in values:
            text = str(value).strip()
            if text:
                out.append(text)
        return out

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
