"""
memory_engine.py — Structured memory retrieval for the Agent OS.

Four memory banks:
  TASK    → facts learned during this specific task (volatile, not persisted)
  PROJECT → repo/project knowledge (persisted, updated across runs)
  USER    → user preferences, patterns, instructions (persisted)
  RUN     → outcomes of prior runs (persisted, used for learning)

Retrieval contract:
  - NEVER dump full history into context
  - Retrieve only chunks relevant to the current task query
  - Return compact, structured summaries
  - Respect model context budget
  - Keep memory entries verifiable (tagged with source)
"""

from __future__ import annotations

import json
import re
import time
import math
import hashlib
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# Memory types
# ─────────────────────────────────────────────────────────────────────────────

class MemoryBank(str, Enum):
    TASK    = "task"       # current task scope
    PROJECT = "project"    # repo / codebase facts
    USER    = "user"       # user prefs and patterns
    RUN     = "run"        # prior run outcomes


class MemoryPriority(str, Enum):
    HIGH   = "high"
    NORMAL = "normal"
    LOW    = "low"


# ─────────────────────────────────────────────────────────────────────────────
# Memory entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemEntry:
    entry_id:    str  = field(default_factory=lambda: uuid.uuid4().hex[:16])
    bank:        MemoryBank    = MemoryBank.TASK
    priority:    MemoryPriority = MemoryPriority.NORMAL
    content:     str  = ""
    tags:        list[str] = field(default_factory=list)
    source:      str  = "agent"     # "tool", "user", "system", "verified"
    task_id:     str  = ""
    created_at:  float = field(default_factory=time.time)
    updated_at:  float = field(default_factory=time.time)
    expires_at:  Optional[float] = None  # None = never
    access_count: int = 0

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    def relevance(self, query: str) -> float:
        """TF-IDF-like relevance score."""
        qt = _tokenize(query)
        ct = _tokenize(self.content)
        if not qt or not ct:
            return 0.0
        content_freq = Counter(ct)
        total = len(ct)
        score = 0.0
        for term in qt:
            if term in content_freq:
                tf  = content_freq[term] / total
                idf = math.log(10 / 1) + 1  # simplified IDF
                score += tf * idf
        # Tag bonus
        tag_hit = len(set(qt) & set(self.tags))
        score += tag_hit * 0.2
        # Priority multiplier
        if self.priority == MemoryPriority.HIGH:
            score *= 1.5
        elif self.priority == MemoryPriority.LOW:
            score *= 0.7
        return score

    def to_context_line(self, max_chars: int = 200) -> str:
        """Compact one-liner for context injection."""
        content = self.content[:max_chars]
        tag_str = f" [{', '.join(self.tags[:3])}]" if self.tags else ""
        src = f" ({self.source})" if self.source != "agent" else ""
        return f"• {content}{tag_str}{src}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bank"] = self.bank.value
        d["priority"] = self.priority.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemEntry":
        d = dict(d)
        d["bank"] = MemoryBank(d.get("bank", "task"))
        d["priority"] = MemoryPriority(d.get("priority", "normal"))
        return cls(**d)


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryRetrieval:
    entries:    list[MemEntry]
    banks_hit:  list[str]
    query:      str
    tokens_est: int

    def to_context_chunks(self, max_per_bank: int = 3,
                           max_chars_each: int = 180) -> list[str]:
        """Return compact strings ready for injection into system prompt."""
        chunks = []
        by_bank: dict[str, list[MemEntry]] = {}
        for e in self.entries:
            by_bank.setdefault(e.bank.value, []).append(e)

        for bank in [MemoryBank.PROJECT, MemoryBank.USER,
                     MemoryBank.RUN, MemoryBank.TASK]:
            bank_entries = by_bank.get(bank.value, [])[:max_per_bank]
            if bank_entries:
                header = f"[{bank.value.upper()} MEMORY]"
                chunks.append(header)
                for e in bank_entries:
                    chunks.append(e.to_context_line(max_chars_each))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Per-bank store
# ─────────────────────────────────────────────────────────────────────────────

class BankStore:
    """One JSON-backed store for one memory bank."""

    MAX_ENTRIES = 200

    def __init__(self, bank: MemoryBank, persist_path: Optional[Path] = None):
        self.bank = bank
        self._path = persist_path
        self._entries: list[MemEntry] = []
        if persist_path and persist_path.exists():
            self._load()

    def add(self, entry: MemEntry):
        entry.bank = self.bank
        self._entries.append(entry)
        # Evict low-priority expired entries if over limit
        if len(self._entries) > self.MAX_ENTRIES:
            self._evict()
        if self._path:
            self._save()

    def retrieve(self, query: str, top_k: int = 5,
                  min_score: float = 0.05) -> list[MemEntry]:
        """Return top_k entries most relevant to query."""
        active = [e for e in self._entries if not e.is_expired()]
        if not active:
            return []
        scored = [(e, e.relevance(query)) for e in active]
        scored.sort(key=lambda x: x[1], reverse=True)
        result = [e for e, s in scored if s >= min_score][:top_k]
        for e in result:
            e.access_count += 1
            e.updated_at = time.time()
        return result

    def get_all_recent(self, n: int = 10) -> list[MemEntry]:
        return sorted(self._entries, key=lambda e: e.created_at, reverse=True)[:n]

    def upsert(self, entry_id: str, content: str,
               bank: MemoryBank = None, **kwargs):
        existing = next((e for e in self._entries if e.entry_id == entry_id), None)
        if existing:
            existing.content = content
            existing.updated_at = time.time()
            for k, v in kwargs.items():
                if hasattr(existing, k):
                    setattr(existing, k, v)
        else:
            e = MemEntry(entry_id=entry_id, bank=bank or self.bank,
                          content=content, **kwargs)
            self.add(e)
        if self._path:
            self._save()

    def delete(self, entry_id: str):
        self._entries = [e for e in self._entries if e.entry_id != entry_id]
        if self._path:
            self._save()

    def clear(self):
        self._entries.clear()
        if self._path:
            self._save()

    def _evict(self):
        """Remove expired and lowest-priority entries."""
        self._entries = [e for e in self._entries if not e.is_expired()]
        if len(self._entries) > self.MAX_ENTRIES:
            low = [e for e in self._entries
                   if e.priority == MemoryPriority.LOW]
            low.sort(key=lambda e: e.access_count)
            to_remove = len(self._entries) - self.MAX_ENTRIES
            for e in low[:to_remove]:
                self._entries.remove(e)

    def _load(self):
        try:
            with open(self._path) as f:
                data = json.load(f)
            if isinstance(data, list):
                self._entries = [MemEntry.from_dict(d) for d in data]
        except Exception:
            pass

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump([e.to_dict() for e in self._entries], f, indent=2)
        except Exception:
            pass

    @property
    def count(self) -> int:
        return len(self._entries)


# ─────────────────────────────────────────────────────────────────────────────
# Memory Engine
# ─────────────────────────────────────────────────────────────────────────────

class MemoryEngine:
    """
    Unified interface to all four memory banks.

    Usage:
        mem = MemoryEngine(agent_id="agent_abc123")
        mem.save_task_fact("main.py has 342 lines", tags=["main.py"])
        mem.save_run_outcome(task_id, success=True, summary="Fixed auth bug in login.py")

        retrieval = mem.retrieve_for_context("fix the login bug", max_chunks=5)
        chunks = retrieval.to_context_chunks()
    """

    def __init__(
        self,
        agent_id: str = "default",
        memory_dir: str = ".agent_os/memory",
        persist_project: bool = True,
        persist_user:    bool = True,
        persist_runs:    bool = True,
    ):
        self.agent_id = agent_id
        base = Path(memory_dir) / agent_id
        base.mkdir(parents=True, exist_ok=True)

        self._stores: dict[MemoryBank, BankStore] = {
            MemoryBank.TASK:    BankStore(MemoryBank.TASK),  # volatile
            MemoryBank.PROJECT: BankStore(
                MemoryBank.PROJECT,
                base / "project.json" if persist_project else None,
            ),
            MemoryBank.USER:    BankStore(
                MemoryBank.USER,
                base / "user.json" if persist_user else None,
            ),
            MemoryBank.RUN:     BankStore(
                MemoryBank.RUN,
                base / "runs.json" if persist_runs else None,
            ),
        }

    # ── Write API ─────────────────────────────────────────────────────────────

    def save_task_fact(self, content: str, tags: list[str] = None,
                        priority: MemoryPriority = MemoryPriority.NORMAL,
                        ttl_seconds: float = None, source: str = "agent"):
        """Short-lived facts about the current task."""
        e = MemEntry(
            bank=MemoryBank.TASK,
            content=content,
            tags=tags or [],
            priority=priority,
            source=source,
            expires_at=time.time() + ttl_seconds if ttl_seconds else None,
        )
        self._stores[MemoryBank.TASK].add(e)

    def save_verified_fact(self, content: str, tags: list[str] = None):
        """A fact verified by a tool call — highest trust."""
        e = MemEntry(
            bank=MemoryBank.TASK,
            content=content,
            tags=tags or [],
            priority=MemoryPriority.HIGH,
            source="verified",
        )
        self._stores[MemoryBank.TASK].add(e)

    def save_project_fact(self, content: str, tags: list[str] = None,
                           priority: MemoryPriority = MemoryPriority.NORMAL):
        """Persisted knowledge about the project/repo."""
        e = MemEntry(
            bank=MemoryBank.PROJECT,
            content=content,
            tags=tags or [],
            priority=priority,
            source="agent",
        )
        self._stores[MemoryBank.PROJECT].add(e)

    def save_user_pref(self, content: str, tags: list[str] = None):
        """Persisted user preference or instruction."""
        e = MemEntry(
            bank=MemoryBank.USER,
            content=content,
            tags=tags or ["user_pref"],
            priority=MemoryPriority.HIGH,
            source="user",
        )
        self._stores[MemoryBank.USER].add(e)

    def save_run_outcome(
        self,
        task_id:  str,
        task_text: str,
        success:  bool,
        summary:  str,
        duration_ms: float = 0.0,
        tokens_used: int = 0,
        tags:     list[str] = None,
    ):
        """Record the outcome of a completed task run."""
        status = "SUCCESS" if success else "FAILED"
        content = (
            f"[{status}] {task_text[:100]} | "
            f"duration={duration_ms:.0f}ms tokens={tokens_used} | "
            f"{summary[:200]}"
        )
        e = MemEntry(
            bank=MemoryBank.RUN,
            content=content,
            tags=(tags or []) + [task_id[:8], status.lower()],
            priority=MemoryPriority.HIGH if not success else MemoryPriority.NORMAL,
            source="system",
            task_id=task_id,
        )
        self._stores[MemoryBank.RUN].add(e)

    def save_error(self, error_msg: str, tool: str = "",
                    ttl_seconds: float = 3600):
        """Record an error to avoid repeating it."""
        content = f"ERROR in {tool}: {error_msg[:200]}" if tool else f"ERROR: {error_msg[:200]}"
        e = MemEntry(
            bank=MemoryBank.TASK,
            content=content,
            tags=["error", tool] if tool else ["error"],
            priority=MemoryPriority.HIGH,
            source="system",
            expires_at=time.time() + ttl_seconds,
        )
        self._stores[MemoryBank.TASK].add(e)

    # ── Read API ──────────────────────────────────────────────────────────────

    def retrieve_for_context(
        self,
        query: str,
        max_chunks: int = 8,        # total entries across all banks
        max_per_bank: int = 3,
        min_score: float = 0.05,
    ) -> MemoryRetrieval:
        """
        Pull the most relevant memory entries for injection into context.

        Banks queried in order: PROJECT > USER > RUN > TASK
        (TASK is most volatile and recent — prioritize at end)
        """
        all_entries: list[MemEntry] = []
        banks_hit: list[str] = []

        bank_order = [MemoryBank.PROJECT, MemoryBank.USER,
                       MemoryBank.RUN, MemoryBank.TASK]

        per_bank = max(1, max_chunks // len(bank_order))
        per_bank = min(per_bank, max_per_bank)

        for bank in bank_order:
            hits = self._stores[bank].retrieve(query, top_k=per_bank,
                                               min_score=min_score)
            if hits:
                all_entries.extend(hits)
                banks_hit.append(bank.value)

        # Sort by relevance (already scored in store.retrieve, so keep order)
        # but re-score globally so highest-priority entries float up
        all_entries_scored = [
            (e, e.relevance(query)) for e in all_entries
        ]
        all_entries_scored.sort(key=lambda x: x[1], reverse=True)
        final = [e for e, _ in all_entries_scored][:max_chunks]

        est_tokens = sum(len(e.content.split()) * 2 for e in final)

        return MemoryRetrieval(
            entries=final,
            banks_hit=banks_hit,
            query=query,
            tokens_est=est_tokens,
        )

    def get_prior_run(self, task_text: str, top_k: int = 2) -> list[MemEntry]:
        """Find prior runs most similar to the current task."""
        return self._stores[MemoryBank.RUN].retrieve(task_text, top_k=top_k)

    def get_errors(self) -> list[MemEntry]:
        """Return all current error entries (to avoid repeating them)."""
        return [e for e in self._stores[MemoryBank.TASK]._entries
                if "error" in (e.tags or []) and not e.is_expired()]

    def get_user_prefs(self) -> list[MemEntry]:
        return self._stores[MemoryBank.USER].get_all_recent(10)

    # ── Clear / reset ─────────────────────────────────────────────────────────

    def clear_task_memory(self):
        """Call at the start of each new task."""
        self._stores[MemoryBank.TASK].clear()

    # ── Debug / export ────────────────────────────────────────────────────────

    def export_summary(self) -> dict:
        return {
            bank.value: {
                "count": store.count,
                "recent": [e.to_context_line() for e in
                           store.get_all_recent(3)],
            }
            for bank, store in self._stores.items()
        }

    def stats(self) -> dict:
        return {bank.value: store.count for bank, store in self._stores.items()}
