"""
memory.py — Tiered memory system for small context window models.

Memory tiers:
  HOT   → current context window (managed by context_manager.py)
  WARM  → structured session memory (this file) — persisted to disk
  COLD  → long-term facts (project, user preferences, skill patterns)

Design for small context windows:
  - Memory is NEVER fully dumped into context
  - RAG-style: retrieval on demand using keyword + TF-IDF similarity
  - Entries have priority (high/normal/low) + TTL
  - Memory is compressed before injection (one-line summaries)
  - Cross-session persistence via JSON files
"""

import json
import time
import math
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import Counter
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Memory Entry
# ─────────────────────────────────────────────────────────────────────────────

class MemoryPriority(str, Enum):
    HIGH   = "high"    # Always included in context if room
    NORMAL = "normal"  # Included if relevant
    LOW    = "low"     # Only included if directly relevant

class MemorySection(str, Enum):
    CURRENT_TASKS  = "current_tasks"
    BLOCKED_BY     = "blocked_by"
    USER_PREFS     = "user_preferences"
    PROJECT_FACTS  = "project_facts"
    DECISIONS      = "decisions"
    ERRORS_SEEN    = "errors_seen"     # past errors → avoid repeating
    SKILL_PATTERNS = "skill_patterns"
    VERIFIED_FACTS = "verified_facts"  # ground truth from tool calls

@dataclass
class MemoryEntry:
    entry_id: str
    section: MemorySection
    content: str
    priority: MemoryPriority = MemoryPriority.NORMAL
    tags: list[str]          = field(default_factory=list)
    created_at: float        = field(default_factory=time.time)
    updated_at: float        = field(default_factory=time.time)
    access_count: int        = 0
    expires_at: Optional[float] = None  # None = never expires
    source: str              = "agent"  # "agent", "tool", "user", "verified"

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    def matches_query(self, query: str) -> float:
        """
        Simple TF-IDF-like relevance score for retrieval.
        No embeddings needed — works entirely offline.
        """
        query_terms = set(re.findall(r'\w+', query.lower()))
        content_terms = re.findall(r'\w+', self.content.lower())
        if not query_terms or not content_terms:
            return 0.0

        content_freq = Counter(content_terms)
        total_content_terms = len(content_terms)

        score = 0.0
        for term in query_terms:
            if term in content_freq:
                tf = content_freq[term] / total_content_terms
                # IDF approximation: shorter terms are more common → lower IDF
                idf = math.log(1 + 10 / max(1, len(term)))
                score += tf * idf

        # Boost for high-priority entries
        priority_boost = {"high": 1.5, "normal": 1.0, "low": 0.7}
        score *= priority_boost.get(self.priority.value, 1.0)

        # Boost for recently accessed / updated
        age_hours = (time.time() - self.updated_at) / 3600
        recency_boost = 1.0 / (1 + age_hours * 0.05)  # decays over 20 hours
        score *= recency_boost

        # Boost for verified facts
        if self.source == "verified":
            score *= 1.3

        return score


# ─────────────────────────────────────────────────────────────────────────────
# Memory Store
# ─────────────────────────────────────────────────────────────────────────────

class MemoryStore:
    """
    In-memory store with disk persistence.
    Supports retrieval, eviction, and RAG-style search.
    """

    MAX_ENTRIES = 200          # hard limit per agent
    MAX_SECTION_ENTRIES = 30   # per section
    EVICT_TO = 150             # evict down to this when at MAX

    def __init__(self, agent_id: str, memory_dir: str = ".agent_os/memory"):
        self.agent_id   = agent_id
        self._dir       = Path(memory_dir) / agent_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[MemoryEntry] = []
        self._load()

    # ── Write ─────────────────────────────────────────────────────────────

    def upsert(self, section: MemorySection, content: str,
               priority: MemoryPriority = MemoryPriority.NORMAL,
               tags: list[str] = None, ttl_hours: float = None,
               source: str = "agent") -> MemoryEntry:
        """
        Add or update a memory entry.
        Deduplication: if same section+content hash exists, update it.
        """
        entry_id = self._make_id(section, content)
        expires_at = time.time() + ttl_hours * 3600 if ttl_hours else None

        # Check for existing entry with same id
        for existing in self._entries:
            if existing.entry_id == entry_id:
                existing.content    = content
                existing.updated_at = time.time()
                existing.priority   = priority
                existing.tags       = tags or []
                existing.expires_at = expires_at
                existing.source     = source
                self._save()
                return existing

        # New entry
        entry = MemoryEntry(
            entry_id=entry_id,
            section=section,
            content=content,
            priority=priority,
            tags=tags or [],
            source=source,
            expires_at=expires_at,
        )
        self._entries.append(entry)
        self._maybe_evict()
        self._save()
        return entry

    def delete(self, entry_id: str) -> bool:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.entry_id != entry_id]
        if len(self._entries) < before:
            self._save()
            return True
        return False

    def update_task_status(self, task_id: str, status: str, note: str = ""):
        """Convenience: update a task entry by ID tag."""
        for entry in self._entries:
            if task_id in entry.tags and entry.section == MemorySection.CURRENT_TASKS:
                entry.content = f"[{status.upper()}] {entry.content.split(']',1)[-1].strip()}"
                if note:
                    entry.content += f" — {note}"
                entry.updated_at = time.time()
                self._save()
                return
        # If not found, create
        self.upsert(
            MemorySection.CURRENT_TASKS,
            f"[{status.upper()}] {task_id}: {note}",
            priority=MemoryPriority.HIGH,
            tags=[task_id],
        )

    # ── Read ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n: int = 8,
                 section_filter: MemorySection = None,
                 min_priority: MemoryPriority = None) -> list[MemoryEntry]:
        """
        RAG-style retrieval. Returns top-n most relevant entries.
        Used to build the 'active_memory' zone in context.
        """
        # Evict expired
        self._evict_expired()

        entries = self._entries
        if section_filter:
            entries = [e for e in entries if e.section == section_filter]
        if min_priority == MemoryPriority.HIGH:
            entries = [e for e in entries if e.priority == MemoryPriority.HIGH]

        # Score each entry
        scored = []
        for entry in entries:
            score = entry.matches_query(query)
            if score > 0:
                scored.append((score, entry))

        # Sort by score descending, then by priority
        scored.sort(key=lambda x: (x[0], {"high":3,"normal":2,"low":1}[x[1].priority.value]),
                    reverse=True)

        results = [e for _, e in scored[:n]]

        # Update access count
        for entry in results:
            entry.access_count += 1

        return results

    def get_section(self, section: MemorySection) -> list[MemoryEntry]:
        """Get all entries for a section, sorted by priority + recency."""
        entries = [e for e in self._entries
                   if e.section == section and not e.is_expired()]
        entries.sort(
            key=lambda e: (
                {"high": 3, "normal": 2, "low": 1}[e.priority.value],
                e.updated_at
            ),
            reverse=True
        )
        return entries

    def get_all_high_priority(self) -> list[MemoryEntry]:
        """Always include these in context regardless of query."""
        return [e for e in self._entries
                if e.priority == MemoryPriority.HIGH and not e.is_expired()]

    def get_verified_facts(self) -> list[MemoryEntry]:
        return [e for e in self._entries
                if e.section == MemorySection.VERIFIED_FACTS and not e.is_expired()]

    def build_context_chunks(self, query: str, max_chunks: int = 8,
                              max_tokens: int = 512) -> list[str]:
        """
        Build list of memory strings ready for injection into context.
        Combines high-priority entries + relevant retrieved entries.
        Respects token budget.
        """
        from .context_manager import estimate_tokens

        chunks = []
        tokens_used = 0

        # Always include: HIGH priority entries
        for entry in self.get_all_high_priority():
            chunk = self._format_entry(entry)
            t = estimate_tokens(chunk)
            if tokens_used + t > max_tokens:
                break
            chunks.append(chunk)
            tokens_used += t

        # Then: RAG-retrieved relevant entries
        retrieved = self.retrieve(query, n=max_chunks * 2)
        for entry in retrieved:
            if entry in [e for e in self.get_all_high_priority()]:
                continue  # already included
            chunk = self._format_entry(entry)
            t = estimate_tokens(chunk)
            if tokens_used + t > max_tokens or len(chunks) >= max_chunks:
                break
            chunks.append(chunk)
            tokens_used += t

        return chunks

    def get_stats(self) -> dict:
        sections = {}
        for section in MemorySection:
            entries = self.get_section(section)
            sections[section.value] = len(entries)
        return {
            "total_entries": len(self._entries),
            "sections": sections,
            "high_priority": len(self.get_all_high_priority()),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self):
        path = self._dir / "memory.json"
        data = [asdict(e) for e in self._entries if not e.is_expired()]
        path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self):
        path = self._dir / "memory.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._entries = []
            for item in data:
                # Reconstruct enums
                item["section"]  = MemorySection(item["section"])
                item["priority"] = MemoryPriority(item["priority"])
                entry = MemoryEntry(**item)
                if not entry.is_expired():
                    self._entries.append(entry)
        except (json.JSONDecodeError, KeyError, ValueError):
            self._entries = []

    # ── Private helpers ───────────────────────────────────────────────────

    def _make_id(self, section: MemorySection, content: str) -> str:
        h = hashlib.sha256(f"{section.value}:{content[:200]}".encode()).hexdigest()
        return h[:16]

    def _format_entry(self, entry: MemoryEntry) -> str:
        prefix = {
            MemorySection.CURRENT_TASKS:  "TASK",
            MemorySection.BLOCKED_BY:     "BLOCKED",
            MemorySection.USER_PREFS:     "PREF",
            MemorySection.PROJECT_FACTS:  "FACT",
            MemorySection.DECISIONS:      "DECIDED",
            MemorySection.ERRORS_SEEN:    "PAST ERROR",
            MemorySection.SKILL_PATTERNS: "PATTERN",
            MemorySection.VERIFIED_FACTS: "VERIFIED",
        }.get(entry.section, entry.section.value.upper())
        verified_tag = " ✓" if entry.source == "verified" else ""
        return f"[{prefix}{verified_tag}] {entry.content}"

    def _evict_expired(self):
        before = len(self._entries)
        self._entries = [e for e in self._entries if not e.is_expired()]
        if len(self._entries) < before:
            self._save()

    def _maybe_evict(self):
        """Evict when at capacity. Keep high-priority + most-used entries."""
        if len(self._entries) <= self.MAX_ENTRIES:
            return
        # Separate high-priority (never evict)
        high_priority = [e for e in self._entries if e.priority == MemoryPriority.HIGH]
        evictable = [e for e in self._entries if e.priority != MemoryPriority.HIGH]
        # Sort evictable: least useful first
        evictable.sort(
            key=lambda e: (e.access_count, e.updated_at)
        )
        # Keep top entries
        keep_n = self.EVICT_TO - len(high_priority)
        kept = evictable[-keep_n:] if keep_n > 0 else []
        self._entries = high_priority + kept
        self._save()

    def export_markdown(self) -> str:
        """Generate MEMORY.md for human reading / debugging."""
        lines = [f"# Agent Memory — {self.agent_id}\n",
                 f"*Generated at {time.strftime('%Y-%m-%d %H:%M:%S UTC')}*\n"]
        for section in MemorySection:
            entries = self.get_section(section)
            if not entries:
                continue
            lines.append(f"\n## {section.value.replace('_', ' ').title()}")
            for e in entries[:10]:
                prio_tag = {"high": "🔴", "normal": "🟡", "low": "⚪"}[e.priority.value]
                lines.append(f"- {prio_tag} {e.content}")
        return "\n".join(lines)
