"""
knowledge_retriever.py — External knowledge retrieval for the Agent OS.

Pulls external or repo-level context when the task requires it.
Keeps a strict separation: internal memory vs external knowledge.

External knowledge sources:
  - Web search (DuckDuckGo via simple HTTP — no API key required)
  - Repo README and key docs (read from disk)
  - Git log / blame (git commands)
  - Package docs (pypi, npm info)

All retrieved knowledge is clearly labeled [EXTERNAL] or [REPO]
so the model knows it came from outside its training data.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class KnowledgeChunk:
    source:  str    # "web", "repo", "git", "package"
    label:   str    # e.g., "README", "git log", "DuckDuckGo"
    content: str
    url:     str    = ""
    retrieved_at: float = 0.0

    def to_context_block(self, max_chars: int = 600) -> str:
        content = self.content[:max_chars]
        if len(self.content) > max_chars:
            content += "... [truncated]"
        return f"[{self.source.upper()} KNOWLEDGE — {self.label}]\n{content}"


class KnowledgeRetriever:
    """
    Conditionally retrieves external knowledge.
    Called only when TaskIntake.needs_external_knowledge is True.
    """

    def __init__(self, cwd: str = "."):
        self.cwd = Path(cwd)

    def retrieve_if_needed(
        self,
        task: str,
        needs_external: bool,
        tech_stack: list[str] = None,
        max_chunks: int = 2,
    ) -> list[KnowledgeChunk]:
        """
        Entry point. Returns relevant knowledge chunks or empty list.
        """
        chunks = []

        # Always check local repo docs (lightweight)
        repo_chunks = self._get_repo_context(task, max_chunks=1)
        chunks.extend(repo_chunks)

        # Web search only when explicitly needed
        if needs_external and len(chunks) < max_chunks:
            web_chunks = self._web_search(task, tech_stack or [], top_k=1)
            chunks.extend(web_chunks[:max_chunks - len(chunks)])

        return chunks[:max_chunks]

    # ── Repo context ──────────────────────────────────────────────────────────

    def _get_repo_context(self, task: str, max_chunks: int = 2) -> list[KnowledgeChunk]:
        chunks = []

        # Read README if it exists and is relevant
        readme = self._find_readme()
        if readme:
            content = self._read_file_safe(readme, max_chars=800)
            if content and self._is_relevant(content, task):
                chunks.append(KnowledgeChunk(
                    source="repo",
                    label=f"README ({readme.name})",
                    content=content,
                    retrieved_at=time.time(),
                ))

        if len(chunks) >= max_chunks:
            return chunks

        # Git log (recent 5 commits) — useful for understanding recent changes
        git_log = self._get_git_log(max_lines=10)
        if git_log and self._is_relevant(git_log, task):
            chunks.append(KnowledgeChunk(
                source="git",
                label="recent git log",
                content=git_log,
                retrieved_at=time.time(),
            ))

        return chunks[:max_chunks]

    def _find_readme(self) -> Optional[Path]:
        for name in ["README.md", "README.rst", "README.txt", "README"]:
            p = self.cwd / name
            if p.exists():
                return p
        return None

    def _read_file_safe(self, path: Path, max_chars: int = 800) -> str:
        try:
            with open(path, errors="replace") as f:
                return f.read(max_chars)
        except Exception:
            return ""

    def _get_git_log(self, max_lines: int = 10) -> str:
        try:
            r = subprocess.run(
                ["git", "log", "--oneline", f"-{max_lines}"],
                capture_output=True, text=True, timeout=5, cwd=str(self.cwd)
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    def _is_relevant(self, content: str, task: str) -> bool:
        """Simple relevance check: keyword overlap."""
        task_words = set(re.findall(r'\b\w{4,}\b', task.lower()))
        content_words = set(re.findall(r'\b\w{4,}\b', content.lower()))
        stop = {"this", "that", "with", "from", "have", "will", "your", "what",
                "which", "when", "where", "there", "their", "about", "more",
                "also", "than", "some"}
        task_words -= stop
        content_words -= stop
        if not task_words:
            return True
        overlap = len(task_words & content_words) / len(task_words)
        return overlap > 0.2

    # ── Web search ────────────────────────────────────────────────────────────

    def _web_search(self, task: str, tech_stack: list[str],
                     top_k: int = 2) -> list[KnowledgeChunk]:
        """
        DuckDuckGo instant answer — no API key, no rate limits for light usage.
        Returns at most top_k results.
        """
        try:
            import urllib.request
            import urllib.parse
            import json as _json

            query = self._build_search_query(task, tech_stack)
            encoded = urllib.parse.quote_plus(query)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"

            req = urllib.request.Request(
                url, headers={"User-Agent": "AgentOS/1.0 (research)"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read().decode())

            chunks = []

            # Abstract text
            abstract = data.get("AbstractText", "")
            if abstract:
                chunks.append(KnowledgeChunk(
                    source="web",
                    label=f"DuckDuckGo: {query[:50]}",
                    content=abstract[:600],
                    url=data.get("AbstractURL", ""),
                    retrieved_at=time.time(),
                ))

            # Related topics
            if len(chunks) < top_k:
                for topic in data.get("RelatedTopics", [])[:3]:
                    text = topic.get("Text", "")
                    if text and len(text) > 30:
                        chunks.append(KnowledgeChunk(
                            source="web",
                            label="DuckDuckGo related",
                            content=text[:300],
                            retrieved_at=time.time(),
                        ))
                    if len(chunks) >= top_k:
                        break

            return chunks[:top_k]

        except Exception:
            return []

    def _build_search_query(self, task: str, tech_stack: list[str]) -> str:
        # Extract most meaningful words from the task
        words = re.findall(r'\b[a-z]{4,}\b', task.lower())
        stop = {"with", "this", "that", "from", "have", "will", "your", "what",
                "when", "then", "into", "also", "more", "some", "just", "does",
                "using", "there"}
        meaningful = [w for w in words if w not in stop][:6]
        tech = tech_stack[:2] if tech_stack else []
        query_parts = list(dict.fromkeys(tech + meaningful))
        return " ".join(query_parts[:8])
