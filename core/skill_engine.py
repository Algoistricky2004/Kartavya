"""
skill_engine.py — Skill retrieval, adaptation, creation, and versioning.

This is the brain of reusable knowledge in the Agent OS.

Responsibilities:
  1. Retrieve: find the best-matching skill for the current task
  2. Adapt: tailor the skill's steps to this specific task + model tier
  3. Create: synthesize a new skill from a successful workflow
  4. Version: bump version on updates, retire bad skills
  5. Persist: save skills to YAML in .agent_os/skills/

Skills are YAML files. The engine loads them at startup and keeps an
in-memory index for fast retrieval.

Retrieval is TF-IDF keyword similarity — no embeddings, works offline,
works on 3b models. Fast enough to call on every task.
"""

from __future__ import annotations

import re
import time
import json
import uuid
import math
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillStep:
    step_id:     str
    description: str
    tools:       list[str]
    prompt_hint: str
    expected_output: str = ""

@dataclass
class Skill:
    name:             str
    version:          str
    description:      str
    tags:             list[str]
    steps:            list[SkillStep]
    success_criteria: list[str]
    token_estimate:   int
    max_steps:        int
    task_type:        str = "mixed"
    source_file:      str = ""

    # Runtime stats
    usage_count:   int   = 0
    success_count: int   = 0
    last_used:     float = 0.0
    created_at:    float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 1.0
        return self.success_count / self.usage_count

    def is_healthy(self) -> bool:
        """Skill is considered healthy if it has a decent success rate."""
        if self.usage_count < 3:
            return True  # Not enough data to judge
        return self.success_rate >= 0.5

    def record_use(self, success: bool):
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.last_used = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "task_type": self.task_type,
            "steps": [
                {
                    "id": s.step_id,
                    "description": s.description,
                    "tools": s.tools,
                    "prompt_hint": s.prompt_hint,
                }
                for s in self.steps
            ],
            "success_criteria": self.success_criteria,
            "token_estimate": self.token_estimate,
            "max_steps": self.max_steps,
        }


@dataclass
class SkillMatch:
    skill:     Skill
    score:     float   # 0.0-1.0
    rationale: str


# ─────────────────────────────────────────────────────────────────────────────
# Builtin skills (hardcoded fallback if no YAML available)
# ─────────────────────────────────────────────────────────────────────────────

_BUILTIN_SKILLS: list[dict] = [
    {
        "name": "debug_workflow",
        "version": "1.0.0",
        "description": "Standard debugging: search for error → read code → identify cause → fix → verify",
        "tags": ["debug", "fix", "error", "bug", "failing", "test", "exception"],
        "task_type": "coding",
        "steps": [
            {"id": "find_error",    "description": "Search for error pattern",
             "tools": ["grep", "find_files"],
             "prompt_hint": "Search for the error message or failing location."},
            {"id": "read_context",  "description": "Read relevant source file",
             "tools": ["read_file"],
             "prompt_hint": "Read the file where the error occurs. Focus on relevant lines."},
            {"id": "apply_fix",     "description": "Write corrected code",
             "tools": ["write_file"],
             "prompt_hint": "Apply the minimal fix. Don't rewrite more than needed."},
            {"id": "verify",        "description": "Run tests to confirm fix",
             "tools": ["bash"],
             "prompt_hint": "Run tests. Check exit code. Report pass/fail."},
        ],
        "success_criteria": ["tests pass", "no error message"],
        "token_estimate": 15000,
        "max_steps": 10,
    },
    {
        "name": "grep_then_edit",
        "version": "1.0.0",
        "description": "Find all occurrences of a pattern, then update each one",
        "tags": ["refactor", "rename", "update", "grep", "find-replace", "change"],
        "task_type": "coding",
        "steps": [
            {"id": "find_all",      "description": "Find all occurrences",
             "tools": ["grep", "find_files"],
             "prompt_hint": "Use grep to find all files and line numbers with the pattern."},
            {"id": "plan_changes",  "description": "List files that need changing",
             "tools": [],
             "prompt_hint": "List each file and what change is needed."},
            {"id": "apply_changes", "description": "Apply changes to each file",
             "tools": ["read_file", "write_file"],
             "prompt_hint": "For each file: read it, apply the change, write it back."},
            {"id": "verify",        "description": "Confirm all occurrences updated",
             "tools": ["grep", "bash"],
             "prompt_hint": "Grep again for old pattern — should be zero matches."},
        ],
        "success_criteria": ["zero matches for old pattern", "tests pass"],
        "token_estimate": 12000,
        "max_steps": 15,
    },
    {
        "name": "test_and_fix",
        "version": "1.0.0",
        "description": "Run tests, identify failures, fix them one by one",
        "tags": ["test", "ci", "failing", "pytest", "jest", "pass", "fix"],
        "task_type": "coding",
        "steps": [
            {"id": "run_tests",        "description": "Run test suite, capture output",
             "tools": ["bash"],
             "prompt_hint": "Run test command. Capture full output including failures."},
            {"id": "analyze_failures", "description": "Parse which tests fail and why",
             "tools": [],
             "prompt_hint": "List each failing test with error message and file location."},
            {"id": "fix_one",          "description": "Fix the first failing test",
             "tools": ["read_file", "grep", "write_file"],
             "prompt_hint": "Fix only the FIRST failing test. Don't touch passing tests."},
            {"id": "verify_one",       "description": "Re-run to confirm fix",
             "tools": ["bash"],
             "prompt_hint": "Run tests again. Check if failure is now passing."},
        ],
        "success_criteria": ["all tests pass"],
        "token_estimate": 18000,
        "max_steps": 20,
    },
    {
        "name": "research_then_report",
        "version": "1.0.0",
        "description": "Gather information about a codebase or topic, then produce a summary",
        "tags": ["research", "analyze", "summarize", "understand", "explain", "find", "list"],
        "task_type": "research",
        "steps": [
            {"id": "survey_files",     "description": "List relevant files and structure",
             "tools": ["find_files", "list_dir"],
             "prompt_hint": "Map the relevant files and directory structure first."},
            {"id": "read_key_files",   "description": "Read the most relevant files",
             "tools": ["read_file", "grep"],
             "prompt_hint": "Read the files most relevant to the question."},
            {"id": "synthesize",       "description": "Synthesize findings into answer",
             "tools": [],
             "prompt_hint": "Produce a clear, specific answer grounded in what you read."},
        ],
        "success_criteria": ["specific answer provided", "grounded in file contents"],
        "token_estimate": 10000,
        "max_steps": 8,
    },
    {
        "name": "install_and_run",
        "version": "1.0.0",
        "description": "Install dependencies and run a project or script",
        "tags": ["install", "run", "setup", "execute", "launch", "start", "pip", "npm"],
        "task_type": "automation",
        "steps": [
            {"id": "check_env",    "description": "Check environment and existing deps",
             "tools": ["bash", "list_dir"],
             "prompt_hint": "Check what's installed. Read requirements/package.json if present."},
            {"id": "install_deps", "description": "Install missing dependencies",
             "tools": ["bash"],
             "prompt_hint": "Install dependencies. Check for errors."},
            {"id": "run_it",       "description": "Run the script or command",
             "tools": ["bash"],
             "prompt_hint": "Execute the target. Capture output and exit code."},
        ],
        "success_criteria": ["exit code 0", "no error output"],
        "token_estimate": 8000,
        "max_steps": 8,
    },
    {
        "name": "create_file_from_spec",
        "version": "1.0.0",
        "description": "Write a new file based on a specification or description",
        "tags": ["create", "write", "generate", "produce", "new file", "add"],
        "task_type": "coding",
        "steps": [
            {"id": "read_context",  "description": "Read related files for context",
             "tools": ["read_file", "list_dir"],
             "prompt_hint": "Read any related files to understand conventions and imports."},
            {"id": "write_file",    "description": "Write the new file",
             "tools": ["write_file"],
             "prompt_hint": "Write the complete file. Follow the same conventions as existing code."},
            {"id": "verify",        "description": "Verify the file is correct",
             "tools": ["bash", "read_file"],
             "prompt_hint": "Run syntax check or import test. Confirm file was created."},
        ],
        "success_criteria": ["file created", "no syntax errors"],
        "token_estimate": 10000,
        "max_steps": 8,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF retriever
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def _tfidf_score(query_tokens: list[str], doc_tokens: list[str],
                  df: dict[str, int], n_docs: int) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_freq = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        if term not in doc_freq:
            continue
        tf = doc_freq[term] / doc_len
        idf = math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1
        score += tf * idf
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class SkillEngine:
    """
    Manages the skill lifecycle: load → match → adapt → create → persist.
    """

    MATCH_THRESHOLD = 0.05    # minimum score to consider a skill relevant
    PROMOTE_THRESHOLD = 3     # times a workflow succeeds before promoting to skill

    def __init__(
        self,
        skill_dirs: Optional[list[str]] = None,
        generated_dir: str = ".agent_os/skills",
    ):
        self.generated_dir = Path(generated_dir)
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        self._skills: list[Skill] = []
        self._df: dict[str, int] = {}  # document frequency for IDF

        # Load builtins
        for raw in _BUILTIN_SKILLS:
            self._register(self._from_dict(raw))

        # Load YAML files from disk
        dirs = list(skill_dirs or [])
        dirs.append(str(self.generated_dir))
        for d in dirs:
            self._load_from_dir(d)

        self._rebuild_df()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_from_dir(self, directory: str):
        """Load all YAML skill files from a directory."""
        if not HAS_YAML:
            return
        path = Path(directory)
        if not path.exists():
            return
        for f in path.glob("*.yaml"):
            try:
                with open(f) as fp:
                    raw = yaml.safe_load(fp)
                if raw and isinstance(raw, dict):
                    skill = self._from_dict(raw, source_file=str(f))
                    self._register(skill)
            except Exception:
                pass

    def _from_dict(self, d: dict, source_file: str = "") -> Skill:
        steps = [
            SkillStep(
                step_id=s.get("id", f"step_{i}"),
                description=s.get("description", ""),
                tools=s.get("tools", []),
                prompt_hint=s.get("prompt_hint", ""),
                expected_output=s.get("expected_output", ""),
            )
            for i, s in enumerate(d.get("steps", []))
        ]
        return Skill(
            name=d.get("name", "unnamed"),
            version=d.get("version", "1.0.0"),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            steps=steps,
            success_criteria=d.get("success_criteria", []),
            token_estimate=d.get("token_estimate", 10000),
            max_steps=d.get("max_steps", 10),
            task_type=d.get("task_type", "mixed"),
            source_file=source_file,
        )

    def _register(self, skill: Skill):
        """Add skill to index, deduplicating by name+version."""
        existing = next((s for s in self._skills if s.name == skill.name), None)
        if existing:
            # Keep higher version
            if self._version_tuple(skill.version) > self._version_tuple(existing.version):
                self._skills.remove(existing)
                self._skills.append(skill)
        else:
            self._skills.append(skill)

    def _rebuild_df(self):
        """Rebuild document frequency index for IDF computation."""
        self._df = {}
        for skill in self._skills:
            doc = " ".join([
                skill.name, skill.description,
                " ".join(skill.tags),
                " ".join(s.description for s in skill.steps),
            ])
            for term in set(_tokenize(doc)):
                self._df[term] = self._df.get(term, 0) + 1

    def _version_tuple(self, v: str) -> tuple:
        try:
            return tuple(int(x) for x in v.split("."))
        except Exception:
            return (0,)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        task_type: str = "",
        top_k: int = 3,
    ) -> list[SkillMatch]:
        """
        Find best-matching skills for a query.
        Returns list of SkillMatch sorted by score descending.
        """
        query_tokens = _tokenize(query)
        n_docs = len(self._skills)
        matches = []

        for skill in self._skills:
            # Build skill document
            doc_text = " ".join([
                skill.name, skill.description,
                " ".join(skill.tags),
                skill.task_type,
                " ".join(s.description for s in skill.steps),
            ])
            doc_tokens = _tokenize(doc_text)

            # TF-IDF score
            tfidf = _tfidf_score(query_tokens, doc_tokens, self._df, n_docs)

            # Tag overlap bonus
            task_terms = set(query_tokens)
            tag_overlap = len(task_terms & set(skill.tags)) / (len(skill.tags) + 1)

            # Task type match bonus
            type_bonus = 0.2 if task_type and skill.task_type == task_type else 0.0

            # Health penalty (bad skills score lower)
            health = 1.0 if skill.is_healthy() else 0.5

            score = (tfidf * 0.6 + tag_overlap * 0.3 + type_bonus) * health

            if score >= self.MATCH_THRESHOLD:
                matches.append(SkillMatch(
                    skill=skill,
                    score=round(score, 4),
                    rationale=self._explain_match(skill, query_tokens, tag_overlap, type_bonus),
                ))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    def _explain_match(self, skill: Skill, query_tokens: list[str],
                        tag_overlap: float, type_bonus: float) -> str:
        matched_tags = [t for t in skill.tags if t in query_tokens]
        parts = []
        if matched_tags:
            parts.append(f"tags: {', '.join(matched_tags[:3])}")
        if type_bonus > 0:
            parts.append(f"type: {skill.task_type}")
        return "; ".join(parts) if parts else "keyword similarity"

    # ── Adaptation ────────────────────────────────────────────────────────────

    def adapt_for_prompt(
        self,
        skill: Skill,
        task: str,
        max_steps: int = 99,
        depth: str = "standard",  # "shallow" | "standard" | "deep"
    ) -> str:
        """
        Convert a skill into a prompt hint for the agent.

        shallow  → just the step names (used for SLMs)
        standard → step names + tool hints
        deep     → full step descriptions + prompt hints
        """
        steps = skill.steps[:max_steps]
        lines = [
            f"Reusable skill matched: '{skill.name}' (v{skill.version})",
            f"Description: {skill.description[:120]}",
            "",
            "Suggested workflow:",
        ]
        for i, step in enumerate(steps, 1):
            if depth == "shallow":
                lines.append(f"  {i}. {step.description}")
            elif depth == "standard":
                tools_str = f"[tools: {', '.join(step.tools)}] " if step.tools else ""
                lines.append(f"  {i}. {step.description} {tools_str}")
            else:  # deep
                tools_str = f"\n     Tools: {', '.join(step.tools)}" if step.tools else ""
                hint_str  = f"\n     Hint: {step.prompt_hint}" if step.prompt_hint else ""
                lines.append(f"  {i}. {step.description}{tools_str}{hint_str}")

        if skill.success_criteria:
            lines.append("")
            lines.append(f"Done when: {'; '.join(skill.success_criteria[:3])}")

        lines.append("")
        lines.append("Adapt this workflow to the specific task. Don't follow it blindly.")

        return "\n".join(lines)

    # ── Skill creation from workflow ──────────────────────────────────────────

    def create_from_workflow(
        self,
        task: str,
        tool_calls: list[dict],   # [{tool, args, result, success}]
        outcome: str,
        task_type: str = "mixed",
        success: bool = True,
    ) -> Optional[Skill]:
        """
        Synthesize a new skill from a successful workflow.
        Called by AgentOS after a successful task completion.

        Returns the new skill if it's worth saving, None otherwise.
        """
        if not success:
            return None
        if len(tool_calls) < 2:
            return None  # too short to be a reusable skill

        # Check if a very similar skill already exists
        query = task[:200]
        existing = self.retrieve(query, task_type=task_type, top_k=1)
        if existing and existing[0].score > 0.6:
            # Don't create duplicates — update existing instead
            self._update_skill_stats(existing[0].skill, success=True)
            return None

        # Build steps from tool calls
        steps = []
        seen_tools = set()
        for i, call in enumerate(tool_calls):
            tool = call.get("tool", "bash")
            args = call.get("args", {})
            result_preview = str(call.get("result", ""))[:100]
            step = SkillStep(
                step_id=f"step_{i+1}",
                description=self._describe_tool_call(tool, args),
                tools=[tool],
                prompt_hint=result_preview,
            )
            steps.append(step)
            seen_tools.add(tool)

        # Generate name and tags from task
        name = self._generate_skill_name(task)
        tags = self._generate_tags(task, list(seen_tools))

        skill = Skill(
            name=name,
            version="1.0.0",
            description=f"Auto-generated from: {task[:120]}",
            tags=tags,
            steps=steps,
            success_criteria=[outcome[:120]] if outcome else ["task completed"],
            token_estimate=len(tool_calls) * 2000,
            max_steps=len(tool_calls) + 3,
            task_type=task_type,
        )
        return skill

    def save_skill(self, skill: Skill) -> bool:
        """Persist skill to YAML and register in memory."""
        self._register(skill)
        self._rebuild_df()

        if not HAS_YAML:
            # Save as JSON fallback
            path = self.generated_dir / f"{skill.name}.json"
            try:
                with open(path, "w") as f:
                    json.dump(skill.to_dict(), f, indent=2)
                return True
            except Exception:
                return False

        path = self.generated_dir / f"{skill.name}.yaml"
        try:
            with open(path, "w") as f:
                yaml.dump(skill.to_dict(), f, default_flow_style=False, sort_keys=False)
            return True
        except Exception:
            return False

    def bump_version(self, skill: Skill) -> Skill:
        """Increment patch version."""
        parts = self._version_tuple(skill.version)
        if len(parts) >= 3:
            new_ver = f"{parts[0]}.{parts[1]}.{parts[2]+1}"
        else:
            new_ver = f"{skill.version}.1"
        skill.version = new_ver
        return skill

    # ── Stats update ──────────────────────────────────────────────────────────

    def _update_skill_stats(self, skill: Skill, success: bool):
        skill.record_use(success)

    def record_use(self, skill_name: str, success: bool):
        """Call this after a skill was used during task execution."""
        skill = next((s for s in self._skills if s.name == skill_name), None)
        if skill:
            skill.record_use(success)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _describe_tool_call(self, tool: str, args: dict) -> str:
        if tool == "bash":
            cmd = str(args.get("command", ""))[:60]
            return f"Run bash: {cmd}"
        if tool in ("read_file", "read"):
            return f"Read file: {args.get('path', '')}"
        if tool in ("write_file", "write"):
            return f"Write file: {args.get('path', '')}"
        if tool == "grep":
            return f"Search for: {args.get('pattern', '')}"
        if tool == "find_files":
            return f"Find files: {args.get('pattern', '')}"
        return f"Use {tool}"

    def _generate_skill_name(self, task: str) -> str:
        words = re.findall(r'[a-z]+', task.lower())[:4]
        base = "_".join(w for w in words if len(w) > 2)
        suffix = hashlib.md5(task.encode()).hexdigest()[:4]
        return f"{base}_{suffix}" if base else f"skill_{suffix}"

    def _generate_tags(self, task: str, tools: list[str]) -> list[str]:
        task_words = re.findall(r'\b[a-z]{3,}\b', task.lower())
        stop_words = {
            "the", "and", "for", "this", "that", "with", "from", "all",
            "are", "was", "its", "our", "but", "not", "can", "has", "have",
        }
        meaningful = [w for w in task_words if w not in stop_words][:6]
        tool_tags = [t for t in tools if t not in ("task_done", "memory_read")]
        return list(dict.fromkeys(meaningful + tool_tags))[:10]

    # ── List & inspect ────────────────────────────────────────────────────────

    def list_skills(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "version": s.version,
                "task_type": s.task_type,
                "tags": s.tags[:5],
                "usage_count": s.usage_count,
                "success_rate": round(s.success_rate, 2),
                "healthy": s.is_healthy(),
            }
            for s in sorted(self._skills, key=lambda x: x.usage_count, reverse=True)
        ]

    def get_skill(self, name: str) -> Optional[Skill]:
        return next((s for s in self._skills if s.name == name), None)

    @property
    def count(self) -> int:
        return len(self._skills)
