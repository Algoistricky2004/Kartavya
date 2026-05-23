"""
skill_registry.py — Skill loader and matcher for Kartavya Agent OS.

Skills are versioned YAML files that encode reusable action workflows.
Each skill describes:
  - What task patterns it applies to (tags, description keywords)
  - What steps the agent should take (ordered, with tool hints)
  - How many tokens/steps to budget
  - Success criteria

The registry:
  1. Loads all .yaml files from skills/ directory at startup
  2. Matches incoming tasks to skills using tag overlap + description similarity
  3. Returns step-level tool hints that the agent injects into its prompt
  4. Tracks usage stats per skill (for telemetry and skill evolution)

Usage:
    registry = SkillRegistry()
    skill, score = registry.match("fix the failing test in test_auth.py")
    if skill:
        step_config = registry.get_step_config(skill, step_idx=0)
        # → {"tools": ["grep", "find_files"], "hint": "Search for the error pattern..."}
"""

from __future__ import annotations

import re
import time
import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillStep:
    step_id:     str
    description: str
    tools:       list[str]
    prompt_hint: str


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
    source_file:      str = ""
    # Runtime stats
    usage_count:      int = 0
    success_count:    int = 0
    last_used:        float = 0.0

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "steps": len(self.steps),
            "max_steps": self.max_steps,
            "token_estimate": self.token_estimate,
            "usage_count": self.usage_count,
            "success_rate": round(self.success_rate, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# YAML Parser — avoids external dep issues by using stdlib as fallback
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> list[dict]:
    """
    Load one or more YAML documents from a file.
    Uses PyYAML if available, falls back to a minimal inline parser.
    Returns a list of dicts (one per --- document in the file).
    """
    try:
        import yaml
        text = path.read_text()
        # Split on --- document separator
        documents = []
        for doc in yaml.safe_load_all(text):
            if doc and isinstance(doc, dict):
                documents.append(doc)
        return documents
    except ImportError:
        pass

    # Minimal inline parser — handles simple flat YAML (no anchors, no nested lists)
    # This is only used if PyYAML is not installed
    return _parse_yaml_minimal(path.read_text())


def _parse_yaml_minimal(text: str) -> list[dict]:
    """Very minimal YAML parser for the skills format."""
    documents = []
    current: dict = {}
    current_list_key: Optional[str] = None
    current_block: Optional[str] = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped == "---":
            if current:
                documents.append(current)
            current = {}
            current_list_key = None
            current_block = None
            continue

        if not stripped or stripped.startswith("#"):
            continue

        # Nested key under a steps/list item
        if stripped.startswith("- id:"):
            if current_list_key is None:
                current_list_key = "steps"
            if "steps" not in current:
                current["steps"] = []
            current["steps"].append({"id": stripped[5:].strip()})
            continue

        if stripped.startswith("- ") and current_list_key:
            val = stripped[2:].strip().strip('"').strip("'")
            if current_list_key in current and isinstance(current[current_list_key], list):
                if current["steps"] and isinstance(current["steps"][-1], dict):
                    pass  # nested under step
                else:
                    current[current_list_key].append(val)
            continue

        if ":" in stripped and not stripped.startswith(" "):
            k, _, v = stripped.partition(":")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if v == "|":
                current_block = k
                current[k] = ""
            elif v:
                current[k] = _coerce(v)
                current_list_key = None
            else:
                current[k] = []
                current_list_key = k

    if current:
        documents.append(current)
    return [d for d in documents if d]


def _coerce(v: str):
    if v.lower() == "true":  return True
    if v.lower() == "false": return False
    try: return int(v)
    except ValueError: pass
    try: return float(v)
    except ValueError: pass
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Skill Registry
# ─────────────────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Loads all skills from the skills/ directory tree and matches them to tasks.

    File structure expected:
        skills/
          builtin/
            skills.yaml       ← multiple skills in one file (--- separated)
          user/
            my_skill.yaml     ← user-defined skills
    """

    # Minimum match score to suggest a skill (0–1 scale)
    MATCH_THRESHOLD = 0.25

    def __init__(self, skills_dir: str = None):
        if skills_dir is None:
            # Resolve relative to this file's location
            skills_dir = str(Path(__file__).parent.parent / "skills")
        self._skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}
        self._stats_file = self._skills_dir / ".skill_stats.json"
        self.load_all()
        self._load_stats()

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_all(self):
        """Scan skills directory and load all .yaml files."""
        self._skills.clear()
        if not self._skills_dir.exists():
            return

        yaml_files = list(self._skills_dir.rglob("*.yaml")) + \
                     list(self._skills_dir.rglob("*.yml"))

        for yaml_path in yaml_files:
            try:
                docs = _load_yaml(yaml_path)
                for doc in docs:
                    skill = self._parse_skill(doc, str(yaml_path))
                    if skill:
                        self._skills[skill.name] = skill
            except Exception as e:
                print(f"[SkillRegistry] Failed to load {yaml_path}: {e}")

    def _parse_skill(self, doc: dict, source: str) -> Optional[Skill]:
        """Parse a skill YAML document into a Skill object."""
        name = doc.get("name", "")
        if not name:
            return None

        # Parse steps
        steps = []
        for raw_step in doc.get("steps", []):
            if isinstance(raw_step, dict):
                steps.append(SkillStep(
                    step_id=raw_step.get("id", f"step_{len(steps)+1}"),
                    description=raw_step.get("description", ""),
                    tools=raw_step.get("tools", []),
                    prompt_hint=raw_step.get("prompt_hint", ""),
                ))

        tags = doc.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        success_criteria = doc.get("success_criteria", [])
        if isinstance(success_criteria, str):
            success_criteria = [success_criteria]

        return Skill(
            name=name,
            version=str(doc.get("version", "1.0.0")),
            description=str(doc.get("description", "")).strip(),
            tags=[str(t) for t in tags],
            steps=steps,
            success_criteria=[str(c) for c in success_criteria],
            token_estimate=int(doc.get("token_estimate", 10000)),
            max_steps=int(doc.get("max_steps", 15)),
            source_file=source,
        )

    # ── Matching ──────────────────────────────────────────────────────────────

    def match(self, task: str) -> tuple[Optional[Skill], float]:
        """
        Find the best matching skill for a task.
        Returns (skill, score) or (None, 0.0) if no skill matches.

        Scoring:
          - Tag overlap:        each matching tag contributes 0.2
          - Description words:  each matching word contributes 0.05 (cap 0.4)
          - Recent usage boost: +0.05 for skills used in last 24h
        """
        if not self._skills:
            return None, 0.0

        task_lower = task.lower()
        task_words = set(re.findall(r'\w{3,}', task_lower))  # words ≥ 3 chars
        # Remove stop words
        task_words -= {"the", "all", "and", "for", "are", "this", "that",
                       "with", "from", "into", "also"}

        best_skill = None
        best_score = 0.0

        for skill in self._skills.values():
            score = 0.0

            # Tag matching (exact)
            for tag in skill.tags:
                if tag.lower() in task_lower:
                    score += 0.20

            # Description word overlap
            desc_words = set(re.findall(r'\w{3,}', skill.description.lower()))
            overlap = task_words & desc_words
            score += min(0.40, len(overlap) * 0.05)

            # Recency boost
            if skill.last_used and (time.time() - skill.last_used) < 86400:
                score += 0.05

            # Success rate boost for well-performing skills
            if skill.usage_count >= 3 and skill.success_rate > 0.7:
                score += 0.05

            if score > best_score:
                best_score = score
                best_skill = skill

        if best_score >= self.MATCH_THRESHOLD:
            return best_skill, round(best_score, 3)
        return None, 0.0

    # ── Step config retrieval ─────────────────────────────────────────────────

    def get_step_config(self, skill: Skill, step_idx: int) -> Optional[dict]:
        """
        Get configuration for a specific step index.
        Returns dict with "tools" and "hint", or None if step doesn't exist.
        """
        if step_idx < 0 or step_idx >= len(skill.steps):
            return None
        step = skill.steps[step_idx]
        return {
            "step_id":   step.step_id,
            "tools":     step.tools,
            "hint":      step.prompt_hint,
            "description": step.description,
        }

    def get_all_step_hints(self, skill: Skill) -> str:
        """
        Format all skill steps as a compact prompt injection.
        Injected into the agent's system prompt to guide its approach.
        """
        if not skill.steps:
            return ""
        lines = [f"[Skill: {skill.name}] Recommended approach:"]
        for i, step in enumerate(skill.steps, 1):
            hint = f"Step {i}: {step.description}"
            if step.prompt_hint:
                hint += f" — {step.prompt_hint}"
            if step.tools:
                hint += f" (tools: {', '.join(step.tools)})"
            lines.append(hint)
        if skill.success_criteria:
            lines.append(f"Done when: {'; '.join(skill.success_criteria)}")
        return "\n".join(lines)

    # ── Usage tracking ────────────────────────────────────────────────────────

    def record_usage(self, skill_name: str, success: bool):
        """Call after a skill-guided task completes to track performance."""
        if skill_name not in self._skills:
            return
        skill = self._skills[skill_name]
        skill.usage_count += 1
        if success:
            skill.success_count += 1
        skill.last_used = time.time()
        self._save_stats()

    # ── Inspection ────────────────────────────────────────────────────────────

    def list_skills(self) -> list[dict]:
        """Return all skills as dicts for display."""
        return [s.to_dict() for s in sorted(self._skills.values(),
                                             key=lambda s: s.usage_count, reverse=True)]

    def get_skill(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def skill_count(self) -> int:
        return len(self._skills)

    def format_skills_list(self) -> str:
        """Human-readable skills list for CLI display."""
        if not self._skills:
            return "No skills loaded."
        lines = [f"{'NAME':<25} {'TAGS':<35} {'STEPS':<6} {'USES':<6} {'SUCCESS'}"]
        lines.append("─" * 85)
        for skill in sorted(self._skills.values(), key=lambda s: s.name):
            tags = ", ".join(skill.tags[:4])
            rate = f"{skill.success_rate*100:.0f}%" if skill.usage_count else "—"
            lines.append(
                f"{skill.name:<25} {tags:<35} {len(skill.steps):<6} "
                f"{skill.usage_count:<6} {rate}"
            )
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_stats(self):
        try:
            stats = {
                name: {
                    "usage_count": s.usage_count,
                    "success_count": s.success_count,
                    "last_used": s.last_used,
                }
                for name, s in self._skills.items()
            }
            self._stats_file.write_text(json.dumps(stats, indent=2))
        except Exception:
            pass

    def _load_stats(self):
        if not self._stats_file.exists():
            return
        try:
            stats = json.loads(self._stats_file.read_text())
            for name, data in stats.items():
                if name in self._skills:
                    s = self._skills[name]
                    s.usage_count  = data.get("usage_count", 0)
                    s.success_count = data.get("success_count", 0)
                    s.last_used    = data.get("last_used", 0.0)
        except Exception:
            pass
