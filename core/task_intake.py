"""
task_intake.py — Task normalization, classification, and risk assessment.

Stage 1 of the Agent OS runtime. Every task passes through here before
any model call, skill lookup, or memory retrieval happens.

Responsibilities:
  - Normalize: strip noise, canonicalize casing, expand abbreviations
  - Classify: detect task TYPE (coding/research/automation/ops/data/mixed)
  - Complexity: simple / moderate / complex / compound
  - Risk: low / medium / high / critical
  - Extract: entities, file references, action verbs, scope
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    CODING      = "coding"        # write / edit / debug / refactor code
    RESEARCH    = "research"      # gather info, read files, explain, summarize
    AUTOMATION  = "automation"    # run scripts, CI/CD, file ops, batch jobs
    OPS         = "ops"           # deploy, monitor, configure, install
    DATA        = "data"          # query, transform, analyze, visualize data
    CREATIVE    = "creative"      # write prose, docs, design, brainstorm
    MIXED       = "mixed"         # multiple types combined


class Complexity(str, Enum):
    SIMPLE    = "simple"      # single action, single file, no dependencies
    MODERATE  = "moderate"    # 2-5 steps, limited scope
    COMPLEX   = "complex"     # multi-step, multi-file, planning needed
    COMPOUND  = "compound"    # requires orchestration / subagents


class RiskLevel(str, Enum):
    LOW      = "low"      # read-only, reversible, no side effects
    MEDIUM   = "medium"   # writes files, minor changes
    HIGH     = "high"     # deletes, deploys, system changes
    CRITICAL = "critical" # destructive, irreversible, or security-sensitive


# ─────────────────────────────────────────────────────────────────────────────
# Output data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskIntake:
    # Identity
    intake_id:    str  = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp:    float = field(default_factory=time.time)

    # Raw and normalized input
    raw_task:     str  = ""
    task:         str  = ""          # normalized form used downstream

    # Classification
    task_type:    TaskType   = TaskType.MIXED
    complexity:   Complexity = Complexity.MODERATE
    risk:         RiskLevel  = RiskLevel.MEDIUM

    # Extracted signal
    action_verbs: list[str]  = field(default_factory=list)
    file_refs:    list[str]  = field(default_factory=list)
    tech_stack:   list[str]  = field(default_factory=list)
    entities:     list[str]  = field(default_factory=list)
    scope:        str        = "local"   # local / repo / system / external

    # Planning hints
    needs_external_knowledge: bool = False
    needs_bash:               bool = False
    needs_file_write:         bool = False
    is_multi_step:            bool = False
    requires_verification:    bool = True

    # Skill retrieval query (generated from task)
    skill_query:  str = ""

    @property
    def is_high_risk(self) -> bool:
        return self.risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    @property
    def needs_orchestration(self) -> bool:
        return self.complexity == Complexity.COMPOUND

    def summary(self) -> str:
        return (
            f"type={self.task_type.value} complexity={self.complexity.value} "
            f"risk={self.risk.value} scope={self.scope}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pattern tables
# ─────────────────────────────────────────────────────────────────────────────

_CODING_PATTERNS = [
    r'\b(write|implement|create|build|code|program|develop|add|refactor|'
    r'fix|debug|patch|update|modify|edit|change)\b.*\b(function|class|method|'
    r'module|script|test|api|endpoint|component|hook|util|helper|service)\b',
    r'\b(function|class|def|import|return|variable|bug|error|exception|'
    r'syntax|type.?error|attribute.?error|import.?error)\b',
    r'\.(py|js|ts|jsx|tsx|java|go|rs|cpp|c|h|cs|rb|php|swift|kt)\b',
    r'\b(pytest|jest|mocha|unittest|test suite|failing test|pass.?ing test)\b',
    r'\bpull request\b|\bcode review\b|\bmerge\b|\bdiff\b',
]

_RESEARCH_PATTERNS = [
    r'\b(find|search|look|grep|locate|identify|discover|list|show|display|'
    r'read|examine|inspect|analyze|understand|explain|summarize|describe|'
    r'what|how|why|where|when)\b',
    r'\b(documentation|readme|doc|spec|requirement|design|architecture|'
    r'codebase|repo|project)\b',
    r'\b(who wrote|what does|how does|what is|where is|count|how many)\b',
]

_AUTOMATION_PATTERNS = [
    r'\b(run|execute|launch|start|stop|restart|automate|schedule|batch|'
    r'pipeline|workflow|trigger|deploy|ci|cd|cron|job)\b',
    r'\b(script|bash|shell|command|terminal|subprocess|process|daemon|'
    r'background|foreground)\b',
    r'\b(install|uninstall|upgrade|package|pip|npm|apt|brew|conda)\b',
]

_OPS_PATTERNS = [
    r'\b(deploy|provision|configure|monitor|log|alert|scale|replicate|'
    r'backup|restore|rollback|cluster|container|docker|kubernetes|k8s)\b',
    r'\b(server|service|host|port|firewall|ssl|certificate|nginx|apache|'
    r'reverse.?proxy|load.?balance)\b',
    r'\b(cpu|memory|disk|network|bandwidth|latency|throughput|metric)\b',
]

_DATA_PATTERNS = [
    r'\b(database|db|sql|query|select|insert|update|delete|table|schema|'
    r'migration|index|join|aggregate)\b',
    r'\b(csv|json|xml|yaml|excel|dataframe|pandas|numpy|plot|chart|graph|'
    r'visualization|analytics|etl|pipeline)\b',
    r'\b(parse|transform|clean|normalize|filter|sort|group|pivot|merge)\b',
]

_HIGH_RISK_PATTERNS = [
    r'\b(delete|remove|drop|destroy|purge|wipe|erase|truncate|format)\b',
    r'\b(production|prod|live|deploy to\b|push to\b|merge to main|main branch)\b',
    r'\b(root|sudo|admin|superuser|privilege|permission|chmod|chown)\b',
    r'\b(secret|password|credential|token|api.?key|private.?key)\b',
    r'\b(database|db).*(delete|drop|truncate|wipe)\b',
    r'\brm\s+-rf\b|\bformat\s+/\b',
]

_MEDIUM_RISK_PATTERNS = [
    r'\b(write|overwrite|modify|edit|update|patch|change)\b.*\b(file|config|setting)\b',
    r'\b(install|uninstall|upgrade|downgrade)\b',
    r'\b(git\s+push|git\s+commit|git\s+merge|git\s+reset)\b',
    r'\b(send|email|notify|alert|message|webhook|post\s+to)\b',
]

_COMPLEXITY_COMPOUND_SIGNALS = [
    r'\band\s+then\b|\bafter\s+that\b|\bfollowed\s+by\b|\bnext\b',
    r'\bfix.*\btest\b|\bimplement.*\bverif\b|\bwrite.*\brun\b',
    r'\bstep\s+\d+\b|\bphase\s+\d+\b|\bfirst.*then.*finally\b',
    r'\brefactor.*\btest\b|\bdeploy.*\bcheck\b|\bupdate.*\bconfirm\b',
    r'\bbuild.*\bpackage.*\bdeploy\b|\btask\s+1\b.*\btask\s+2\b',
]

_COMPLEXITY_COMPLEX_SIGNALS = [
    r'\bacross\s+(multiple|all|several|many)\b',
    r'\bthe\s+entire\s+(codebase|repo|project)\b',
    r'\brefactor\b|\barchitect\b|\bmigrate\b|\bport\b',
    r'\bintegrat\b|\bconnect\b.*\bto\b.*\bservice\b',
]

_EXTERNAL_KNOWLEDGE_SIGNALS = [
    r'\b(latest|current|recent|today|now|up.?to.?date|2024|2025|2026)\b',
    r'\b(documentation|docs\.)\b.*\b(official|external|online)\b',
    r'\b(how\s+to|tutorial|example)\b.*\b(library|framework|api)\b',
    r'\b(check\s+if|verify\s+that|confirm)\b.*\b(available|exists|works)\b',
    r'\b(search|look\s+up|find)\b.*\b(online|web|internet|stackoverflow)\b',
]

_TECH_STACK_PATTERNS = {
    "python":     r'\bpython|\.py\b|pytest|pip\b|flask|django|fastapi',
    "javascript": r'\bjavascript|\.js\b|\.jsx\b|node\.?js|npm\b|react|vue|angular',
    "typescript": r'\btypescript|\.ts\b|\.tsx\b|tsc\b',
    "rust":       r'\brust\b|\.rs\b|cargo\b',
    "go":         r'\bgolang|\bgo\b.*\bmod\b|\.go\b',
    "java":       r'\bjava\b|\.java\b|maven|gradle',
    "docker":     r'\bdocker\b|dockerfile|container\b',
    "git":        r'\bgit\b|commit|branch|merge|pull request',
    "sql":        r'\bsql\b|postgres|mysql|sqlite|database',
    "bash":       r'\bbash\b|\.sh\b|shell script|terminal',
}


# ─────────────────────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────────────────────

class TaskIntakeProcessor:
    """
    Single-responsibility processor: task string in, TaskIntake out.
    No LLM calls — pure regex + heuristic. Fast and deterministic.
    """

    def process(self, raw_task: str) -> TaskIntake:
        task = self._normalize(raw_task)
        tl = task.lower()

        intake = TaskIntake(
            raw_task=raw_task,
            task=task,
        )

        intake.task_type   = self._classify_type(tl)
        intake.risk        = self._assess_risk(tl)
        intake.complexity  = self._assess_complexity(tl)
        intake.action_verbs = self._extract_action_verbs(tl)
        intake.file_refs   = self._extract_file_refs(task)
        intake.tech_stack  = self._extract_tech_stack(tl)
        intake.entities    = self._extract_entities(task)
        intake.scope       = self._assess_scope(tl)

        # Planning flags
        intake.needs_external_knowledge = self._needs_external(tl)
        intake.needs_bash  = self._needs_bash(tl, intake.task_type)
        intake.needs_file_write = self._needs_file_write(tl, intake.task_type)
        intake.is_multi_step = intake.complexity in (
            Complexity.COMPLEX, Complexity.COMPOUND)

        # Skill query: compact form of the task for embedding/TF-IDF lookup
        intake.skill_query = self._build_skill_query(intake)

        return intake

    # ── Normalize ────────────────────────────────────────────────────────────

    def _normalize(self, task: str) -> str:
        """Strip extra whitespace, trailing punctuation; keep case."""
        t = re.sub(r'\s+', ' ', task).strip()
        t = re.sub(r'[.!?]+$', '', t).strip()
        return t

    # ── Type classification ───────────────────────────────────────────────────

    def _classify_type(self, tl: str) -> TaskType:
        scores = {
            TaskType.CODING:     self._score(tl, _CODING_PATTERNS),
            TaskType.RESEARCH:   self._score(tl, _RESEARCH_PATTERNS),
            TaskType.AUTOMATION: self._score(tl, _AUTOMATION_PATTERNS),
            TaskType.OPS:        self._score(tl, _OPS_PATTERNS),
            TaskType.DATA:       self._score(tl, _DATA_PATTERNS),
        }
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Mixed if top two scores are close
        if len(top) >= 2 and top[0][1] > 0 and top[1][1] > top[0][1] * 0.7:
            return TaskType.MIXED
        if top[0][1] == 0:
            return TaskType.MIXED
        return top[0][0]

    def _score(self, text: str, patterns: list[str]) -> float:
        """Count pattern matches and return normalized score."""
        hits = sum(1 for p in patterns if re.search(p, text, re.I))
        return hits / len(patterns) if patterns else 0.0

    # ── Risk assessment ───────────────────────────────────────────────────────

    def _assess_risk(self, tl: str) -> RiskLevel:
        if any(re.search(p, tl, re.I) for p in _HIGH_RISK_PATTERNS):
            return RiskLevel.HIGH
        if any(re.search(p, tl, re.I) for p in _MEDIUM_RISK_PATTERNS):
            return RiskLevel.MEDIUM
        # Read-only signals
        if re.search(
            r'^(list|find|show|read|search|grep|cat|explain|describe|summarize|'
            r'what|how|why|count|display)\b', tl
        ):
            return RiskLevel.LOW
        return RiskLevel.MEDIUM

    # ── Complexity assessment ─────────────────────────────────────────────────

    def _assess_complexity(self, tl: str) -> Complexity:
        # Compound: multiple distinct phases / agents needed
        if any(re.search(p, tl, re.I) for p in _COMPLEXITY_COMPOUND_SIGNALS):
            return Complexity.COMPOUND
        # Complex: multi-file, broad scope
        if any(re.search(p, tl, re.I) for p in _COMPLEXITY_COMPLEX_SIGNALS):
            return Complexity.COMPLEX
        # Word count heuristic
        word_count = len(tl.split())
        if word_count > 30:
            return Complexity.COMPLEX
        if word_count > 12:
            return Complexity.MODERATE
        # Simple: short task, single-verb pattern
        if re.match(r'^(list|find|show|read|run|check|print|echo|grep)\b', tl):
            return Complexity.SIMPLE
        return Complexity.MODERATE

    # ── Extraction ────────────────────────────────────────────────────────────

    def _extract_action_verbs(self, tl: str) -> list[str]:
        verbs = re.findall(
            r'\b(write|create|build|fix|debug|refactor|update|delete|run|'
            r'install|deploy|test|verify|find|search|read|analyze|generate|'
            r'implement|modify|add|remove|rename|move|copy|migrate|convert)\b',
            tl
        )
        return list(dict.fromkeys(verbs))[:6]  # deduplicated, max 6

    def _extract_file_refs(self, task: str) -> list[str]:
        # Match file paths and extensions
        refs = re.findall(
            r'(?:[\w./\\-]+\.(?:py|js|ts|go|rs|java|rb|cpp|c|h|json|yaml|yml|'
            r'toml|md|txt|sh|env|config|conf|ini|dockerfile))\b',
            task, re.I
        )
        # Also match quoted paths
        refs += re.findall(r'["\']([^"\']+/[^"\']+)["\']', task)
        return list(dict.fromkeys(refs))[:8]

    def _extract_tech_stack(self, tl: str) -> list[str]:
        found = [name for name, pattern in _TECH_STACK_PATTERNS.items()
                 if re.search(pattern, tl, re.I)]
        return found[:5]

    def _extract_entities(self, task: str) -> list[str]:
        """Extract named entities: function names, class names, module names."""
        # CamelCase identifiers
        camel = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', task)
        # snake_case identifiers
        snake = re.findall(r'\b[a-z]+(?:_[a-z]+){2,}\b', task)
        # Backtick-quoted
        quoted = re.findall(r'`([^`]+)`', task)
        all_entities = (quoted + camel + snake)[:10]
        return list(dict.fromkeys(all_entities))

    def _assess_scope(self, tl: str) -> str:
        if re.search(r'\b(entire|whole|all|every|codebase|repo|project|global)\b', tl):
            return "repo"
        if re.search(r'\b(system|os|machine|host|server|environment)\b', tl):
            return "system"
        if re.search(r'\b(api|service|external|remote|http|url|endpoint)\b', tl):
            return "external"
        return "local"

    # ── Planning flags ────────────────────────────────────────────────────────

    def _needs_external(self, tl: str) -> bool:
        return any(re.search(p, tl, re.I) for p in _EXTERNAL_KNOWLEDGE_SIGNALS)

    def _needs_bash(self, tl: str, task_type: TaskType) -> bool:
        if task_type in (TaskType.AUTOMATION, TaskType.OPS):
            return True
        return bool(re.search(
            r'\b(run|execute|install|test|build|compile|start|stop|'
            r'launch|deploy|check|verify)\b', tl
        ))

    def _needs_file_write(self, tl: str, task_type: TaskType) -> bool:
        if task_type == TaskType.CODING:
            return bool(re.search(
                r'\b(write|create|implement|build|add|fix|update|modify|edit|patch)\b', tl
            ))
        return bool(re.search(r'\b(write|create|generate|produce|output)\b.*\bfile\b', tl))

    # ── Skill query ───────────────────────────────────────────────────────────

    def _build_skill_query(self, intake: TaskIntake) -> str:
        """
        Build a compact keyword query for skill retrieval.
        Combines: action verbs + task type + tech stack + entities.
        """
        parts = (
            intake.action_verbs[:3]
            + [intake.task_type.value]
            + intake.tech_stack[:2]
            + intake.entities[:2]
        )
        # Also include first N words of the task
        head = " ".join(intake.task.lower().split()[:8])
        return " ".join(parts) + " " + head


# Module-level convenience
_processor = TaskIntakeProcessor()

def process_task(raw_task: str) -> TaskIntake:
    """Convenience wrapper — call this from AgentOS."""
    return _processor.process(raw_task)
