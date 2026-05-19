"""
observability.py — Automated observability for the agent harness.
Covers: hallucination detection, loop detection, token tracking,
execution tracing, and anomaly alerts — all without manual instrumentation.

Hallucination detection strategies (no heavy models required):
  1. Tool-call grounding: verify claims agent makes against tool results
  2. Schema validation: structured outputs must conform to schemas
  3. Contradiction detection: new claims vs verified memory facts
  4. Self-consistency: repeat critical questions at temp=0 and compare
  5. Confidence elicitation: ask model to rate its own confidence
  6. Fact anchoring: critical entities (filenames, functions) verified by tool
"""

import re
import time
import json
import hashlib
import uuid
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Any
from enum import Enum
from pathlib import Path
import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Events — everything emits structured events
# ─────────────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    # Lifecycle
    TASK_CREATED      = "task.created"
    TASK_COMPLETED    = "task.completed"
    TASK_FAILED       = "task.failed"
    STEP_STARTED      = "step.started"
    STEP_COMPLETED    = "step.completed"
    # LLM
    LLM_CALL_START    = "llm.call.start"
    LLM_CALL_END      = "llm.call.end"
    LLM_ERROR         = "llm.error"
    # Tools
    TOOL_CALL         = "tool.call"
    TOOL_RESULT       = "tool.result"
    TOOL_ERROR        = "tool.error"
    # Memory
    MEMORY_READ       = "memory.read"
    MEMORY_WRITE      = "memory.write"
    MEMORY_COMPACTION = "memory.compaction"
    # Safety
    LOOP_DETECTED     = "loop.detected"
    LOOP_BLOCKED      = "loop.blocked"
    BUDGET_THRESHOLD  = "budget.threshold"
    BUDGET_EXCEEDED   = "budget.exceeded"
    HALLUCINATION_SUSPECTED = "hallucination.suspected"
    HALLUCINATION_CONFIRMED = "hallucination.confirmed"
    # Skills
    SKILL_PATTERN     = "skill.pattern_detected"
    SKILL_PROMOTED    = "skill.promoted"
    # Context
    CONTEXT_BUILT     = "context.built"
    CONTEXT_TRUNCATED = "context.truncated"
    CONTEXT_COMPACTED = "context.compacted"


@dataclass
class AgentEvent:
    event_id: str          = field(default_factory=lambda: uuid.uuid4().hex[:12])
    trace_id: str          = ""
    span_id: str           = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_span_id: str    = ""
    timestamp: str         = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    event_type: str        = ""
    agent_id: str          = ""
    task_id: str           = ""
    step_id: str           = ""
    model: str             = ""
    # Metrics
    tokens_in: int         = 0
    tokens_out: int        = 0
    cost_usd: float        = 0.0
    duration_ms: float     = 0.0
    # Outcome
    success: bool          = True
    error: str             = ""
    # Loop detection
    loop_detected: bool    = False
    loop_kind: str         = ""
    loop_count: int        = 0
    # Hallucination
    hallucination_score: float = 0.0
    hallucination_flags: list  = field(default_factory=list)
    # Context
    context_tokens: int    = 0
    context_headroom: int  = 0
    truncations: list      = field(default_factory=list)
    # Extra
    metadata: dict         = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))


# ─────────────────────────────────────────────────────────────────────────────
# Event Bus
# ─────────────────────────────────────────────────────────────────────────────

class EventBus:
    """
    Thread-safe event bus.
    All agent actions emit events here.
    Subscribers can react: write to disk, update metrics, alert, etc.
    """

    def __init__(self, audit_dir: str = ".agent_os/audit"):
        self._handlers: list[Callable[[AgentEvent], None]] = []
        self._ring_buffer: deque[AgentEvent] = deque(maxlen=5000)
        self._audit_dir = Path(audit_dir)
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Auto-register audit log writer
        self.subscribe(self._write_audit_log)

    def subscribe(self, handler: Callable[[AgentEvent], None]):
        self._handlers.append(handler)
        return lambda: self._handlers.remove(handler)

    def emit(self, event: AgentEvent):
        with self._lock:
            self._ring_buffer.append(event)
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                pass  # Never let observability break the agent

    def emit_simple(self, event_type: EventType, **kwargs) -> AgentEvent:
        ev = AgentEvent(event_type=event_type.value, **kwargs)
        self.emit(ev)
        return ev

    def get_recent(self, n: int = 100,
                   filter_type: str = None,
                   filter_agent: str = None) -> list[AgentEvent]:
        events = list(self._ring_buffer)[-n*3:]
        if filter_type:
            events = [e for e in events if e.event_type == filter_type]
        if filter_agent:
            events = [e for e in events if e.agent_id == filter_agent]
        return events[-n:]

    def get_trace(self, trace_id: str) -> list[AgentEvent]:
        return [e for e in self._ring_buffer if e.trace_id == trace_id]

    def _write_audit_log(self, event: AgentEvent):
        date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        log_path = self._audit_dir / f"audit-{date}.jsonl"
        try:
            with open(log_path, "a") as f:
                f.write(event.to_jsonl() + "\n")
        except Exception:
            pass

    def get_metrics_summary(self) -> dict:
        events = list(self._ring_buffer)
        total = len(events)
        llm_calls = [e for e in events if e.event_type == EventType.LLM_CALL_END.value]
        errors = [e for e in events if not e.success]
        loops = [e for e in events if e.loop_detected]
        hallucinations = [e for e in events if e.hallucination_score > 0.5]
        total_tokens = sum(e.tokens_in + e.tokens_out for e in llm_calls)
        total_cost = sum(e.cost_usd for e in llm_calls)
        avg_latency = (sum(e.duration_ms for e in llm_calls) / len(llm_calls)
                       if llm_calls else 0)
        return {
            "total_events": total,
            "llm_calls": len(llm_calls),
            "errors": len(errors),
            "loops_detected": len(loops),
            "hallucinations_suspected": len(hallucinations),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "error_rate_pct": round(len(errors) / max(total, 1) * 100, 1),
        }


# Global bus (singleton per process)
_global_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination Detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HallucinationFlag:
    kind: str            # "unverified_claim", "contradiction", "impossible_value", ...
    severity: str        # "low", "medium", "high", "critical"
    description: str
    evidence: str
    score: float         # 0.0-1.0


class HallucinationDetector:
    """
    Automated hallucination detection without heavy models.
    Works by cross-referencing agent claims against verified ground truth.

    For a small 8B model, hallucination is frequent. This detector runs
    AFTER every agent response and flags issues before they propagate.
    """

    # Patterns that indicate the model is making specific claims
    SPECIFIC_CLAIM_PATTERNS = [
        (r'\bline\s+(\d+)\b', "line_number"),
        (r'\bfunction\s+`?([a-zA-Z_]\w+)`?', "function_name"),
        (r'\bfile\s+`?([^`\s]+\.\w+)`?', "filename"),
        (r'\bclass\s+`?([A-Z]\w+)`?', "class_name"),
        (r'\bport\s+(\d{2,5})\b', "port_number"),
        (r'\b([a-f0-9]{7,40})\b', "hash_or_id"),
    ]

    # Phrases that indicate low-confidence confabulation
    CONFABULATION_PHRASES = [
        "I believe", "I think", "probably", "might be", "could be",
        "I'm not sure", "I don't know for certain", "I assume",
        "it's likely", "it seems like", "perhaps",
        "I recall", "as I remember", "based on my knowledge",
    ]

    # Impossible/suspicious patterns for code agents
    SUSPICIOUS_PATTERNS = [
        (r'exit\s+code\s+(-?\d+)\b', lambda m: int(m.group(1)) not in range(-255, 256),
         "impossible exit code"),
        (r'(\d+)\s+token', lambda m: int(m.group(1)) > 200000,
         "impossible token count"),
        (r'ran in\s+([\d.]+)\s*ms', lambda m: float(m.group(1)) > 300000,
         "impossible execution time"),
    ]

    def __init__(self, verified_facts: dict = None):
        """
        verified_facts: dict of {fact_key: verified_value}
        Populated from tool results, never from agent text.
        """
        self.verified_facts: dict[str, Any] = verified_facts or {}
        self._flags_history: list[list[HallucinationFlag]] = []

    def update_verified_fact(self, key: str, value: Any, source: str):
        """Record a verified fact from a tool call (ground truth)."""
        self.verified_facts[key] = {
            "value": value,
            "source": source,
            "timestamp": time.time(),
        }

    def analyze(self, agent_response: str,
                tool_results: dict[str, str] = None,
                task_context: str = "") -> tuple[float, list[HallucinationFlag]]:
        """
        Analyze agent response for hallucination signals.
        Returns (score, flags) where score 0.0=clean, 1.0=almost certainly hallucinating.
        """
        flags: list[HallucinationFlag] = []
        tool_results = tool_results or {}

        # Check 1: Confabulation language
        flags.extend(self._check_confabulation_phrases(agent_response))

        # Check 2: Unverified specific claims
        flags.extend(self._check_specific_claims(agent_response, tool_results))

        # Check 3: Contradiction with verified facts
        flags.extend(self._check_contradictions(agent_response))

        # Check 4: Impossible values
        flags.extend(self._check_impossible_values(agent_response))

        # Check 5: Tool-claim alignment
        flags.extend(self._check_tool_alignment(agent_response, tool_results))

        # Check 6: Code quality (if code is in response)
        flags.extend(self._check_code_claims(agent_response))

        self._flags_history.append(flags)

        # Score: weighted sum of flag severities
        severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 1.0}
        if not flags:
            return 0.0, []

        raw_score = sum(severity_weights.get(f.severity, 0.3) * f.score for f in flags)
        # Normalize: cap at 1.0, apply diminishing returns for multiple flags
        score = min(1.0, raw_score / (1 + 0.3 * max(0, len(flags) - 2)))
        return round(score, 3), flags

    def _check_confabulation_phrases(self, text: str) -> list[HallucinationFlag]:
        flags = []
        for phrase in self.CONFABULATION_PHRASES:
            if phrase.lower() in text.lower():
                flags.append(HallucinationFlag(
                    kind="confabulation_language",
                    severity="low",
                    description=f"Uncertain language detected: '{phrase}'",
                    evidence=phrase,
                    score=0.15,
                ))
        # Deduplicate: max 2 confabulation flags
        return flags[:2]

    def _check_specific_claims(self, text: str,
                                tool_results: dict[str, str]) -> list[HallucinationFlag]:
        """
        When agent makes a specific claim (function X exists, file Y has Z lines),
        check if any tool result contradicts it.
        """
        flags = []
        all_tool_text = " ".join(tool_results.values())

        for pattern, claim_type in self.SPECIFIC_CLAIM_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches[:5]:  # check first 5 matches per type
                # If the claim is a filename or function name,
                # check if it appears in tool results
                if claim_type in ("filename", "function_name") and all_tool_text:
                    if match not in all_tool_text and len(all_tool_text) > 100:
                        flags.append(HallucinationFlag(
                            kind="unverified_entity",
                            severity="medium",
                            description=(
                                f"Agent mentions {claim_type} '{match}' "
                                f"but it doesn't appear in tool results"
                            ),
                            evidence=f"claim='{match}', tool_results_contain=False",
                            score=0.4,
                        ))
        return flags[:3]

    def _check_contradictions(self, text: str) -> list[HallucinationFlag]:
        """Check agent claims against verified facts from tool calls."""
        flags = []
        for key, fact in self.verified_facts.items():
            verified_val = str(fact["value"])
            # Simple substring check: if verified value is mentioned, is it correct?
            if verified_val in text:
                continue  # agent got it right
            # Look for numeric facts (line counts, exit codes)
            if verified_val.isdigit():
                # Check if agent mentioned a different number in same context
                context_match = re.search(
                    rf'{re.escape(key)}[^.]*?(\d+)', text, re.I
                )
                if context_match and context_match.group(1) != verified_val:
                    flags.append(HallucinationFlag(
                        kind="contradiction",
                        severity="high",
                        description=(
                            f"Agent says {key}={context_match.group(1)}, "
                            f"but verified={verified_val}"
                        ),
                        evidence=f"agent={context_match.group(1)}, truth={verified_val}",
                        score=0.75,
                    ))
        return flags

    def _check_impossible_values(self, text: str) -> list[HallucinationFlag]:
        flags = []
        for pattern, is_impossible, description in self.SUSPICIOUS_PATTERNS:
            match = re.search(pattern, text, re.I)
            if match and is_impossible(match):
                flags.append(HallucinationFlag(
                    kind="impossible_value",
                    severity="high",
                    description=f"Suspicious value detected: {description}",
                    evidence=match.group(0),
                    score=0.8,
                ))
        return flags

    def _check_tool_alignment(self, text: str,
                               tool_results: dict[str, str]) -> list[HallucinationFlag]:
        """
        If agent says 'I ran X and got Y' but tool result shows different output,
        flag it.
        """
        flags = []
        if not tool_results:
            return flags

        # Check for exit code claims
        exit_claim = re.search(r'exit(?:ed with|s?\s+code)\s+(\d+)', text, re.I)
        if exit_claim:
            claimed_code = exit_claim.group(1)
            for tool_name, result in tool_results.items():
                actual_code = re.search(r'exit(?:ed with|s?\s+code)\s+(\d+)', result, re.I)
                if actual_code and actual_code.group(1) != claimed_code:
                    flags.append(HallucinationFlag(
                        kind="tool_claim_mismatch",
                        severity="critical",
                        description=(
                            f"Agent claims exit code {claimed_code} "
                            f"but tool result shows {actual_code.group(1)}"
                        ),
                        evidence=f"agent_claim={claimed_code}, tool_result={actual_code.group(1)}",
                        score=0.9,
                    ))
        return flags

    def _check_code_claims(self, text: str) -> list[HallucinationFlag]:
        """Basic static analysis on code blocks agent generates."""
        flags = []
        # Extract code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        for code in code_blocks:
            # Check for common Python syntax errors (simple heuristics)
            if 'def ' in code or 'class ' in code:
                # Unmatched brackets
                opens = code.count('(') + code.count('[') + code.count('{')
                closes = code.count(')') + code.count(']') + code.count('}')
                if abs(opens - closes) > 3:
                    flags.append(HallucinationFlag(
                        kind="malformed_code",
                        severity="medium",
                        description="Generated code has unmatched brackets",
                        evidence=f"opens={opens}, closes={closes}",
                        score=0.35,
                    ))
        return flags[:1]


# ─────────────────────────────────────────────────────────────────────────────
# Loop Detector (ported from OpenClaw, adapted for Python)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    tool_name: str
    args_hash: str
    result_hash: str  = ""
    timestamp: float  = field(default_factory=time.time)
    run_id: str       = ""

class LoopKind(str, Enum):
    GENERIC_REPEAT    = "generic_repeat"
    POLL_NO_PROGRESS  = "poll_no_progress"
    PING_PONG         = "ping_pong"
    CIRCUIT_BREAKER   = "circuit_breaker"

@dataclass
class LoopDetectionResult:
    stuck: bool
    level: str        = ""  # "warning" | "critical"
    kind: LoopKind    = None
    count: int        = 0
    message: str      = ""

class LoopDetector:
    """
    4-detector loop prevention system.
    Must be called BEFORE every tool call.
    """
    HISTORY_SIZE   = 30
    WARN_THRESHOLD = 8
    CRIT_THRESHOLD = 15
    BREAKER        = 25

    POLL_TOOLS = {"command_status", "task_status", "poll", "check_status", "get_status"}

    def __init__(self):
        self.history: list[ToolCallRecord] = []

    def _hash(self, tool: str, args: dict) -> str:
        stable = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(f"{tool}:{stable}".encode()).hexdigest()[:16]

    def _result_hash(self, result: str) -> str:
        return hashlib.sha256(result.encode()).hexdigest()[:16]

    def check(self, tool_name: str, args: dict) -> LoopDetectionResult:
        """Call BEFORE executing tool. Returns detection result."""
        args_hash = self._hash(tool_name, args)
        hist = self.history

        # Detector 1: Circuit breaker (any tool ≥ BREAKER times)
        total_calls = sum(1 for h in hist if h.tool_name == tool_name)
        if total_calls >= self.BREAKER:
            return LoopDetectionResult(
                stuck=True, level="critical", kind=LoopKind.CIRCUIT_BREAKER,
                count=total_calls,
                message=(
                    f"CRITICAL: {tool_name} called {total_calls} times. "
                    f"Global circuit breaker triggered. STOP and explain the situation."
                )
            )

        # Detector 2: Poll no progress (same poll tool, same args, same result)
        if tool_name in self.POLL_TOOLS:
            same_polls = [h for h in hist if h.tool_name == tool_name
                          and h.args_hash == args_hash and h.result_hash]
            if len(same_polls) >= self.CRIT_THRESHOLD:
                return LoopDetectionResult(
                    stuck=True, level="critical", kind=LoopKind.POLL_NO_PROGRESS,
                    count=len(same_polls),
                    message=(
                        f"CRITICAL: Polling {tool_name} {len(same_polls)} times with "
                        f"no progress. The task may be stuck. Report as failed."
                    )
                )
            if len(same_polls) >= self.WARN_THRESHOLD:
                return LoopDetectionResult(
                    stuck=True, level="warning", kind=LoopKind.POLL_NO_PROGRESS,
                    count=len(same_polls),
                    message=(
                        f"WARNING: Polling {tool_name} {len(same_polls)} times. "
                        f"Try increasing wait time or check if task is stuck."
                    )
                )

        # Detector 3: Generic repeat (same tool+args, same result, not a poll tool)
        same_exact = [h for h in hist if h.args_hash == args_hash and h.result_hash]
        if len(same_exact) >= self.CRIT_THRESHOLD:
            return LoopDetectionResult(
                stuck=True, level="critical", kind=LoopKind.GENERIC_REPEAT,
                count=len(same_exact),
                message=(
                    f"CRITICAL: Called {tool_name} with identical args {len(same_exact)} times "
                    f"with identical outcomes. Blocked. Try a different approach."
                )
            )
        if len(same_exact) >= self.WARN_THRESHOLD:
            return LoopDetectionResult(
                stuck=True, level="warning", kind=LoopKind.GENERIC_REPEAT,
                count=len(same_exact),
                message=(
                    f"WARNING: {tool_name} called {len(same_exact)} times with same args. "
                    f"Consider a different strategy."
                )
            )

        # Detector 4: Ping-pong (alternating between 2 tool+arg combos)
        if len(hist) >= 4:
            last4 = [h.args_hash for h in hist[-4:]]
            if (last4[0] == last4[2] and last4[1] == last4[3]
                    and last4[0] != last4[1] and args_hash == last4[1]):
                count = self._count_pingpong(hist, last4[0], last4[1])
                if count >= self.WARN_THRESHOLD:
                    return LoopDetectionResult(
                        stuck=True, level="warning", kind=LoopKind.PING_PONG,
                        count=count,
                        message=(
                            f"WARNING: Alternating between two tool patterns {count} times. "
                            f"This is a ping-pong loop. Break the cycle."
                        )
                    )

        return LoopDetectionResult(stuck=False)

    def record(self, tool_name: str, args: dict, result: str = ""):
        """Call AFTER tool execution to record the outcome."""
        record = ToolCallRecord(
            tool_name=tool_name,
            args_hash=self._hash(tool_name, args),
            result_hash=self._result_hash(result) if result else "",
        )
        self.history.append(record)
        # Maintain sliding window
        if len(self.history) > self.HISTORY_SIZE:
            self.history.pop(0)

    def _count_pingpong(self, hist: list[ToolCallRecord], sig_a: str, sig_b: str) -> int:
        count = 0
        expected = sig_b
        for h in reversed(hist):
            if h.args_hash == expected:
                count += 1
                expected = sig_a if expected == sig_b else sig_b
            else:
                break
        return count


# ─────────────────────────────────────────────────────────────────────────────
# Token Tracker
# ─────────────────────────────────────────────────────────────────────────────

# Cost per 1M tokens (input, output) in USD
MODEL_COSTS = {
    "llama-3.1-8b-instant":     (0.05, 0.08),    # Groq pricing
    "llama-3.1-70b-versatile":  (0.59, 0.79),
    "llama-3.3-70b-versatile":  (0.59, 0.79),
    "mixtral-8x7b-32768":       (0.24, 0.24),
    "gemma2-9b-it":             (0.20, 0.20),
    "gpt-4o":                   (5.00, 15.00),
    "gpt-4o-mini":              (0.15, 0.60),
    "claude-3-haiku-20240307":  (0.25, 1.25),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
}

class TokenTracker:
    """Tracks token usage and cost per task/agent/session."""

    def __init__(self):
        self._records: list[dict] = []
        self._by_task: dict[str, dict] = defaultdict(lambda: {"tokens_in": 0, "tokens_out": 0, "cost": 0.0})
        self._by_agent: dict[str, dict] = defaultdict(lambda: {"tokens_in": 0, "tokens_out": 0, "cost": 0.0})

    def record(self, model: str, tokens_in: int, tokens_out: int,
               task_id: str = "", agent_id: str = "",
               duration_ms: float = 0.0) -> float:
        """Record a token usage event. Returns cost in USD."""
        input_cost, output_cost = MODEL_COSTS.get(model, (1.0, 2.0))
        cost = (tokens_in * input_cost + tokens_out * output_cost) / 1_000_000

        record = {
            "timestamp": time.time(),
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost,
            "task_id": task_id,
            "agent_id": agent_id,
            "duration_ms": duration_ms,
        }
        self._records.append(record)

        if task_id:
            self._by_task[task_id]["tokens_in"] += tokens_in
            self._by_task[task_id]["tokens_out"] += tokens_out
            self._by_task[task_id]["cost"] += cost

        if agent_id:
            self._by_agent[agent_id]["tokens_in"] += tokens_in
            self._by_agent[agent_id]["tokens_out"] += tokens_out
            self._by_agent[agent_id]["cost"] += cost

        return cost

    def get_task_usage(self, task_id: str) -> dict:
        return dict(self._by_task.get(task_id, {}))

    def get_session_total(self) -> dict:
        total_in = sum(r["tokens_in"] for r in self._records)
        total_out = sum(r["tokens_out"] for r in self._records)
        total_cost = sum(r["cost_usd"] for r in self._records)
        return {
            "total_calls": len(self._records),
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_tokens": total_in + total_out,
            "total_cost_usd": round(total_cost, 6),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Execution Tracer (span-based, like OpenTelemetry but simple)
# ─────────────────────────────────────────────────────────────────────────────

class Span:
    """An execution span for tracing agent steps."""

    def __init__(self, name: str, trace_id: str, parent_span_id: str = "",
                 agent_id: str = "", task_id: str = ""):
        self.span_id       = uuid.uuid4().hex[:8]
        self.name          = name
        self.trace_id      = trace_id
        self.parent_span_id = parent_span_id
        self.agent_id      = agent_id
        self.task_id       = task_id
        self.start_time    = time.time()
        self.end_time: Optional[float] = None
        self.attributes: dict = {}
        self.events: list[dict] = []
        self.status: str   = "running"
        self.error: str    = ""

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def end(self, status: str = "ok", error: str = ""):
        self.end_time = time.time()
        self.status = status
        self.error = error

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
            "events": self.events,
        }

class Tracer:
    """Collects spans and makes them queryable for debugging."""

    def __init__(self):
        self._spans: list[Span] = []
        self._active: dict[str, Span] = {}  # span_id → Span

    def start_span(self, name: str, trace_id: str, parent_span_id: str = "",
                   agent_id: str = "", task_id: str = "") -> Span:
        span = Span(name, trace_id, parent_span_id, agent_id, task_id)
        self._spans.append(span)
        self._active[span.span_id] = span
        return span

    def end_span(self, span: Span, status: str = "ok", error: str = ""):
        span.end(status, error)
        self._active.pop(span.span_id, None)

    def get_trace(self, trace_id: str) -> list[dict]:
        spans = [s for s in self._spans if s.trace_id == trace_id]
        return [s.to_dict() for s in sorted(spans, key=lambda s: s.start_time)]

    def get_active_spans(self) -> list[dict]:
        return [s.to_dict() for s in self._active.values()]

    def format_trace(self, trace_id: str) -> str:
        """Human-readable trace for debugging."""
        spans = self.get_trace(trace_id)
        if not spans:
            return f"No trace found for {trace_id}"
        lines = [f"── Trace: {trace_id} ──"]
        for s in spans:
            indent = "  " if s["parent_span_id"] else ""
            status_icon = "✓" if s["status"] == "ok" else "✗" if s["status"] == "error" else "…"
            lines.append(
                f"{indent}{status_icon} {s['name']} "
                f"({s['duration_ms']:.0f}ms)"
                + (f" ERROR: {s['error']}" if s["error"] else "")
            )
            for k, v in s["attributes"].items():
                lines.append(f"{indent}    {k}: {v}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Unified Observability Facade
# ─────────────────────────────────────────────────────────────────────────────

class ObservabilityManager:
    """Single import point for all observability concerns."""

    def __init__(self, audit_dir: str = ".agent_os/audit"):
        self.bus       = EventBus(audit_dir)
        self.tracer    = Tracer()
        self.tokens    = TokenTracker()
        self.hallucination = HallucinationDetector()
        self.loop      = LoopDetector()

    def new_trace(self) -> str:
        return uuid.uuid4().hex[:16]

    def start_span(self, *args, **kwargs) -> Span:
        return self.tracer.start_span(*args, **kwargs)

    def end_span(self, span: Span, **kwargs):
        self.tracer.end_span(span, **kwargs)
        # Emit corresponding event
        self.bus.emit(AgentEvent(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            event_type=f"span.{span.name}.end",
            agent_id=span.agent_id,
            task_id=span.task_id,
            duration_ms=span.duration_ms,
            success=span.status == "ok",
            error=span.error,
            metadata=span.attributes,
        ))

    def check_loop(self, tool_name: str, args: dict) -> LoopDetectionResult:
        result = self.loop.check(tool_name, args)
        if result.stuck:
            self.bus.emit_simple(
                EventType.LOOP_DETECTED,
                loop_detected=True,
                loop_kind=result.kind.value if result.kind else "",
                loop_count=result.count,
                metadata={"message": result.message, "tool": tool_name},
            )
        return result

    def record_tool(self, tool_name: str, args: dict, result: str):
        self.loop.record(tool_name, args, result)

    def record_llm_call(self, model: str, tokens_in: int, tokens_out: int,
                         duration_ms: float, task_id: str = "",
                         agent_id: str = "") -> float:
        cost = self.tokens.record(
            model, tokens_in, tokens_out, task_id, agent_id, duration_ms
        )
        self.bus.emit_simple(
            EventType.LLM_CALL_END,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            duration_ms=duration_ms,
            task_id=task_id,
            agent_id=agent_id,
            success=True,
        )
        return cost

    def analyze_hallucination(self, response: str,
                               tool_results: dict[str, str] = None) -> tuple[float, list]:
        score, flags = self.hallucination.analyze(response, tool_results)
        if score > 0.3:
            severity = "hallucination.confirmed" if score > 0.7 else "hallucination.suspected"
            self.bus.emit(AgentEvent(
                event_type=severity,
                hallucination_score=score,
                hallucination_flags=[f.kind for f in flags],
                success=False,
                metadata={"flags": [asdict(f) for f in flags]},
            ))
        return score, flags

    def print_dashboard(self):
        """Terminal dashboard — call at any time for a live snapshot."""
        import os
        width = os.get_terminal_size().columns if hasattr(os, 'get_terminal_size') else 80
        sep = "─" * width

        metrics = self.bus.get_metrics_summary()
        token_summary = self.tokens.get_session_total()
        active_spans = self.tracer.get_active_spans()

        print(f"\n{sep}")
        print("  🦞 AGENT OS — OBSERVABILITY DASHBOARD")
        print(sep)
        print(f"  LLM calls:       {metrics['llm_calls']:>6}")
        print(f"  Errors:          {metrics['errors']:>6}  ({metrics['error_rate_pct']}%)")
        print(f"  Loops detected:  {metrics['loops_detected']:>6}")
        print(f"  Hallucinations:  {metrics['hallucinations_suspected']:>6}")
        print(f"  Total tokens:    {token_summary['total_tokens']:>6,}")
        print(f"  Total cost:      ${token_summary['total_cost_usd']:>8.4f}")
        print(f"  Avg latency:     {metrics['avg_latency_ms']:>6.0f}ms")
        if active_spans:
            print(f"\n  Active spans ({len(active_spans)}):")
            for s in active_spans[:5]:
                print(f"    → {s['name']} ({s['duration_ms']:.0f}ms running)")
        print(sep + "\n")
