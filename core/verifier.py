"""
verifier.py — Output verification, failure detection, and retry strategy.

Every task output passes through here before being accepted.
The verifier detects:
  - Tool call failures (non-zero exit, error strings)
  - Incomplete execution (task was cut short)
  - Bad fallback behavior (model gave up without trying)
  - Contradictions between stated outcome and actual evidence
  - Missing required outputs (files not written, commands not run)

When verification fails, the verifier recommends a specific retry strategy
— not a random alternate command, but a targeted re-attempt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

class VerifyStatus(str, Enum):
    PASS    = "pass"
    WARN    = "warn"       # passed but with caveats
    FAIL    = "fail"
    PARTIAL = "partial"    # some criteria met, not all


@dataclass
class VerifyFlag:
    kind:        str    # "tool_error", "incomplete", "bad_fallback", etc.
    severity:    str    # "low", "medium", "high", "critical"
    description: str
    evidence:    str


@dataclass
class VerificationResult:
    status:          VerifyStatus
    flags:           list[VerifyFlag]
    criteria_met:    list[str]
    criteria_missed: list[str]
    retry_strategy:  str    # specific advice for retry
    retry_viable:    bool   # is it worth retrying?
    confidence:      float  # 0.0-1.0 that the result is correct

    @property
    def passed(self) -> bool:
        return self.status in (VerifyStatus.PASS, VerifyStatus.WARN)

    def summary(self) -> str:
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌", "partial": "🔶"}.get(
            self.status.value, "?")
        s = f"{icon} {self.status.value.upper()}"
        if self.flags:
            s += f" | {len(self.flags)} flag(s)"
        if self.criteria_missed:
            s += f" | missed: {', '.join(self.criteria_missed[:2])}"
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that indicate tool/bash failure
_FAILURE_PATTERNS = [
    (r'\bcommand not found\b',           "command_not_found",  "high"),
    (r'\bno such file or directory\b',   "file_not_found",     "high"),
    (r'\bpermission denied\b',           "permission_denied",  "high"),
    (r'\bsyntax error\b',                "syntax_error",       "high"),
    (r'\btraceback \(most recent\b',     "python_traceback",   "high"),
    (r'\berror: \b',                     "generic_error",      "medium"),
    (r'\bfailed\b.*\bwith\b.*\bexit\b', "non_zero_exit",      "high"),
    (r'\bexit code [1-9]\d*\b',          "non_zero_exit",      "high"),
    (r'\bnot installed\b',               "missing_dep",        "medium"),
    (r'\bmodule not found\b',            "missing_module",     "medium"),
    (r'\bconnection refused\b',          "connection_refused", "high"),
    (r'\btimeout\b',                     "timeout",            "medium"),
    (r'\b(assertion|assert) error\b',    "assertion_error",    "high"),
]

# Patterns that indicate the model gave up / hallucinated completion
_BAD_FALLBACK_PATTERNS = [
    r"\bi (cannot|can't|can not) (access|run|execute|use)\b",
    r"\bi don't have (access|the ability|permission)\b",
    r"\bas an ai\b.*\bi (cannot|can't)\b",
    r"\bi (would|should) (suggest|recommend) (that you|you)\b",
    r"\bplease (run|execute|check) (this|the following)\b",
    r"\bhere is (a|an|the) (script|command|code) (you can|to) run\b",
    r"\bi (am|'m) unable to (directly|actually)\b",
]

# Patterns indicating incomplete execution
_INCOMPLETE_PATTERNS = [
    r"\bstep \d+ (is|was) (skipped|not completed)\b",
    r"\b(todo|fixme|placeholder)\b",
    r"\b\.\.\.\s*$",                        # trailing ellipsis
    r"\b(rest of|remaining) (the )?(steps?|code|implementation)\b",
    r"\b(not|haven't|have not) (yet |)(run|executed|tested|verified)\b",
]

_SUCCESS_PATTERNS = [
    r"\b(all tests? pass(ed)?|test(s)? (passed|succeed))\b",
    r"\bexit code 0\b",
    r"\bsuccess(fully)?\b",
    r"\b(done|completed|finished)\b",
    r"\bno errors?\b",
    r"\bno failures?\b",
    r"\[passed\]",
    r"\[pass\]",
    r"all tests passed",
    r"\bpassed\b",
]


class Verifier:
    """
    Verifies agent outputs against success criteria and known failure patterns.
    Works on the combined agent response + tool outputs.
    """

    def verify(
        self,
        agent_response:  str,
        tool_outputs:    list[str],
        success_criteria: list[str],
        task_text:       str = "",
        steps_taken:     int = 0,
        max_steps:       int = 20,
    ) -> VerificationResult:

        combined = agent_response + "\n" + "\n".join(tool_outputs)
        combined_lower = combined.lower()
        flags: list[VerifyFlag] = []

        # ── Check 1: Tool/bash failures ───────────────────────────────────────
        for pattern, kind, severity in _FAILURE_PATTERNS:
            m = re.search(pattern, combined_lower)
            if m:
                flags.append(VerifyFlag(
                    kind=f"tool_error.{kind}",
                    severity=severity,
                    description=f"Failure pattern detected: {kind.replace('_', ' ')}",
                    evidence=combined[max(0, combined_lower.find(m.group())-30):
                                      combined_lower.find(m.group())+100],
                ))

        # ── Check 2: Bad fallback (model gave up) ─────────────────────────────
        for pattern in _BAD_FALLBACK_PATTERNS:
            if re.search(pattern, combined_lower):
                flags.append(VerifyFlag(
                    kind="bad_fallback",
                    severity="high",
                    description="Agent refused to act or delegated to user",
                    evidence=re.search(pattern, combined_lower).group(0)[:80],
                ))
                break  # one bad fallback flag is enough

        # ── Check 3: Incomplete execution ────────────────────────────────────
        for pattern in _INCOMPLETE_PATTERNS:
            if re.search(pattern, combined_lower):
                flags.append(VerifyFlag(
                    kind="incomplete",
                    severity="medium",
                    description="Response appears incomplete or placeholder",
                    evidence=re.search(pattern, combined_lower).group(0)[:80],
                ))
                break

        # ── Check 4: Step budget exceeded ────────────────────────────────────
        if max_steps > 0 and steps_taken >= max_steps:
            flags.append(VerifyFlag(
                kind="max_steps_hit",
                severity="medium",
                description=f"Task hit max steps ({max_steps}) — may be incomplete",
                evidence=f"steps_taken={steps_taken}",
            ))

        # ── Check 5: Success criteria ─────────────────────────────────────────
        criteria_met    = []
        criteria_missed = []
        for criterion in success_criteria:
            if self._criterion_met(criterion, combined_lower):
                criteria_met.append(criterion)
            else:
                criteria_missed.append(criterion)

        # ── Check 6: Presence of success signals ──────────────────────────────
        has_success_signal = any(
            re.search(p, combined_lower) for p in _SUCCESS_PATTERNS
        )

        # ── Determine overall status ──────────────────────────────────────────
        critical_flags = [f for f in flags if f.severity == "critical"]
        high_flags     = [f for f in flags if f.severity == "high"]
        bad_fallbacks  = [f for f in flags if f.kind == "bad_fallback"]

        if bad_fallbacks:
            status = VerifyStatus.FAIL
        elif critical_flags:
            status = VerifyStatus.FAIL
        elif len(high_flags) >= 2:
            status = VerifyStatus.FAIL
        elif high_flags and not has_success_signal:
            status = VerifyStatus.FAIL
        elif criteria_missed and not criteria_met:
            status = VerifyStatus.FAIL
        elif criteria_missed:
            status = VerifyStatus.PARTIAL
        elif flags:
            status = VerifyStatus.WARN
        else:
            status = VerifyStatus.PASS

        # ── Confidence estimate ───────────────────────────────────────────────
        confidence = self._estimate_confidence(
            status, flags, criteria_met, criteria_missed, has_success_signal)

        # ── Retry strategy ────────────────────────────────────────────────────
        retry_strategy = self._suggest_retry(flags, criteria_missed, task_text)
        retry_viable   = (
            status != VerifyStatus.PASS and
            len([f for f in flags if f.kind == "bad_fallback"]) == 0 and
            steps_taken < max_steps * 2
        )

        return VerificationResult(
            status=status,
            flags=flags,
            criteria_met=criteria_met,
            criteria_missed=criteria_missed,
            retry_strategy=retry_strategy,
            retry_viable=retry_viable,
            confidence=confidence,
        )

    def _criterion_met(self, criterion: str, text: str) -> bool:
        """Check if a success criterion is satisfied in the output."""
        c = criterion.lower()
        if c in text:
            return True
        if c in ("passed", "pass") and ("[passed]" in text or "[pass]" in text):
            return True
        if ("all tests pass" in c or "tests pass" in c) and (
                "[passed]" in text or "all tests passed" in text):
            return True
        words = re.findall(r"\w+", c)
        if not words:
            return False
        hits = sum(1 for w in words if len(w) > 3 and w in text)
        return hits / len(words) >= 0.6

    def _estimate_confidence(
        self,
        status:           VerifyStatus,
        flags:            list[VerifyFlag],
        criteria_met:     list[str],
        criteria_missed:  list[str],
        has_success_signal: bool,
    ) -> float:
        base = {
            VerifyStatus.PASS:    0.85,
            VerifyStatus.WARN:    0.65,
            VerifyStatus.PARTIAL: 0.45,
            VerifyStatus.FAIL:    0.15,
        }[status]

        if has_success_signal:
            base += 0.1
        if criteria_met:
            base += 0.05 * len(criteria_met)
        for f in flags:
            base -= {"low": 0.02, "medium": 0.08, "high": 0.15, "critical": 0.3}.get(
                f.severity, 0.05)

        return round(min(0.99, max(0.01, base)), 2)

    def _suggest_retry(
        self,
        flags:           list[VerifyFlag],
        criteria_missed: list[str],
        task_text:       str,
    ) -> str:
        kinds = {f.kind for f in flags}

        if "bad_fallback" in str(kinds):
            return (
                "The agent refused to act. Retry with explicit permission to use bash "
                "and a reminder that it has tool access."
            )
        if "tool_error.command_not_found" in kinds:
            return (
                "A required command was not found. Retry after checking PATH or "
                "installing the missing tool."
            )
        if "tool_error.permission_denied" in kinds:
            return (
                "Permission denied. Retry with appropriate permissions or "
                "check if the file is locked."
            )
        if "tool_error.python_traceback" in kinds:
            return (
                "Python error detected. Read the traceback, identify the root cause, "
                "then make a targeted fix — don't rewrite the whole file."
            )
        if "tool_error.syntax_error" in kinds:
            return (
                "Syntax error in generated code. Read the file back and fix only "
                "the syntax error on the specific line."
            )
        if "incomplete" in kinds:
            return (
                "Response was incomplete. Continue from where the agent left off. "
                "Don't restart from the beginning."
            )
        if "max_steps_hit" in kinds:
            return (
                "Task hit step limit. Retry with a more focused sub-task — "
                "break the original task into smaller pieces."
            )
        if criteria_missed:
            missed_str = "; ".join(criteria_missed[:2])
            return f"Retry focusing specifically on: {missed_str}"
        return "Re-examine the task requirements and try a different approach."

    def quick_check_tool_output(self, tool_name: str, output: str) -> tuple[bool, str]:
        """
        Fast check on a single tool output — called inline during execution.
        Returns (success, error_description).
        """
        if not output:
            return False, "Empty output from tool"
        out_lower = output.lower()
        for pattern, kind, severity in _FAILURE_PATTERNS:
            if severity == "high" and re.search(pattern, out_lower):
                return False, f"{kind}: {output[:200]}"
        return True, ""