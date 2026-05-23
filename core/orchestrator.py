"""
orchestrator.py — Multi-agent orchestrator for Kartavya Agent OS.

Pattern: Anthropic orchestrator-worker model (supervisor + specialist workers).

Flow:
  1. Coordinator receives complex task
  2. TaskDecomposer breaks it into subtasks and assigns roles
  3. Each subtask runs in a specialist Agent instance
  4. ResultMerger combines outputs into a final answer
  5. Human-in-the-loop gate before merging conflicting outputs

Roles:
  researcher  — information gathering, file reading, web search
  coder       — writing, editing, and refactoring code
  tester      — running tests, verifying fixes, bash execution
  reviewer    — code review, output validation, correctness checking
  general     — general-purpose fallback (same as base Agent)

Design principles:
  - Sequential execution by default (parallel is a v2 feature, deadlock-free)
  - Each sub-agent gets its own memory + context (no shared mutable state)
  - Full observability: every subtask emits events under the same trace
  - Budget tracking: orchestrator enforces per-task and per-subtask limits
  - Graceful degradation: if decomposition fails, run as single agent
"""

from __future__ import annotations

import re
import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import Agent, AgentConfig, TaskResult


# ─────────────────────────────────────────────────────────────────────────────
# Role definitions
# Each role gets a specialized system prompt prefix injected into the agent
# ─────────────────────────────────────────────────────────────────────────────

ROLE_CONFIGS = {
    "researcher": {
        "description": "information gathering, reading files, searching code",
        "allowed_tools": ["bash", "read_file", "grep", "find_files", "list_dir",
                          "web_search", "memory_read", "task_done"],
        "temperature": 0.1,
        "max_steps": 8,
        "prompt_prefix": (
            "You are a RESEARCHER agent. Your job is to gather information only — "
            "do NOT write or modify any files. Use read_file, grep, find_files to "
            "collect facts. Return all findings in task_done."
        ),
    },
    "coder": {
        "description": "writing, editing, and creating code",
        "allowed_tools": ["bash", "read_file", "write_file", "append_file",
                          "grep", "find_files", "list_dir", "task_done"],
        "temperature": 0.1,
        "max_steps": 15,
        "prompt_prefix": (
            "You are a CODER agent. Your job is to write or modify code. "
            "Always read the file before writing it. Make minimal, targeted changes. "
            "Do NOT run tests — the tester agent will do that."
        ),
    },
    "tester": {
        "description": "running tests, executing bash commands, verifying results",
        "allowed_tools": ["bash", "read_file", "grep", "find_files", "task_done"],
        "temperature": 0.0,
        "max_steps": 10,
        "prompt_prefix": (
            "You are a TESTER agent. Your job is to RUN commands and verify correctness. "
            "Run the test suite, check exit codes, and report exactly what passed or failed. "
            "Do NOT modify code — only observe and report."
        ),
    },
    "reviewer": {
        "description": "reviewing output, checking correctness, validating results",
        "allowed_tools": ["read_file", "grep", "find_files", "list_dir",
                          "memory_read", "task_done"],
        "temperature": 0.2,
        "max_steps": 6,
        "prompt_prefix": (
            "You are a REVIEWER agent. Critically examine the work done and identify "
            "any issues, inconsistencies, or gaps. Be specific — quote line numbers "
            "and file names. Return a structured verdict: PASS or NEEDS_FIX with details."
        ),
    },
    "general": {
        "description": "general-purpose task execution",
        "allowed_tools": [],       # empty = all tools
        "temperature": 0.2,
        "max_steps": 20,
        "prompt_prefix": "",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubTask:
    subtask_id:  str
    role:        str                  # researcher | coder | tester | reviewer | general
    task_text:   str                  # the task this sub-agent will receive
    depends_on:  list[str] = field(default_factory=list)  # subtask_ids this needs
    result:      Optional[TaskResult] = None
    skipped:     bool = False
    skip_reason: str  = ""

    @property
    def succeeded(self) -> bool:
        return self.result is not None and self.result.success


@dataclass
class DecompositionResult:
    subtasks:  list[SubTask]
    strategy:  str   # "sequential" | "single"
    rationale: str   # why we decomposed this way (for observability)
    is_complex: bool = True


@dataclass
class OrchestratorResult:
    task:           str
    success:        bool
    final_output:   str
    subtasks:       list[SubTask]
    total_tokens:   int
    total_cost_usd: float
    duration_ms:    float
    trace_id:       str
    strategy:       str
    errors:         list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Task Decomposer
# ─────────────────────────────────────────────────────────────────────────────

class TaskDecomposer:
    """
    Decomposes a complex task into ordered subtasks with assigned roles.

    Two strategies:
      A) LLM-based: asks the model to produce a JSON decomposition plan
      B) Heuristic fallback: pattern-match task keywords to detect multi-step needs

    The LLM call uses the cheapest/fastest model available to minimize overhead.
    """

    # Keywords that strongly suggest a multi-step task requiring role specialization
    COMPLEX_PATTERNS = [
        (r'\band\s+(then|also|after)\b',        "sequential action"),
        (r'\bfix.*\btest',                       "fix then test"),
        (r'\bimplement.*\bverif',                "implement then verify"),
        (r'\brefactor.*\btest',                  "refactor then test"),
        (r'\bwrite.*\brun',                      "write then run"),
        (r'\bfind.*\bfix',                       "find then fix"),
        (r'\bdeploy.*\bcheck',                   "deploy then check"),
        (r'\bupdate.*\bconfirm',                 "update then confirm"),
    ]

    # Simple single-step tasks — don't decompose these
    SIMPLE_PATTERNS = [
        r'^list\b', r'^find\b', r'^show\b', r'^read\b',
        r'^count\b', r'^search\b', r'^grep\b', r'^cat\b',
        r'^echo\b', r'^print\b',
    ]

    def decompose_heuristic(self, task: str) -> DecompositionResult:
        """
        Pattern-based decomposition — no LLM call required.
        Fast and predictable. Used when LLM decomposition fails or is disabled.
        """
        t = task.lower().strip()

        # Is it obviously simple?
        for pattern in self.SIMPLE_PATTERNS:
            if re.match(pattern, t):
                return DecompositionResult(
                    subtasks=[SubTask(
                        subtask_id=uuid.uuid4().hex[:8],
                        role="general", task_text=task,
                    )],
                    strategy="single",
                    rationale="Simple single-step task detected by pattern match",
                    is_complex=False,
                )

        # Detect fix+test pattern
        if re.search(r'\bfix\b', t) and re.search(r'\btests?\b', t):
            return self._fix_and_test_plan(task)

        # Detect implement+verify pattern
        if (re.search(r'\bimplement\b|\bwrite\b|\bcreate\b', t) and
                re.search(r'\btests?\b|\bverif|\bcheck\b', t)):
            return self._implement_and_verify_plan(task)

        # Detect research+code pattern
        if (re.search(r'\bunderstand\b|\banalyze\b|\breadd?\b', t) and
                re.search(r'\bfix\b|\bupdate\b|\bchanges?\b', t)):
            return self._research_and_code_plan(task)

        # Check generic complexity patterns
        for pattern, reason in self.COMPLEX_PATTERNS:
            if re.search(pattern, t, re.I):
                return self._generic_two_phase_plan(task, reason)

        # Long tasks (many words) likely need multiple steps
        if len(task.split()) > 25:
            return self._generic_two_phase_plan(task, "long/complex task")

        # Default: single agent
        return DecompositionResult(
            subtasks=[SubTask(
                subtask_id=uuid.uuid4().hex[:8],
                role="general", task_text=task,
            )],
            strategy="single",
            rationale="No complex pattern detected — running as single agent",
            is_complex=False,
        )

    def decompose_with_llm(self, task: str, provider,
                            model: str) -> Optional[DecompositionResult]:
        """
        Ask the LLM to decompose the task. Returns None if call fails.
        Uses a structured JSON prompt to get a reliable plan.
        """
        system = (
            "You are a task planner. Given a user task, decide if it needs multiple "
            "specialist agents or a single agent.\n\n"
            "Respond ONLY with valid JSON — no markdown, no preamble:\n"
            "{\n"
            "  \"is_complex\": true/false,\n"
            "  \"strategy\": \"sequential\" or \"single\",\n"
            "  \"rationale\": \"one sentence\",\n"
            "  \"subtasks\": [\n"
            "    {\"role\": \"researcher|coder|tester|reviewer|general\",\n"
            "     \"task\": \"specific instruction for this role\"}\n"
            "  ]\n"
            "}\n\n"
            "Roles:\n"
            "  researcher = gather info (read files, grep, search)\n"
            "  coder      = write/modify code\n"
            "  tester     = run tests/commands\n"
            "  reviewer   = check correctness\n"
            "  general    = anything else\n\n"
            "Keep it simple: max 3 subtasks. Single agent if the task is straightforward."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task: {task}"},
        ]
        try:
            response = provider.complete(
                messages=messages, model=model,
                max_tokens=400, temperature=0.0, stream=False,
            )
            text = response.content.strip()
            # Strip markdown fences if present
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            plan = json.loads(text)

            if not plan.get("is_complex", True) or not plan.get("subtasks"):
                return None

            subtasks = []
            for i, st in enumerate(plan["subtasks"][:4]):  # max 4
                subtasks.append(SubTask(
                    subtask_id=uuid.uuid4().hex[:8],
                    role=st.get("role", "general"),
                    task_text=st.get("task", task),
                ))

            return DecompositionResult(
                subtasks=subtasks,
                strategy=plan.get("strategy", "sequential"),
                rationale=plan.get("rationale", "LLM decomposition"),
                is_complex=True,
            )
        except Exception as e:
            print(f"   ⚠️  LLM decomposition failed ({e}), using heuristic")
            return None

    # ── Canned plans ──────────────────────────────────────────────────────────

    def _fix_and_test_plan(self, task: str) -> DecompositionResult:
        r_id = uuid.uuid4().hex[:8]
        c_id = uuid.uuid4().hex[:8]
        t_id = uuid.uuid4().hex[:8]
        return DecompositionResult(
            subtasks=[
                SubTask(r_id, "researcher",
                        f"Find the bug/issue for this task: {task}. "
                        f"Read the relevant files and identify what needs changing."),
                SubTask(c_id, "coder",
                        f"Apply the fix for: {task}. "
                        f"Make only the necessary code changes.",
                        depends_on=[r_id]),
                SubTask(t_id, "tester",
                        f"Run the tests to verify the fix is correct for: {task}. "
                        f"Report pass/fail with full output.",
                        depends_on=[c_id]),
            ],
            strategy="sequential",
            rationale="fix-then-test pattern detected",
            is_complex=True,
        )

    def _implement_and_verify_plan(self, task: str) -> DecompositionResult:
        c_id = uuid.uuid4().hex[:8]
        t_id = uuid.uuid4().hex[:8]
        return DecompositionResult(
            subtasks=[
                SubTask(c_id, "coder",
                        f"Implement the following: {task}. "
                        f"Write the code but do not run tests."),
                SubTask(t_id, "tester",
                        f"Test and verify the implementation for: {task}. "
                        f"Run all relevant tests and report results.",
                        depends_on=[c_id]),
            ],
            strategy="sequential",
            rationale="implement-then-verify pattern detected",
            is_complex=True,
        )

    def _research_and_code_plan(self, task: str) -> DecompositionResult:
        r_id = uuid.uuid4().hex[:8]
        c_id = uuid.uuid4().hex[:8]
        return DecompositionResult(
            subtasks=[
                SubTask(r_id, "researcher",
                        f"Understand the codebase and gather all information needed for: {task}. "
                        f"Read relevant files and summarize findings."),
                SubTask(c_id, "coder",
                        f"Using the research findings, make the required changes for: {task}.",
                        depends_on=[r_id]),
            ],
            strategy="sequential",
            rationale="research-then-code pattern detected",
            is_complex=True,
        )

    def _generic_two_phase_plan(self, task: str, reason: str) -> DecompositionResult:
        g1_id = uuid.uuid4().hex[:8]
        g2_id = uuid.uuid4().hex[:8]
        return DecompositionResult(
            subtasks=[
                SubTask(g1_id, "researcher",
                        f"Phase 1 — Gather information and understand the current state for: {task}"),
                SubTask(g2_id, "general",
                        f"Phase 2 — Complete the task using what was found: {task}",
                        depends_on=[g1_id]),
            ],
            strategy="sequential",
            rationale=f"Two-phase plan ({reason})",
            is_complex=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Result Merger
# ─────────────────────────────────────────────────────────────────────────────

class ResultMerger:
    """
    Combines results from multiple sub-agents into a single final output.

    Merge strategy:
      - If only one subtask: return its output directly
      - If all succeeded: combine with role-appropriate headers
      - If some failed: include partial results + error summary
      - If reviewer role present: use its verdict as the final output
      - Conflict detection: flag when outputs contradict each other
    """

    def merge(self, task: str, subtasks: list[SubTask]) -> tuple[str, bool]:
        """
        Returns (final_output, overall_success).
        """
        completed = [st for st in subtasks if st.result and not st.skipped]

        if not completed:
            return "All subtasks failed or were skipped.", False

        # Single agent path
        if len(completed) == 1:
            r = completed[0].result
            return r.output, r.success

        # Check for reviewer verdict first
        reviewer_tasks = [st for st in completed if st.role == "reviewer" and st.succeeded]
        if reviewer_tasks:
            verdict = reviewer_tasks[-1].result.output
            # Combine with previous outputs for context
            other_outputs = [
                f"[{st.role.upper()} output]\n{st.result.output[:500]}"
                for st in completed if st.role != "reviewer" and st.succeeded
            ]
            if "NEEDS_FIX" in verdict.upper() or "FAIL" in verdict.upper():
                combined = "\n\n".join(other_outputs)
                return (f"⚠️  Reviewer flagged issues:\n{verdict}\n\n"
                        f"Work done:\n{combined}"), False
            return f"✅ Review passed:\n{verdict}\n\n" + "\n\n".join(other_outputs), True

        # Tester result is authoritative for success/failure
        tester_tasks = [st for st in completed if st.role == "tester"]
        overall_success = all(st.succeeded for st in completed)
        if tester_tasks:
            last_test = tester_tasks[-1]
            overall_success = last_test.succeeded

        # Build merged output
        sections = []
        for st in completed:
            if st.result and st.result.output:
                header = f"[{st.role.upper()} — {st.task_text[:60]}{'...' if len(st.task_text) > 60 else ''}]"
                output_preview = st.result.output[:800]
                sections.append(f"{header}\n{output_preview}")

        failed = [st for st in subtasks if st.result and not st.result.success and not st.skipped]
        skipped = [st for st in subtasks if st.skipped]

        combined = "\n\n" + "─"*40 + "\n\n".join(sections)

        if failed:
            combined += f"\n\n⚠️  Failed subtasks: {', '.join(st.role for st in failed)}"
        if skipped:
            combined += f"\n⏭  Skipped: {', '.join(f'{st.role} ({st.skip_reason})' for st in skipped)}"

        return combined.strip(), overall_success

    def detect_conflicts(self, subtasks: list[SubTask]) -> list[str]:
        """
        Simple conflict detection: look for contradictory signals in outputs.
        Returns list of conflict descriptions (empty = no conflicts).
        """
        conflicts = []
        outputs = [(st.role, st.result.output) for st in subtasks
                   if st.result and st.result.output]

        # Check test pass vs test fail
        pass_signals = ["passed", "all tests pass", "success", "ok"]
        fail_signals = ["failed", "error", "assertion", "traceback"]

        has_pass = any(any(s in out.lower() for s in pass_signals) for _, out in outputs)
        has_fail = any(any(s in out.lower() for s in fail_signals) for _, out in outputs)

        if has_pass and has_fail:
            conflicts.append(
                "Conflicting test signals: some outputs indicate success while others show failure. "
                "Human review recommended before accepting results."
            )

        return conflicts


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorConfig:
    model:             str   = "llama-3.1-8b-instant"
    provider:          str   = "groq"
    use_llm_decompose: bool  = True    # True = LLM plans; False = heuristic only
    require_review:    bool  = False   # True = always adds a reviewer sub-agent
    human_gate:        bool  = False   # True = ask human before merging
    token_budget_usd:  float = None    # None = unlimited
    max_subtasks:      int   = 4
    max_steps_per_subtask: int = 15
    audit_dir:         str   = ".agent_os/audit"
    memory_dir:        str   = ".agent_os/memory"
    run_dir:           str   = ".agent_os/runs"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Coordinates multiple specialist agents to complete complex tasks.

    Usage:
        orch = Orchestrator(OrchestratorConfig(model="llama-3.1-8b-instant"))
        result = orch.coordinate("find the bug in auth.py and fix it, then run tests")
        print(result.final_output)
    """

    def __init__(self, config: OrchestratorConfig = None, api_key: str = None,
                 ollama_host: str = None):
        import os
        self.config = config or OrchestratorConfig()
        self._api_key     = api_key
        self._ollama_host = ollama_host
        self.decomposer   = TaskDecomposer()
        self.merger       = ResultMerger()

        # Bootstrap provider for decomposition calls
        from providers.providers import get_provider
        env_key_map = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY",
                       "anthropic": "ANTHROPIC_API_KEY", "ollama": ""}
        resolved_key = api_key or os.environ.get(
            env_key_map.get(self.config.provider, ""), "")
        self._provider = get_provider(
            self.config.provider,
            api_key=resolved_key or None,
            host=ollama_host,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def coordinate(self, task: str) -> OrchestratorResult:
        """
        Decompose task → execute sub-agents → merge results.
        """
        trace_id   = uuid.uuid4().hex[:16]
        start_time = time.time()
        errors: list[str] = []

        print(f"\n{'='*60}")
        print(f"🎯 ORCHESTRATOR  {trace_id}")
        print(f"   Model:   {self.config.model}  via  {self.config.provider}")
        print(f"   Task:    {task}")
        print(f"{'='*60}")

        # ── Step 1: Decompose ─────────────────────────────────────────────
        plan = None
        if self.config.use_llm_decompose:
            plan = self.decomposer.decompose_with_llm(
                task, self._provider, self.config.model)

        if plan is None:
            plan = self.decomposer.decompose_heuristic(task)

        # Cap subtask count
        if len(plan.subtasks) > self.config.max_subtasks:
            plan.subtasks = plan.subtasks[:self.config.max_subtasks]

        # Optionally append reviewer
        if self.config.require_review and plan.is_complex:
            last_id = plan.subtasks[-1].subtask_id if plan.subtasks else ""
            plan.subtasks.append(SubTask(
                subtask_id=uuid.uuid4().hex[:8],
                role="reviewer",
                task_text=(
                    f"Review all the work done on the following task and give a "
                    f"final verdict (PASS or NEEDS_FIX with specific details): {task}"
                ),
                depends_on=[last_id] if last_id else [],
            ))

        print(f"\n📋 Plan: {plan.strategy} | {len(plan.subtasks)} subtask(s)")
        print(f"   Rationale: {plan.rationale}")
        for i, st in enumerate(plan.subtasks):
            dep_str = f" (after {','.join(st.depends_on)})" if st.depends_on else ""
            print(f"   {i+1}. [{st.role}]{dep_str}: {st.task_text[:70]}{'…' if len(st.task_text)>70 else ''}")
        print()

        # ── Step 2: Execute subtasks ──────────────────────────────────────
        total_tokens = 0
        total_cost   = 0.0
        context_so_far: dict[str, str] = {}  # subtask_id → output (for chaining)

        for idx, subtask in enumerate(plan.subtasks):
            # Check dependencies
            if subtask.depends_on:
                failed_deps = [d for d in subtask.depends_on
                               if d in context_so_far and not context_so_far.get(d)]
                if failed_deps:
                    subtask.skipped = True
                    subtask.skip_reason = f"dependency {failed_deps[0]} failed"
                    print(f"\n⏭  Skipping [{subtask.role}]: {subtask.skip_reason}")
                    continue

            # Check budget
            if (self.config.token_budget_usd and
                    total_cost >= self.config.token_budget_usd * 0.9):
                subtask.skipped = True
                subtask.skip_reason = "global token budget nearly exhausted"
                print(f"\n⏭  Skipping [{subtask.role}]: {subtask.skip_reason}")
                continue

            # Build task text — inject previous context for chaining
            task_text = subtask.task_text
            if context_so_far and subtask.depends_on:
                prev_outputs = [
                    f"[Previous {plan.subtasks[i2].role} output]:\n{ctx[:600]}"
                    for i2, st2 in enumerate(plan.subtasks)
                    if st2.subtask_id in subtask.depends_on
                    and st2.subtask_id in context_so_far
                    for ctx in [context_so_far[st2.subtask_id]]
                    if ctx
                ]
                if prev_outputs:
                    task_text = "\n\n".join(prev_outputs) + f"\n\nNow: {task_text}"

            print(f"\n── Subtask {idx+1}/{len(plan.subtasks)}: [{subtask.role}] ──")

            # Create specialist agent
            agent = self._create_agent(subtask.role, subtask.subtask_id)

            # Override task-level budget if we have remaining
            if self.config.token_budget_usd:
                remaining = self.config.token_budget_usd - total_cost
                agent.config.token_budget_usd = max(0.001, remaining * 0.6)

            # Run the subtask
            try:
                result = agent.run_task(task_text, task_id=subtask.subtask_id)
                subtask.result = result
                total_tokens += result.tokens_used
                total_cost   += result.cost_usd
                context_so_far[subtask.subtask_id] = result.output if result.success else ""
                if result.errors:
                    errors.extend(result.errors[:2])

            except Exception as e:
                err = f"Subtask {subtask.subtask_id} [{subtask.role}] crashed: {e}"
                errors.append(err)
                print(f"\n   ❌ {err}")
                context_so_far[subtask.subtask_id] = ""

        # ── Step 3: Human gate (optional) ────────────────────────────────
        conflicts = self.merger.detect_conflicts(plan.subtasks)
        if conflicts:
            print(f"\n⚠️  CONFLICTS DETECTED:")
            for c in conflicts:
                print(f"   {c}")

        if self.config.human_gate and plan.is_complex:
            print(f"\n🔎 HUMAN REVIEW GATE")
            print(f"   {len(plan.subtasks)} subtasks completed. Merge results?")
            try:
                answer = input("   Approve merge? (y/n): ").strip().lower()
                if answer not in ("y", "yes"):
                    return OrchestratorResult(
                        task=task, success=False,
                        final_output="Merge rejected by human reviewer.",
                        subtasks=plan.subtasks,
                        total_tokens=total_tokens, total_cost_usd=total_cost,
                        duration_ms=(time.time()-start_time)*1000,
                        trace_id=trace_id, strategy=plan.strategy,
                        errors=["Human rejected merge"],
                    )
            except (EOFError, KeyboardInterrupt):
                pass  # Non-interactive mode — auto-approve

        # ── Step 4: Merge ─────────────────────────────────────────────────
        final_output, overall_success = self.merger.merge(task, plan.subtasks)

        duration_ms = (time.time() - start_time) * 1000

        print(f"\n{'='*60}")
        icon = "✅ ORCHESTRATED" if overall_success else "⚠️  PARTIAL"
        print(f"{icon} | {len(plan.subtasks)} subtasks | "
              f"{total_tokens:,}tk | ${total_cost:.5f} | {duration_ms:.0f}ms")
        if conflicts:
            print(f"⚠️  {len(conflicts)} conflict(s) detected — review recommended")
        print(f"{'='*60}\n")

        return OrchestratorResult(
            task=task, success=overall_success, final_output=final_output,
            subtasks=plan.subtasks, total_tokens=total_tokens,
            total_cost_usd=total_cost, duration_ms=duration_ms,
            trace_id=trace_id, strategy=plan.strategy, errors=errors,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Agent factory
    # ─────────────────────────────────────────────────────────────────────────

    def _create_agent(self, role: str, agent_id_suffix: str = "") -> Agent:
        """
        Create a specialist Agent instance for the given role.
        Each agent gets its own isolated memory.
        """
        role_cfg = ROLE_CONFIGS.get(role, ROLE_CONFIGS["general"])

        cfg = AgentConfig(
            agent_id=f"agent_{role}_{agent_id_suffix[:6]}",
            model=self.config.model,
            provider=self.config.provider,
            max_steps=min(role_cfg["max_steps"], self.config.max_steps_per_subtask),
            temperature=role_cfg["temperature"],
            allowed_tools=role_cfg["allowed_tools"] or [],
            memory_dir=self.config.memory_dir,
            audit_dir=self.config.audit_dir,
            run_dir=self.config.run_dir,
            role=role,
        )

        return Agent(config=cfg, api_key=self._api_key,
                     ollama_host=self._ollama_host)
