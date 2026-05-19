"""
agent.py — The main agent harness runner.

This is the brain that ties together:
  - Context management (small window engineering)
  - Memory (tiered RAG-style)
  - Observability (automated, no manual instrumentation)
  - Loop detection (4 detectors)
  - Hallucination detection
  - Tool execution with permission checking
  - Skill loading

Designed so that a llama-3.1-8b-instant model can run effectively
within its 8192-token context window on real engineering tasks.
"""

import os
import time
import json
import uuid
import traceback
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

# Internal imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context_manager import ContextManager, BUDGET_PRESETS
from core.memory import MemoryStore, MemorySection, MemoryPriority
from observability.observability import (
    ObservabilityManager, EventType, AgentEvent
)
from providers.providers import get_provider, LLMProvider, LLMResponse
from tools.tools import ToolRegistry, ToolResult


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt — compressed for small context window
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an AI agent assistant. Complete tasks step by step using tools.

TOOL FORMAT — emit tool calls as:
<tool>TOOL_NAME</tool>
<args>{{"arg1": "value1", "arg2": "value2"}}</args>

RULES:
1. One tool call per response (small context window — be precise)
2. After each tool result, decide: next tool or final answer
3. If stuck (same result twice), try a different approach or report failure
4. Never guess file contents — read them first
5. Always verify before reporting success
6. If you can't complete a task, say WHY clearly

AGENT: {agent_id} | MODEL: {model} | STEP: {step}/{max_steps}
"""

# Compressed tool instruction (replaces verbose JSON schemas)
TOOL_INSTRUCTION = """
Use tools by emitting: <tool>name</tool><args>{{"key":"val"}}</args>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    agent_id: str            = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:6]}")
    model: str               = "llama-3.1-8b-instant"
    provider: str            = "groq"
    max_steps: int           = 20          # per task
    temperature: float       = 0.3         # low temp = more deterministic
    max_tokens_per_call: int = 1024
    memory_dir: str          = ".agent_os/memory"
    audit_dir: str           = ".agent_os/audit"
    allowed_tools: list[str] = field(default_factory=list)  # empty = all allowed
    require_approval_for: list[str] = field(default_factory=lambda: [
        "bash_destructive", "write_file", "delete_file"
    ])
    hallucination_threshold: float = 0.5   # flag if score > this
    loop_break_on_warning: bool    = False  # True = break on warning (not just critical)
    stream: bool                   = True


@dataclass
class TaskResult:
    task_id: str
    success: bool
    output: str
    steps_taken: int
    tokens_used: int
    cost_usd: float
    duration_ms: float
    trace_id: str
    errors: list[str]
    hallucination_flags: list[str]
    loop_breaks: int
    memory_entries_written: int


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """
    The core agent harness.

    Usage:
        agent = Agent(config=AgentConfig(model="llama-3.1-8b-instant"))
        result = agent.run_task("Find and fix the failing test in src/tests/")
    """

    def __init__(self, config: AgentConfig = None, api_key: str = None):
        self.config  = config or AgentConfig()
        self.obs     = ObservabilityManager(audit_dir=self.config.audit_dir)
        self.ctx     = ContextManager(model=self.config.model)
        self.memory  = MemoryStore(self.config.agent_id, self.config.memory_dir)
        self.tools   = ToolRegistry(
            allowed=self.config.allowed_tools or None,  # None = all
            require_approval=self.config.require_approval_for,
        )
        self.provider: LLMProvider = get_provider(
            self.config.provider, api_key=api_key or os.environ.get(
                f"{self.config.provider.upper()}_API_KEY", ""
            )
        )
        self._approval_callback: Optional[Callable[[str, dict], bool]] = None

    def set_approval_callback(self, callback: Callable[[str, dict], bool]):
        """Register a function that handles human approval requests."""
        self._approval_callback = callback

    # ── Main entry point ──────────────────────────────────────────────────

    def run_task(self, task: str, task_id: str = None,
                 context_hint: str = "") -> TaskResult:
        """
        Run a task through the agent loop.
        This is the main method — call this for everything.
        """
        task_id    = task_id or uuid.uuid4().hex[:12]
        trace_id   = self.obs.new_trace()
        start_time = time.time()

        # Observability: task start
        task_span = self.obs.start_span(
            "task", trace_id, agent_id=self.config.agent_id, task_id=task_id
        )
        self.obs.bus.emit_simple(
            EventType.TASK_CREATED,
            trace_id=trace_id,
            agent_id=self.config.agent_id,
            task_id=task_id,
            metadata={"task": task[:200], "model": self.config.model},
        )

        # Memory: write task as high-priority entry
        self.memory.upsert(
            MemorySection.CURRENT_TASKS,
            f"[ACTIVE] {task_id}: {task[:100]}",
            priority=MemoryPriority.HIGH,
            tags=[task_id],
        )

        errors: list[str]           = []
        hallucination_flags: list[str] = []
        loop_breaks: int            = 0
        step_count: int             = 0
        total_tokens: int           = 0
        total_cost: float           = 0.0
        final_output: str           = ""
        tool_results_for_hallcheck: dict[str, str] = {}

        print(f"\n{'='*60}")
        print(f"🦞 AGENT TASK START")
        print(f"   ID:    {task_id}")
        print(f"   Model: {self.config.model}")
        print(f"   Task:  {task[:80]}...")
        print(f"{'='*60}")

        for step in range(1, self.config.max_steps + 1):
            step_id   = f"{task_id}_s{step}"
            step_span = self.obs.start_span(
                f"step_{step}", trace_id, task_span.span_id,
                self.config.agent_id, task_id
            )
            step_count = step
            print(f"\n── Step {step}/{self.config.max_steps} ──")

            try:
                # ── Build context ─────────────────────────────────────────
                system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    agent_id=self.config.agent_id,
                    model=self.config.model,
                    step=step,
                    max_steps=self.config.max_steps,
                )
                task_context = (
                    f"Task: {task}\n"
                    f"Step: {step}/{self.config.max_steps}\n"
                    f"{'Hint: ' + context_hint if context_hint else ''}"
                ).strip()

                # Memory: retrieve relevant chunks for current task
                memory_chunks = self.memory.build_context_chunks(
                    query=task, max_chunks=6, max_tokens=400
                )

                # Tools: only load tools relevant for current step
                # (step-scoped to save tokens)
                step_tools = self.tools.get_definitions_for_step(step, task)

                # User message = task description (step 1) or "continue" (later steps)
                current_input = (
                    f"Task: {task}"
                    if step == 1
                    else "Continue with the next step. What's your next action?"
                )

                messages, ctx_snapshot = self.ctx.build(
                    system_prompt=system_prompt,
                    current_input=current_input,
                    task_context=task_context,
                    memory_chunks=memory_chunks,
                    tool_definitions=step_tools,
                    step_hint=context_hint,
                )

                # Log context build
                self.obs.bus.emit(AgentEvent(
                    trace_id=trace_id,
                    span_id=step_span.span_id,
                    event_type=EventType.CONTEXT_BUILT.value,
                    agent_id=self.config.agent_id,
                    task_id=task_id,
                    step_id=step_id,
                    context_tokens=ctx_snapshot.total_tokens,
                    context_headroom=ctx_snapshot.headroom_tokens,
                    truncations=ctx_snapshot.truncations,
                    metadata={
                        "zones": ctx_snapshot.zones,
                        "tools_loaded": ctx_snapshot.tools_loaded,
                        "memory_chunks": ctx_snapshot.memory_chunks_injected,
                        "summaries": ctx_snapshot.summaries_used,
                    },
                ))

                if ctx_snapshot.truncations:
                    print(f"   📦 Context: {ctx_snapshot.total_tokens}tk "
                          f"(headroom: {ctx_snapshot.headroom_tokens}tk) "
                          f"| {len(ctx_snapshot.truncations)} truncation(s)")

                # ── LLM call ──────────────────────────────────────────────
                llm_start = time.time()
                self.obs.bus.emit_simple(
                    EventType.LLM_CALL_START,
                    trace_id=trace_id, agent_id=self.config.agent_id,
                    task_id=task_id, model=self.config.model,
                )

                response: LLMResponse = self.provider.complete_with_retry(
                    messages=messages,
                    model=self.config.model,
                    max_tokens=self.config.max_tokens_per_call,
                    temperature=self.config.temperature,
                    stream=self.config.stream,
                )

                llm_duration = (time.time() - llm_start) * 1000
                cost = self.obs.record_llm_call(
                    model=self.config.model,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    duration_ms=llm_duration,
                    task_id=task_id,
                    agent_id=self.config.agent_id,
                )
                total_tokens += response.tokens_in + response.tokens_out
                total_cost   += cost

                print(f"   🤖 LLM: {response.tokens_in}in/{response.tokens_out}out tk "
                      f"| ${cost:.4f} | {llm_duration:.0f}ms")

                # ── Hallucination check ───────────────────────────────────
                h_score, h_flags = self.obs.analyze_hallucination(
                    response.content, tool_results_for_hallcheck
                )
                if h_score > self.config.hallucination_threshold:
                    flag_names = [f.kind for f in h_flags]
                    hallucination_flags.extend(flag_names)
                    print(f"   ⚠️  Hallucination score: {h_score:.2f} | flags: {flag_names}")

                # Print response preview
                preview = response.content[:150].replace('\n', ' ')
                print(f"   💬 Response: {preview}...")

                # ── Tool execution ────────────────────────────────────────
                tool_results_for_hallcheck = {}

                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call.get("args", {})

                        print(f"   🔧 Tool: {tool_name}({list(tool_args.keys())})")

                        # Loop detection BEFORE tool call
                        loop_result = self.obs.check_loop(tool_name, tool_args)
                        if loop_result.stuck:
                            print(f"   🔄 LOOP: [{loop_result.level}] {loop_result.message[:80]}")
                            loop_breaks += 1
                            if loop_result.level == "critical" or self.config.loop_break_on_warning:
                                # Inject loop warning into next context
                                self.ctx.record_turn(
                                    current_input,
                                    f"[LOOP BLOCKED] {loop_result.message}"
                                )
                                break  # Exit tool call loop, go to next step with warning

                        # Approval check
                        if self._requires_approval(tool_name, tool_args):
                            approved = self._request_approval(tool_name, tool_args, task_id)
                            if not approved:
                                result_str = f"[DENIED] Tool {tool_name} was not approved."
                                print(f"   ❌ Approval denied for {tool_name}")
                                self.ctx.record_turn(current_input, result_str)
                                continue

                        # Execute tool
                        tool_span = self.obs.start_span(
                            f"tool.{tool_name}", trace_id, step_span.span_id,
                            self.config.agent_id, task_id
                        )
                        self.obs.bus.emit_simple(
                            EventType.TOOL_CALL,
                            trace_id=trace_id, agent_id=self.config.agent_id,
                            task_id=task_id, metadata={"tool": tool_name, "args": tool_args},
                        )

                        tool_result: ToolResult = self.tools.execute(tool_name, tool_args)
                        self.obs.record_tool(tool_name, tool_args, tool_result.output)

                        # Store verified facts from tool results
                        self.obs.hallucination.update_verified_fact(
                            f"tool_{tool_name}_result",
                            tool_result.output[:100],
                            source=f"tool:{tool_name}",
                        )
                        tool_results_for_hallcheck[tool_name] = tool_result.output

                        self.obs.end_span(
                            tool_span,
                            status="ok" if tool_result.success else "error",
                            error=tool_result.error or "",
                        )
                        self.obs.bus.emit_simple(
                            EventType.TOOL_RESULT,
                            trace_id=trace_id, agent_id=self.config.agent_id,
                            task_id=task_id,
                            success=tool_result.success,
                            metadata={"tool": tool_name, "output_len": len(tool_result.output)},
                        )

                        output_preview = tool_result.output[:80].replace('\n',' ')
                        status_icon = "✓" if tool_result.success else "✗"
                        print(f"   {status_icon} Result: {output_preview}...")

                        # Update history with tool result
                        self.ctx.record_turn(
                            f"[Tool: {tool_name}({tool_args})]",
                            f"[Result]: {tool_result.output[:200]}",
                            tool_name=tool_name,
                            tool_args=tool_args,
                        )

                        # Memory: save errors to avoid repeat mistakes
                        if not tool_result.success and tool_result.error:
                            self.memory.upsert(
                                MemorySection.ERRORS_SEEN,
                                f"{tool_name} failed: {tool_result.error[:100]}",
                                priority=MemoryPriority.NORMAL,
                                ttl_hours=2,
                                tags=[task_id],
                            )

                else:
                    # No tool call — agent is giving a response/decision
                    self.ctx.record_turn(current_input, response.content)
                    final_output = response.content

                    # Check if task is complete
                    done_signals = [
                        "task complete", "task completed", "done", "finished",
                        "successfully", "i have completed", "the task is done",
                        "here is the answer", "here's the answer",
                    ]
                    if any(sig in response.content.lower() for sig in done_signals):
                        print(f"\n   ✅ Task completed at step {step}")
                        # Update memory: task done
                        self.memory.update_task_status(task_id, "DONE", response.content[:80])
                        self.obs.end_span(step_span)
                        break

                self.obs.end_span(step_span)

            except KeyboardInterrupt:
                errors.append("User interrupted")
                self.obs.end_span(step_span, status="error", error="interrupted")
                break
            except Exception as e:
                error_msg = f"Step {step} error: {type(e).__name__}: {e}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
                print(f"      {traceback.format_exc().splitlines()[-1]}")
                self.obs.end_span(step_span, status="error", error=error_msg)
                # Write error to memory so agent knows about it
                self.memory.upsert(
                    MemorySection.ERRORS_SEEN,
                    f"step_{step} error: {error_msg[:100]}",
                    priority=MemoryPriority.HIGH,
                    ttl_hours=1,
                )
                continue

        # ── Task complete ─────────────────────────────────────────────────
        duration_ms = (time.time() - start_time) * 1000
        success     = not bool(errors) and bool(final_output)

        self.obs.end_span(task_span, status="ok" if success else "error")
        self.obs.bus.emit_simple(
            EventType.TASK_COMPLETED if success else EventType.TASK_FAILED,
            trace_id=trace_id,
            agent_id=self.config.agent_id,
            task_id=task_id,
            success=success,
            duration_ms=duration_ms,
            metadata={
                "steps": step_count,
                "tokens": total_tokens,
                "cost": total_cost,
                "loops": loop_breaks,
            },
        )

        # Update memory: mark task status
        self.memory.update_task_status(
            task_id,
            "DONE" if success else "FAILED",
            final_output[:80] if final_output else (errors[-1][:80] if errors else ""),
        )

        result = TaskResult(
            task_id=task_id,
            success=success,
            output=final_output,
            steps_taken=step_count,
            tokens_used=total_tokens,
            cost_usd=total_cost,
            duration_ms=duration_ms,
            trace_id=trace_id,
            errors=errors,
            hallucination_flags=hallucination_flags,
            loop_breaks=loop_breaks,
            memory_entries_written=len(self.memory.get_section(MemorySection.CURRENT_TASKS)),
        )

        print(f"\n{'='*60}")
        print(f"{'✅ DONE' if success else '❌ FAILED'} | {step_count} steps | "
              f"{total_tokens:,}tk | ${total_cost:.4f} | {duration_ms:.0f}ms")
        if errors:
            print(f"   Errors: {'; '.join(errors[:2])}")
        if hallucination_flags:
            print(f"   Hallucination flags: {hallucination_flags}")
        print(f"{'='*60}\n")

        return result

    # ── Private helpers ───────────────────────────────────────────────────

    def _requires_approval(self, tool_name: str, args: dict) -> bool:
        if tool_name in self.config.require_approval_for:
            return True
        # Check for dangerous patterns in args
        args_str = json.dumps(args).lower()
        dangerous = ["production", "prod-db", "rm -rf", "drop table",
                     "delete from", "truncate"]
        return any(d in args_str for d in dangerous)

    def _request_approval(self, tool_name: str, args: dict, task_id: str) -> bool:
        self.obs.bus.emit_simple(
            EventType.BUDGET_THRESHOLD,
            task_id=task_id,
            agent_id=self.config.agent_id,
            metadata={"tool": tool_name, "args": args, "reason": "approval_required"},
        )
        if self._approval_callback:
            return self._approval_callback(tool_name, args)
        # Default: ask in terminal
        print(f"\n   ⚠️  APPROVAL REQUIRED")
        print(f"   Tool: {tool_name}")
        print(f"   Args: {json.dumps(args, indent=4)}")
        resp = input("   Approve? (y/n): ").strip().lower()
        return resp in ("y", "yes")

    def get_trace(self, trace_id: str) -> str:
        return self.obs.tracer.format_trace(trace_id)

    def dashboard(self):
        self.obs.print_dashboard()

    def memory_export(self) -> str:
        return self.memory.export_markdown()
