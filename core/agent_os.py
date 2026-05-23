"""
agent_os.py — The Agent Operating System runtime.

This is the single top-level entry point for all task execution.
It wires together every subsystem and enforces the full 9-stage pipeline:

  Stage 1  — Task Intake        (task_intake.py)
  Stage 2  — Model-Aware Route  (model_router.py)
  Stage 3  — Skill Retrieval    (skill_engine.py)
  Stage 4  — Memory Retrieval   (memory_engine.py)
  Stage 5  — Knowledge Retrieval (knowledge_retriever.py)
  Stage 6  — Planning           (planner.py)
  Stage 7  — Execution          (Agent loop via providers + tools)
  Stage 8  — Verification       (verifier.py)
  Stage 9  — Memory + Skill Save (memory_engine.py + skill_engine.py)

Key properties:
  - Model-aware: SLMs get tight loops; LLMs get richer context
  - Grounded: every claim must come from a tool call
  - Loop-safe: 4-detector loop prevention from observability
  - Skill-driven: always check skill registry before improvising
  - Verifiable: every output is checked before acceptance
  - Self-improving: successful workflows → new skills
"""

from __future__ import annotations

import os, re, sys, time, json, uuid, traceback, hashlib, platform, subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

# ── Core subsystems ───────────────────────────────────────────────────────────
from core.task_intake     import TaskIntake, process_task
from core.model_router    import ModelRuntimeConfig, route as router_route
from core.skill_engine    import SkillEngine, SkillMatch
from core.memory_engine   import MemoryEngine, MemoryPriority
from core.knowledge_retriever import KnowledgeRetriever, KnowledgeChunk
from core.planner         import Planner, Plan
from core.verifier        import Verifier, VerifyStatus

# ── Existing infrastructure (unchanged) ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from core.context_manager import ContextManager, ContextBudget, BUDGET_PRESETS
from core.memory          import MemoryStore                          # legacy compat
from observability.observability import (
    ObservabilityManager, EventType, LoopDetector, LoopDetectionResult
)
from providers.providers  import get_provider, LLMProvider, LLMResponse, parse_tool_calls
from tools.tools          import ToolRegistry, ToolResult


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentOSConfig:
    # Identity
    agent_id:   str  = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:6]}")

    # Model
    model:      str  = "llama-3.1-8b-instant"
    provider:   str  = "groq"
    temperature: float = 0.2
    stream:     bool = True

    # Hard limits (overridden by model router, but serve as floor/ceiling)
    max_steps:  int  = 20
    token_budget_usd: Optional[float] = None

    # Directories
    memory_dir: str  = ".agent_os/memory"
    audit_dir:  str  = ".agent_os/audit"
    run_dir:    str   = ".agent_os/runs"
    skill_dirs: list[str] = field(default_factory=list)

    # Behaviour flags
    verify_outputs:     bool = True
    max_retries:        int  = 1
    create_skills:      bool = True
    skill_promote_threshold: int = 3
    allowed_tools:      list[str] = field(default_factory=list)

    # Orchestration (only used when complexity == COMPOUND)
    use_subagents: bool = False

    # Role (injected by orchestrator)
    role:       str = "general"


# ─────────────────────────────────────────────────────────────────────────────
# Task result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id:     str
    task_text:   str
    success:     bool
    output:      str
    steps_taken: int
    tokens_used: int
    cost_usd:    float
    duration_ms: float
    trace_id:    str
    errors:      list[str] = field(default_factory=list)
    skill_used:  str = ""
    verify_status: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Execution state (one per task run)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunState:
    task_id:      str
    intake:       TaskIntake
    route_config: ModelRuntimeConfig
    plan:         Optional[Plan] = None
    skill_match:  Optional[SkillMatch] = None

    steps_taken:  int = 0
    tokens_in:    int = 0
    tokens_out:   int = 0
    cost_usd:     float = 0.0
    errors:       list[str] = field(default_factory=list)
    tool_history: list[dict] = field(default_factory=list)  # for skill creation
    final_output: str = ""
    done:         bool = False

    # Live tool outputs (for hallucination detection)
    tool_outputs: dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(
    config:     ModelRuntimeConfig,
    plan:       Plan,
    skill_hint: str,
    memory_chunks: list[str],
    knowledge_chunks: list[str],
    role:       str,
) -> str:
    """
    Assemble the full system prompt for this task run.
    Model-tier aware: SLMs get a stripped-down prompt; LLMs get the full version.
    """
    is_slm = config.is_slm

    # ── Core identity ─────────────────────────────────────────────────────────
    if role and role != "general":
        role_lines = {
            "researcher": "You are a RESEARCHER. Gather information only — do NOT modify files.",
            "coder":      "You are a CODER. Write and edit code. Always read before writing.",
            "tester":     "You are a TESTER. Run commands and verify results. Do NOT modify code.",
            "reviewer":   "You are a REVIEWER. Check correctness. Give a PASS/NEEDS_FIX verdict.",
        }
        identity = role_lines.get(role, "You are an expert AI agent.")
    else:
        identity = "You are an expert AI agent operating in a grounded execution environment."

    lines = [identity, ""]

    # ── Operating principles (SLM = minimal) ─────────────────────────────────
    if is_slm:
        lines += [
            "Rules:",
            "1. Use tools to gather facts before making claims.",
            "2. One action per step. Check the result before the next.",
            "3. Stop when the task is done. Signal with Done: <answer>",
            "",
        ]
    else:
        lines += [
            "Operating principles:",
            "- GROUNDED: every claim must come from a tool result, not assumption.",
            "- MINIMAL: make the smallest correct change needed.",
            "- VERIFIED: always verify your output before reporting done.",
            "- PRECISE: cite file names, line numbers, and exit codes.",
            "- NO LOOPS: if a tool fails twice with the same args, try a different approach.",
            "",
        ]

    # ── Tool format instruction ───────────────────────────────────────────────
    fmt = config.spec.prompt_format
    if fmt == "react":
        lines += [
            "Tool format — use EXACTLY this format:",
            "  Thought: <your reasoning>",
            "  Bash: <shell command>",
            "",
            "When done:",
            "  Done: <your complete answer>",
            "",
        ]
    elif fmt == "xml":
        lines += [
            "Use tools via XML tags. When done, include Done: <answer> on its own line.",
            "",
        ]

    # ── Memory ────────────────────────────────────────────────────────────────
    if memory_chunks:
        lines.append("## Relevant Memory")
        lines.extend(memory_chunks[:8 if not is_slm else 3])
        lines.append("")

    # ── External knowledge ────────────────────────────────────────────────────
    if knowledge_chunks and not is_slm:
        lines.append("## External Context")
        lines.extend(knowledge_chunks[:2])
        lines.append("")

    # ── Skill hint ────────────────────────────────────────────────────────────
    if skill_hint:
        if is_slm:
            # SLMs: just the step names
            step_lines = [l for l in skill_hint.split("\n")
                          if re.match(r'\s+\d+\.', l)]
            lines.append("## Suggested Approach")
            lines.extend(step_lines[:4])
            lines.append("")
        else:
            lines.append(skill_hint)
            lines.append("")

    # ── Plan ─────────────────────────────────────────────────────────────────
    lines.append(plan.to_prompt_fragment())
    lines.append("")

    # ── Environment snapshot ──────────────────────────────────────────────────
    env_ctx = _build_env_context(max_chars=400 if is_slm else 800)
    if env_ctx:
        lines.append("## Environment")
        lines.append(env_ctx)
        lines.append("")

    return "\n".join(lines)


def _build_env_context(max_chars: int = 600) -> str:
    """Minimal environment snapshot — CWD, OS, top-level files."""
    parts = []
    parts.append(f"OS: {platform.system()} {platform.release()}")
    parts.append(f"CWD: {os.getcwd()}")
    try:
        entries = []
        for p in sorted(Path(".").rglob("*")):
            parts_p = p.parts
            if any(seg in ("__pycache__", ".git") for seg in parts_p):
                continue
            if p.suffix == ".pyc":
                continue
            if len(parts_p) > 3:  # maxdepth=2 equivalent
                continue
            entries.append(str(p))
            if len(entries) >= 25:
                break
        parts.append("Files: " + ", ".join(entries[:15]))
    except Exception:
        pass
    try:
        git = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, timeout=3
        )
        if git.returncode == 0 and git.stdout.strip():
            parts.append("Git: " + git.stdout.strip()[:150])
    except Exception:
        pass
    return "\n".join(parts)[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent OS
# ─────────────────────────────────────────────────────────────────────────────

class AgentOS:
    """
    The Agent Operating System.

    Responsibilities:
      - Orchestrates all 9 pipeline stages
      - Maintains a model-aware context window
      - Executes a grounded ReAct / tool-call loop
      - Verifies outputs and retries on failures
      - Updates memory and skill registry after each run
    """

    def __init__(
        self,
        config:       AgentOSConfig = None,
        api_key:      str = None,
        ollama_host:  str = None,
    ):
        self.config = config or AgentOSConfig()
        cfg = self.config

        # ── Provider ──────────────────────────────────────────────────────────
        env_key_map = {
            "groq":      "GROQ_API_KEY",
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama":    "",
        }
        resolved_key = (
            api_key
            or os.environ.get(env_key_map.get(cfg.provider, ""), "")
            or None
        )
        self._provider: LLMProvider = get_provider(
            cfg.provider, api_key=resolved_key, host=ollama_host
        )

        # ── Subsystems ────────────────────────────────────────────────────────
        self._skill_engine = SkillEngine(
            skill_dirs=cfg.skill_dirs,
            generated_dir=str(Path(cfg.memory_dir).parent / "skills"),
        )
        self._memory = MemoryEngine(
            agent_id=cfg.agent_id,
            memory_dir=cfg.memory_dir,
        )
        self._knowledge = KnowledgeRetriever(cwd=os.getcwd())
        self._planner   = Planner()
        self._verifier  = Verifier()
        self._obs       = ObservabilityManager(audit_dir=cfg.audit_dir)

        # ── Tool registry ─────────────────────────────────────────────────────
        self._tools = ToolRegistry()

        # ── Persistent session memory (run log) ───────────────────────────────
        self._run_log_path = Path(cfg.run_dir) / "session_memory.jsonl"
        self._run_log_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Context manager (initialized per-task) ───────────────────────────
        self._ctx: Optional[ContextManager] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run_task(self, raw_task: str, task_id: str = None) -> TaskResult:
        """
        Execute a task through the full 9-stage pipeline.
        This is the only method external callers need to use.
        """
        task_id   = task_id or uuid.uuid4().hex[:16]
        trace_id  = self._obs.new_trace()
        start_ms  = time.time() * 1000

        self._print_header(raw_task, task_id)

        # ── Stage 1: Task Intake ─────────────────────────────────────────────
        intake = self._stage_intake(raw_task)
        self._print_stage(1, "INTAKE",
            f"{intake.summary()} | action_verbs={intake.action_verbs[:3]}")

        # ── Stage 2: Model-Aware Route ────────────────────────────────────────
        route_config = self._stage_route(intake)
        self._print_stage(2, "ROUTE", self._obs.tracer.__class__.__name__ and
                          f"tier={route_config.tier.value} steps={route_config.max_steps} "
                          f"planning={route_config.planning_depth}")

        # ── Stage 3: Skill Retrieval ──────────────────────────────────────────
        skill_match, skill_hint = self._stage_skill_retrieval(intake, route_config)
        skill_name = skill_match.skill.name if skill_match else "—"
        self._print_stage(3, "SKILL",
            f"matched={skill_name} score={skill_match.score:.3f}" if skill_match
            else "no match")

        # ── Stage 4: Memory Retrieval ─────────────────────────────────────────
        memory_chunks = self._stage_memory_retrieval(intake, route_config)
        self._print_stage(4, "MEMORY",
            f"{len(memory_chunks)} chunks from "
            f"{self._memory.stats()}")

        # ── Stage 5: External Knowledge ───────────────────────────────────────
        knowledge_chunks = self._stage_knowledge(intake)
        self._print_stage(5, "KNOWLEDGE",
            f"{len(knowledge_chunks)} external chunks" if knowledge_chunks
            else "none needed")

        # ── Stage 6: Planning ─────────────────────────────────────────────────
        plan = self._stage_plan(intake, route_config, skill_match)
        self._print_stage(6, "PLAN",
            f"{plan.estimated_steps} steps | depth={plan.depth} | skill={plan.skill_name or '—'}")

        # ── Build run state ───────────────────────────────────────────────────
        run_state = RunState(
            task_id=task_id,
            intake=intake,
            route_config=route_config,
            plan=plan,
            skill_match=skill_match,
        )

        # ── Stage 7: Execution ────────────────────────────────────────────────
        self._print_stage(7, "EXECUTE", "starting execution loop...")
        result = self._stage_execute(
            run_state=run_state,
            task_id=task_id,
            trace_id=trace_id,
            skill_hint=skill_hint,
            memory_chunks=memory_chunks,
            knowledge_chunks=knowledge_chunks,
        )

        # ── Stage 8: Verification ─────────────────────────────────────────────
        if self.config.verify_outputs:
            verify_result = self._stage_verify(
                run_state=run_state,
                result=result,
                plan=plan,
            )
            self._print_stage(8, "VERIFY", verify_result.summary())

            # Retry once if verification failed and it's viable
            if not verify_result.passed and verify_result.retry_viable:
                result = self._stage_retry(
                    run_state=run_state,
                    task_id=task_id,
                    trace_id=trace_id,
                    skill_hint=skill_hint,
                    memory_chunks=memory_chunks,
                    knowledge_chunks=knowledge_chunks,
                    retry_strategy=verify_result.retry_strategy,
                )
                verify_result = self._stage_verify(run_state, result, plan)
                self._print_stage(8, "VERIFY (retry)", verify_result.summary())

            result.verify_status = verify_result.status.value
            result.success = verify_result.passed

        # ── Stage 9: Memory + Skill Update ───────────────────────────────────
        self._stage_update_memory(run_state, result, intake)
        if self.config.create_skills:
            self._stage_update_skills(run_state, result, intake)
        self._print_stage(9, "PERSIST", "memory + skills updated")

        # ── Final summary ─────────────────────────────────────────────────────
        result.duration_ms = time.time() * 1000 - start_ms
        self._print_footer(result)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Stage implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _stage_intake(self, raw_task: str) -> TaskIntake:
        return process_task(raw_task)

    def _stage_route(self, intake: TaskIntake) -> ModelRuntimeConfig:
        override_tools = self.config.allowed_tools or None
        cfg = router_route(
            self.config.model,
            self.config.provider,
            intake,
            override_max_steps=self.config.max_steps if self.config.max_steps != 20 else None,
            override_allowed_tools=override_tools,
        )
        # Respect hard config ceiling
        cfg.max_steps = min(cfg.max_steps, self.config.max_steps)
        return cfg

    def _stage_skill_retrieval(
        self, intake: TaskIntake, config: ModelRuntimeConfig
    ) -> tuple[Optional[SkillMatch], str]:
        matches = self._skill_engine.retrieve(
            query=intake.skill_query,
            task_type=intake.task_type.value,
            top_k=3,
        )
        if not matches:
            return None, ""
        best = matches[0]
        if best.score < 0.08:
            return None, ""
        depth = config.skill_adapt_depth
        hint = self._skill_engine.adapt_for_prompt(
            best.skill, intake.task, config.max_skill_steps, depth
        )
        return best, hint

    def _stage_memory_retrieval(
        self, intake: TaskIntake, config: ModelRuntimeConfig
    ) -> list[str]:
        self._memory.clear_task_memory()
        retrieval = self._memory.retrieve_for_context(
            query=intake.skill_query,
            max_chunks=config.max_memory_chunks,
            max_per_bank=3 if not config.is_slm else 2,
        )
        return retrieval.to_context_chunks(
            max_per_bank=3,
            max_chars_each=120 if config.is_slm else 200,
        )

    def _stage_knowledge(self, intake: TaskIntake) -> list[str]:
        chunks = self._knowledge.retrieve_if_needed(
            task=intake.task,
            needs_external=intake.needs_external_knowledge,
            tech_stack=intake.tech_stack,
            max_chunks=2,
        )
        return [c.to_context_block() for c in chunks]

    def _stage_plan(
        self, intake: TaskIntake,
        config: ModelRuntimeConfig,
        skill_match: Optional[SkillMatch],
    ) -> Plan:
        prior_runs = self._memory.get_prior_run(intake.task, top_k=1)
        prior_summary = prior_runs[0].content if prior_runs else ""
        return self._planner.generate(
            intake=intake,
            config=config,
            skill_match=skill_match,
            prior_run_summary=prior_summary,
        )
    
    def _stage_execute(
        self,
        run_state:        RunState,
        task_id:          str,
        trace_id:         str,
        skill_hint:       str,
        memory_chunks:    list[str],
        knowledge_chunks: list[str],
        retry_hint:       str = "",
    ) -> TaskResult:
        """Core execution loop — ReAct / tool-call driven."""
        config = run_state.route_config
        intake = run_state.intake
        plan   = run_state.plan
        content = ""

        # ── Initialize context manager with model-specific budget ─────────────
        ctx_budget = config.context_budget
        if config.model not in BUDGET_PRESETS:
            BUDGET_PRESETS[config.model] = ContextBudget(
                total_context=ctx_budget["total_context"],
                max_output=ctx_budget["max_output"],
                system_prompt=ctx_budget["system_prompt"],
                tool_definitions=ctx_budget["tool_definitions"],
                active_memory=ctx_budget["active_memory"],
                task_context=ctx_budget["task_context"],
                conversation_history=ctx_budget["conversation_history"],
                current_input=ctx_budget["current_input"],
                safety_buffer=ctx_budget["safety_buffer"],
            )
        self._ctx = ContextManager(model=config.model)

        # ── Build system prompt ────────────────────────────────────────────────
        system_prompt = _build_system_prompt(
            config=config,
            plan=plan,
            skill_hint=skill_hint,
            memory_chunks=memory_chunks,
            knowledge_chunks=knowledge_chunks,
            role=self.config.role,
        )
        if retry_hint:
            system_prompt += f"\n\n## Retry Note\n{retry_hint}"

        # ── Build tool definitions ─────────────────────────────────────────────
        tool_defs = self._tools.get_definition(allowed=config.allowed_tools or None)

        # ── Loop ──────────────────────────────────────────────────────────────
        loop_detector = self._obs.loop
        errors: list[str] = []
        step = 0

        # Initial user message
        current_input = intake.task

        while step < config.step_limit:
            step += 1
            run_state.steps_taken = step

            if run_state.done:
                break

            # ── Budget check ──────────────────────────────────────────────────
            if self.config.token_budget_usd:
                if run_state.cost_usd >= self.config.token_budget_usd:
                    errors.append(f"token budget exhausted (${run_state.cost_usd:.5f})")
                    break

            # ── Build context ──────────────────────────────────────────────────
            original_input = current_input  # save before any mutation this step
            task_ctx = (
                f"Task: {intake.task}\n"
                f"Step {step}/{config.max_steps}\n"
                f"Type: {intake.task_type.value} | Risk: {intake.risk.value}"
            )
            messages, _snapshot = self._ctx.build(
                system_prompt=system_prompt,
                current_input=current_input,
                task_context=task_ctx,
                tool_definitions=tool_defs,
            )

            # ── LLM call ──────────────────────────────────────────────────────
            t0 = time.time()
            try:
                response: LLMResponse = self._provider.complete_with_retry(
                    messages=messages,
                    model=config.model,
                    max_tokens=config.spec.max_output,
                    temperature=self.config.temperature,
                    stream=self.config.stream and not config.is_slm,
                    tools=tool_defs if config.spec.supports_tools else None,
                )
            except Exception as e:
                errors.append(f"step {step} — LLM error: {e}")
                self._memory.save_error(str(e), tool="llm")
                break

            dur_ms = (time.time() - t0) * 1000
            cost = self._obs.record_llm_call(
                config.model, response.tokens_in, response.tokens_out,
                dur_ms, task_id=task_id, agent_id=self.config.agent_id,
            )
            run_state.tokens_in  += response.tokens_in
            run_state.tokens_out += response.tokens_out
            run_state.cost_usd   += cost

            content = response.content or ""

            # Print step progress
            print(f"\n  ── Step {step} ──────────────────────────────────")
            if content:
                print("  " + content[:200].replace("\n", "\n  "))

            # ── Parse tool calls / ReAct signals ─────────────────────────────
            tool_calls = parse_tool_calls(content)
            if not tool_calls and response.tool_calls:
                tool_calls = response.tool_calls

            # ── Check for done signal ─────────────────────────────────────────
            done_match = re.search(r'^Done:\s*(.+)$', content, re.MULTILINE | re.DOTALL)
            if done_match:
                run_state.final_output = done_match.group(1).strip()
                run_state.done = True
                self._ctx.record_turn(original_input, content)
                break

            # ── Execute tool calls ────────────────────────────────────────────
            tool_results_this_step = []
            for tc in tool_calls[:config.max_tools_per_step]:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", tc.get("arguments", {}))

                if tool_name == "task_done":
                    answer = (tool_args.get("answer") or
                               tool_args.get("result") or
                               content[:500])
                    run_state.final_output = str(answer)
                    run_state.done = True
                    break

                # ── Loop detection ────────────────────────────────────────────
                loop_result = loop_detector.check(tool_name, tool_args)
                if loop_result.stuck:
                    msg = f"LOOP [{loop_result.level}]: {loop_result.message}"
                    print(f"\n  ⚠️  {msg}")
                    if loop_result.level == "critical":
                        errors.append(msg)
                        current_input = (
                            f"CRITICAL LOOP DETECTED on {tool_name}. "
                            f"You MUST stop repeating this and either "
                            f"try a completely different approach or call task_done "
                            f"with your best current answer."
                        )
                        break
                    else:
                        current_input = (
                            f"WARNING: You've called {tool_name} with the same args "
                            f"multiple times. Try a different approach."
                        )
                        break

                # ── Execute tool ──────────────────────────────────────────────
                if tool_name not in (config.allowed_tools or self._tools.all_tool_names()):
                    result_str = f"[Tool '{tool_name}' not in allowed tools for this session]"
                    tool_results_this_step.append((tool_name, result_str))
                    continue

                print(f"\n  🔧 {tool_name}({str(tool_args)[:80]})")
                tr: ToolResult = self._tools.execute(tool_name, tool_args)
                result_str = tr.output if tr.success else f"ERROR: {tr.error}"

                # Quick verify the tool output
                ok, err_desc = self._verifier.quick_check_tool_output(
                    tool_name, result_str)
                if not ok:
                    self._memory.save_error(err_desc, tool=tool_name)

                # Record for loop detection
                loop_detector.record(tool_name, tool_args, result_str)

                # Store in run state
                tool_results_this_step.append((tool_name, result_str))
                run_state.tool_outputs[f"{tool_name}_{step}"] = result_str
                run_state.tool_history.append({
                    "tool":    tool_name,
                    "args":    tool_args,
                    "result":  result_str[:200],
                    "success": tr.success,
                    "step":    step,
                })

                # Save verified facts to memory
                if tr.success and tool_name in ("bash", "read_file"):
                    snippet = result_str[:150].strip()
                    if snippet:
                        self._memory.save_verified_fact(
                            f"{tool_name} → {snippet}",
                            tags=[tool_name, f"step_{step}"],
                        )

                print(f"     → {result_str[:150]}")

            if run_state.done:
                break

            # ── Compress tool results into next input ─────────────────────────
            if tool_results_this_step:
                results_block = "\n".join(
                    f"[{name} result]\n{out[:400]}"
                    for name, out in tool_results_this_step
                )
                current_input = results_block
            elif not tool_calls:
                # No tool calls — model is thinking or stuck
                current_input = (
                    "Continue. Use a tool to make progress or call task_done "
                    "with your best answer."
                )

            self._ctx.record_turn(original_input, content)

        # ── Finalize output ───────────────────────────────────────────────────
        final_out = run_state.final_output
        if not final_out:
            # Best-effort: extract last meaningful response
            final_out = self._extract_best_output(content, run_state)

        run_state.errors = errors

        return TaskResult(
            task_id=task_id,
            task_text=intake.task,
            success=run_state.done and not bool(errors),
            output=final_out,
            steps_taken=step,
            tokens_used=run_state.tokens_in + run_state.tokens_out,
            cost_usd=run_state.cost_usd,
            duration_ms=0.0,  # filled by run_task
            trace_id=trace_id,
            errors=errors,
            skill_used=run_state.skill_match.skill.name if run_state.skill_match else "",
        )

    def _stage_retry(
        self,
        run_state:        RunState,
        task_id:          str,
        trace_id:         str,
        skill_hint:       str,
        memory_chunks:    list[str],
        knowledge_chunks: list[str],
        retry_strategy:   str,
    ) -> TaskResult:
        print(f"\n  🔁 Retrying with strategy: {retry_strategy[:80]}")
        run_state.done = False
        run_state.final_output = ""
        return self._stage_execute(
            run_state=run_state,
            task_id=task_id + "_retry",
            trace_id=trace_id,
            skill_hint=skill_hint,
            memory_chunks=memory_chunks,
            knowledge_chunks=knowledge_chunks,
            retry_hint=f"Previous attempt failed. {retry_strategy}",
        )

    def _stage_verify(
        self,
        run_state: RunState,
        result:    TaskResult,
        plan:      Plan,
    ):
        tool_outputs_flat = list(run_state.tool_outputs.values())
        return self._verifier.verify(
            agent_response=result.output,
            tool_outputs=tool_outputs_flat,
            success_criteria=plan.success_criteria,
            task_text=run_state.intake.task,
            steps_taken=result.steps_taken,
            max_steps=run_state.route_config.max_steps,
        )

    def _stage_update_memory(
        self,
        run_state: RunState,
        result:    TaskResult,
        intake:    TaskIntake,
    ):
        # Always save the run outcome
        self._memory.save_run_outcome(
            task_id=result.task_id,
            task_text=intake.task[:100],
            success=result.success,
            summary=result.output[:200] if result.success else
                    ("; ".join(result.errors[:2]) or "failed"),
            duration_ms=result.duration_ms,
            tokens_used=result.tokens_used,
            tags=intake.tech_stack[:3],
        )

        # Save project facts from verified tool outputs
        for key, val in run_state.tool_outputs.items():
            if "read_file" in key and len(val) > 20:
                self._memory.save_project_fact(
                    f"file content: {val[:100]}",
                    tags=intake.file_refs[:2],
                )

        # Save errors seen
        for err in result.errors[:3]:
            self._memory.save_error(err)

        # Update skill stats
        if run_state.skill_match:
            self._skill_engine.record_use(
                run_state.skill_match.skill.name, result.success
            )

    def _stage_update_skills(
        self,
        run_state: RunState,
        result:    TaskResult,
        intake:    TaskIntake,
    ):
        if not result.success:
            return
        if len(run_state.tool_history) < 2:
            return
        new_skill = self._skill_engine.create_from_workflow(
            task=intake.task,
            tool_calls=run_state.tool_history,
            outcome=result.output[:120],
            task_type=intake.task_type.value,
            success=result.success,
        )
        if new_skill:
            saved = self._skill_engine.save_skill(new_skill)
            if saved:
                print(f"\n  💡 New skill created: '{new_skill.name}'")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_best_output(self, last_content: str, run_state: RunState) -> str:
        """Extract a usable output when task_done was never called."""
        if last_content and len(last_content) > 20:
            return last_content[:1000]
        # Fall back to last tool output
        if run_state.tool_outputs:
            last = list(run_state.tool_outputs.values())[-1]
            return last[:500]
        return "Task completed but no explicit output was captured."

    def _print_header(self, task: str, task_id: str):
        print(f"\n{'═'*62}")
        print(f"  🧠 AGENT OS  {self.config.agent_id}")
        print(f"  Model:  {self.config.model}  via  {self.config.provider}")
        print(f"  Task:   {task[:72]}")
        print(f"  ID:     {task_id}")
        print(f"{'═'*62}")

    def _print_stage(self, n: int, name: str, detail: str):
        print(f"  [{n}] {name:<12} {detail}")

    def _print_footer(self, result: TaskResult):
        icon = "✅" if result.success else "❌"
        print(f"\n{'─'*62}")
        print(f"  {icon}  steps={result.steps_taken}  "
              f"tokens={result.tokens_used:,}  "
              f"cost=${result.cost_usd:.5f}  "
              f"time={result.duration_ms:.0f}ms")
        if result.errors:
            print(f"  errors: {'; '.join(result.errors[:2])}")
        print(f"{'─'*62}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Dashboard & export
    # ─────────────────────────────────────────────────────────────────────────

    def dashboard(self):
        self._obs.print_dashboard()
        mem = self._memory.stats()
        print(f"  Memory banks: {mem}")
        print(f"  Skills loaded: {self._skill_engine.count}")

    def memory_export(self) -> str:
        return json.dumps(self._memory.export_summary(), indent=2)

    def list_skills(self) -> list[dict]:
        return self._skill_engine.list_skills()