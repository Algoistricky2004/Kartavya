"""
agent.py — Real harness engineering.

How Claude Code / Aider / OpenCode actually work:
  1. Bootstrap: inject cwd, ls, git status, OS info BEFORE first LLM call
  2. ReAct loop: Thought → Action (bash cmd) → Observation → repeat
  3. Bash is the primary tool — find, grep, git, cat, ls — all bash
  4. Small models (llama3.2:3b): ReAct text format, NOT XML/JSON tool schemas
  5. Auto-complete: if task is satisfied by accumulated outputs, stop
  6. Hard loop kill: force complete with best result on critical repeat
  7. Persistent run memory: save results, inject similar past runs on step 1

ReAct prompt format (works reliably on 3b models):
  Thought: I need to find python files
  Bash: find . -name "*.py" | grep -v __pycache__

  → Agent sees: Observation: core/agent.py\ncli/main.py\n...

  Thought: I have all files now
  Done: core/agent.py\ncli/main.py\ntools/tools.py

IMPORTANT: task_done is NOT a model tool for small models.
  The harness detects "Done:" in model output and auto-completes.
"""

import os, re, sys, time, json, uuid, traceback, hashlib, platform, subprocess
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context_manager import ContextManager, BUDGET_PRESETS
from core.memory import MemoryStore, MemorySection, MemoryPriority
from observability.observability import ObservabilityManager, EventType
from providers.providers import (get_provider, LLMProvider, LLMResponse,
                                 parse_tool_calls, REACT_ONLY_MODELS)
from tools.tools import ToolRegistry, ToolResult


# ─────────────────────────────────────────────────────────────────────────────
# Environment Bootstrap — injected at step 1
# Real harnesses do this. Model needs to know WHERE it is.
# ─────────────────────────────────────────────────────────────────────────────

def build_env_context(max_chars: int = 800) -> str:
    """
    Build shell environment snapshot: cwd, top-level files, git status, OS.
    Uses pure Python (no Unix find command) so it works on Windows too.
    """
    import platform
    lines = []

    # OS + Python version
    os_name = platform.system()
    lines.append(f"OS: {os_name} {platform.release()}")
    lines.append(f"Python: {platform.python_version()}")

    # CWD
    cwd = os.getcwd()
    lines.append(f"CWD: {cwd}")

    # Directory listing — pure Python, works on Windows and Unix
    _excluded = {
        '__pycache__', '.git', '.hg', '.svn', 'node_modules',
        'venv', '.venv', 'env', 'virtualenv',
        'site-packages', 'dist-packages', '.tox', 'dist', 'build',
        '.pytest_cache', '.mypy_cache', '.eggs',
    }
    try:
        entries = []
        cwd_path = Path(cwd)
        for item in sorted(cwd_path.iterdir()):
            if item.name in _excluded or item.name.startswith('.'):
                continue
            entries.append(str(item.relative_to(cwd_path)))
            if item.is_dir():
                try:
                    for sub in sorted(item.iterdir())[:6]:
                        if sub.name not in _excluded and not sub.name.startswith('.'):
                            entries.append("  " + str(sub.relative_to(cwd_path)))
                except PermissionError:
                    pass
            if len(entries) >= 40:
                break
        lines.append("Files:\n" + "\n".join(entries[:40]))
    except Exception as e:
        lines.append(f"Files: (unavailable: {e})")

    # Git status (optional — may not be available on Windows without git)
    try:
        git = subprocess.run(
            ["git", "status", "--short", "--branch"],
            capture_output=True, text=True, timeout=3, cwd=cwd
        )
        if git.returncode == 0 and git.stdout.strip():
            lines.append(f"Git:\n{git.stdout.strip()[:200]}")
    except Exception:
        pass  # git not installed or not in a repo

    # Windows-specific guidance injected into env context
    if os_name == "Windows":
        lines.append(
            "SHELL: Windows cmd.exe. Use python -c '...' for complex file ops. "
            "Use provided tools (find_files, grep, read_file) instead of Unix bash commands."
        )

    ctx = "\n".join(lines)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n...(truncated)"
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# System prompts — one per model family
# ─────────────────────────────────────────────────────────────────────────────

# ReAct format for small local models (llama3.2:3b, gemma2:2b, phi3:mini etc.)
# These models DON'T reliably produce JSON tool calls.
# They DO reliably produce Thought/Bash/Done when shown an example.
REACT_SYSTEM = """\
You are a shell agent. Complete tasks by running commands and using tools.

ENVIRONMENT:
{env_context}

FORMAT — use EXACTLY this, one action per reply:
Thought: what you need to do
Bash: <the shell command>

When you have the final answer:
Thought: I have the answer
Done: <paste the actual command output here>

RULES:
- ONE Bash command per reply — nothing else
- After Bash runs, you see the output — then decide next step
- Write Done: only when you have the real answer from a command
- NEVER repeat a Bash command you already ran — use the output you got
- On Windows: prefer the find_files/grep/read_file tools over bash for file ops
- Step {step}/{max_steps}
"""

# Full tool-calling system prompt for capable models (Claude, GPT-4, Groq)
TOOL_SYSTEM = """\
You are a coding agent. Complete tasks step-by-step using tools.

ENVIRONMENT:
{env_context}

TOOL FORMAT (one tool per response, nothing else):
<tool>tool_name</tool>
<args>{{"key": "value"}}</args>

AVAILABLE TOOLS: bash, read_file, write_file, find_files, list_dir, grep, task_done

RULES:
1. ONE tool per response
2. After getting results — call task_done immediately with the actual output
3. NEVER call the same tool twice with the same args — you already have the answer
4. task_done output = real tool result, not a description
5. Step {step}/{max_steps}
"""

def _pick_system(model: str, env_context: str, step: int, max_steps: int) -> str:
    is_react = any(m in model.lower() for m in REACT_ONLY_MODELS) or \
               any(s in model.lower() for s in ["3b", "1b", "2b", "7b", "8b"])
    template = REACT_SYSTEM if is_react else TOOL_SYSTEM
    return template.format(env_context=env_context, step=step, max_steps=max_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Persistent Run Memory
# ─────────────────────────────────────────────────────────────────────────────

class RunMemory:
    def __init__(self, run_dir: str = ".agent_os/runs"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.run_dir / "session_memory.jsonl"
        self._cache: dict = {}
        self._load()

    def _load(self):
        if self.session_file.exists():
            for line in self.session_file.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    key = self._key(rec.get("task", ""))
                    self._cache[key] = rec
                except Exception:
                    pass

    def _key(self, task: str) -> str:
        return hashlib.md5(task.strip().lower().encode()).hexdigest()[:12]

    def save_result(self, task: str, output: str, success: bool,
                    steps: int, tokens: int, cost: float, duration_ms: float):
        rec = {
            "task": task, "output": output[:2000], "success": success,
            "steps": steps, "tokens": tokens, "cost": round(cost, 6),
            "duration_ms": round(duration_ms, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._cache[self._key(task)] = rec
        with open(self.session_file, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def get_similar(self, task: str, max_results: int = 2) -> list[dict]:
        words = set(task.lower().split())
        scored = []
        for rec in self._cache.values():
            prev = set(rec.get("task","").lower().split())
            overlap = len(words & prev) / max(len(words), 1)
            if overlap > 0.4:
                scored.append((overlap, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:max_results]]

    def format_for_context(self, task: str) -> str:
        similar = self.get_similar(task)
        if not similar:
            return ""
        lines = ["[Similar past tasks:]"]
        for r in similar[:2]:
            icon = "✓" if r["success"] else "✗"
            lines.append(f"{icon} Task: '{r['task'][:60]}'\n  Output: {r['output'][:150]}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Task Intent Detector — knows when we already have the answer
# ─────────────────────────────────────────────────────────────────────────────

class TaskIntentDetector:
    COMPLETION_MAP = {
        "list":   ["bash", "find_files", "list_dir"],
        "find":   ["bash", "find_files", "grep"],
        "search": ["bash", "grep", "web_search"],
        "show":   ["bash", "read_file", "find_files", "list_dir"],
        "read":   ["bash", "read_file"],
        "count":  ["bash", "find_files"],
        "check":  ["bash", "read_file", "grep"],
        "run":    ["bash"],
        "execute":["bash"],
        "git":    ["bash"],
    }

    # A result that is ONLY a shell command — not real output
    _CMD_RE = re.compile(
        r'^(find|ls|grep|cat|echo|python|pip|git|bash|sh|cd|rm|mv|cp|curl)\b',
        re.IGNORECASE)

    @classmethod
    def _is_real_output(cls, result: str) -> bool:
        """True if result is real command output, not a bare command string."""
        stripped = result.strip()
        has_newline = "\n" in stripped
        has_path    = "/" in stripped or ".py" in stripped or ".yaml" in stripped
        if cls._CMD_RE.match(stripped) and not has_newline and len(stripped) < 120:
            return False
        return has_newline or has_path or len(stripped) > 100

    @classmethod
    def is_satisfied(cls, task: str, tool_results: dict) -> tuple[bool, str]:
        task_lower = task.lower().strip()
        for verb, tools in cls.COMPLETION_MAP.items():
            if task_lower.startswith(verb) or f" {verb} " in f" {task_lower} ":
                for tool in tools:
                    if tool in tool_results:
                        result = tool_results[tool]
                        if len(result) > 30 and cls._is_real_output(result):
                            return True, result
        # Short task + any substantial real output
        if len(task.split()) <= 8:
            for tool, result in tool_results.items():
                if tool not in ("task_done",) and len(result) > 30:
                    if cls._is_real_output(result):
                        return True, result
        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Config & Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    agent_id:             str   = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:6]}")
    model:                str   = "llama-3.1-8b-instant"
    provider:             str   = "groq"
    max_steps:            int   = 20
    temperature:          float = 0.1      # lower = more deterministic commands
    max_tokens_per_call:  int   = 512
    memory_dir:           str   = ".agent_os/memory"
    audit_dir:            str   = ".agent_os/audit"
    run_dir:              str   = ".agent_os/runs"
    allowed_tools:        list  = field(default_factory=list)
    require_approval_for: list  = field(default_factory=lambda: ["bash_destructive"])
    hallucination_threshold: float = 0.6
    stream:               bool  = False    # False for reliable token counts on Ollama
    token_budget_usd:     float = None     # None = unlimited; set e.g. 0.10 for $0.10 cap
    role:                 str   = "general" # agent role: general | researcher | coder | tester | reviewer


@dataclass
class TaskResult:
    task_id:             str
    success:             bool
    output:              str
    steps_taken:         int
    tokens_used:         int
    cost_usd:            float
    duration_ms:         float
    trace_id:            str
    errors:              list
    hallucination_flags: list
    loop_breaks:         int


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, config: AgentConfig = None,
                 api_key: str = None, ollama_host: str = None):
        self.config  = config or AgentConfig()
        self.obs     = ObservabilityManager(audit_dir=self.config.audit_dir)
        self.ctx     = ContextManager(model=self.config.model)
        self.memory  = MemoryStore(self.config.agent_id, self.config.memory_dir)
        self.run_mem = RunMemory(self.config.run_dir)
        self.tools   = ToolRegistry(
            allowed=self.config.allowed_tools or None,
            require_approval=self.config.require_approval_for,
        )

        env_key_map = {"groq":"GROQ_API_KEY","openai":"OPENAI_API_KEY",
                       "anthropic":"ANTHROPIC_API_KEY","ollama":""}
        resolved_key = api_key or os.environ.get(env_key_map.get(self.config.provider,""),"")
        self.provider: LLMProvider = get_provider(
            self.config.provider, api_key=resolved_key or None, host=ollama_host)

        # Is this a tiny model that uses ReAct format?
        self._is_react = (
            self.config.model in REACT_ONLY_MODELS or
            any(s in self.config.model.lower() for s in ["3b","1b","2b","7b","8b","mini"])
        )

        # Build env context once per session (cwd, files, git)
        self._env_context = build_env_context()

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def run_task(self, task: str, task_id: str = None) -> TaskResult:
        task_id    = task_id or uuid.uuid4().hex[:12]
        trace_id   = self.obs.new_trace()
        start_time = time.time()

        past_ctx = self.run_mem.format_for_context(task)

        self.memory.upsert(MemorySection.CURRENT_TASKS,
                           f"[ACTIVE] {task_id}: {task[:100]}",
                           priority=MemoryPriority.HIGH, tags=[task_id])

        errors:              list  = []
        hallucination_flags: list  = []
        loop_breaks:         int   = 0
        step_count:          int   = 0
        total_tokens:        int   = 0
        total_cost:          float = 0.0
        final_output:        str   = ""
        task_complete:       bool  = False
        all_tool_results:    dict  = {}   # tool_name → output string

        self.obs.bus.emit_simple(EventType.TASK_CREATED, trace_id=trace_id,
                                 agent_id=self.config.agent_id, task_id=task_id,
                                 metadata={"task": task[:200]})

        print(f"\n{'='*60}")
        print(f"🚀 TASK  {task_id}")
        print(f"   Model:    {self.config.model}  via  {self.config.provider}")
        print(f"   Mode:     {'ReAct/bash' if self._is_react else 'tool-calling'}")
        print(f"   Task:     {task}")
        if past_ctx:
            print(f"   💾 Similar past run found")
        print(f"{'='*60}")

        task_span = self.obs.start_span("task", trace_id,
                                        agent_id=self.config.agent_id, task_id=task_id)

        for step in range(1, self.config.max_steps + 1):
            step_count = step
            step_span  = self.obs.start_span(f"step_{step}", trace_id, task_span.span_id,
                                             self.config.agent_id, task_id)
            print(f"\n── Step {step}/{self.config.max_steps} ──")

            # ── PRE-STEP: satisfied already? ──────────────────────────────
            if step > 1:
                satisfied, sat_result = TaskIntentDetector.is_satisfied(task, all_tool_results)
                if satisfied:
                    if self._is_react:
                        # Tiny models: auto-complete — don't trust model to say Done
                        print(f"   ✅ Auto-complete: have answer from tools")
                        final_output  = sat_result
                        task_complete = True
                        self.obs.end_span(step_span)
                        break
                    # Capable models: inject hint but let model call task_done
                    # (they might want to do more steps)

            try:
                # ── 1. Build prompt ───────────────────────────────────────
                system_prompt = _pick_system(
                    self.config.model, self._env_context, step, self.config.max_steps)

                # Build current_input: task + accumulated observations
                if step == 1:
                    # PRE-BOOTSTRAP: find and read relevant files before first LLM call.
                    # Claude Code always grounds in real file content before reasoning.
                    bootstrap = self._quick_bootstrap(task)
                    if bootstrap:
                        current_input = (
                            f"Task: {task}\n\n"
                            f"I already read the relevant files. Here is the ACTUAL content:\n\n"
                            f"{bootstrap}\n\n"
                            f"Answer using ONLY the file content above. "
                            f"Do NOT answer from training data. "
                            f"If this fully answers the task, write Done: <answer>."
                        )
                    else:
                        current_input = f"Task: {task}\nFirst: use find_files to locate relevant files."
                    if past_ctx:
                        current_input += f"\n\n{past_ctx}"
                else:
                    # Show what the model already knows
                    obs_lines = []
                    for tool_name, result in list(all_tool_results.items())[-3:]:
                        obs_lines.append(f"[{tool_name} output]:\n{result[:400]}")
                    obs_block = "\n\n".join(obs_lines)
                    current_input = (
                        f"Task: {task}\n\n"
                        f"Observations so far:\n{obs_block}\n\n"
                        f"{'→ You have the answer — write Done: <output>' if self._is_react else '→ Call task_done if you have the answer, otherwise use next tool.'}"
                    )

                memory_chunks = self.memory.build_context_chunks(
                    query=task, max_chunks=2, max_tokens=150)

                # Tool definitions — only for capable models
                step_tools = [] if self._is_react else \
                             self.tools.get_definitions_for_step(step, task)

                messages, ctx_snap = self.ctx.build(
                    system_prompt=system_prompt,
                    current_input=current_input,
                    task_context=f"Step {step}/{self.config.max_steps}",
                    memory_chunks=memory_chunks,
                    tool_definitions=step_tools,
                )

                print(f"   📦 ctx:{ctx_snap.total_tokens}tk "
                      f"headroom:{ctx_snap.headroom_tokens}tk "
                      f"tools:{len(step_tools)} mem:{ctx_snap.memory_chunks_injected}")
                for t in ctx_snap.truncations:
                    print(f"   ✂  {t}")

                # ── 2. LLM call ───────────────────────────────────────────
                llm_start = time.time()
                response: LLMResponse = self.provider.complete_with_retry(
                    messages=messages,
                    model=self.config.model,
                    max_tokens=self.config.max_tokens_per_call,
                    temperature=self.config.temperature,
                    stream=self.config.stream,
                    tools=step_tools or None,
                )
                llm_ms = (time.time() - llm_start) * 1000

                cost = self.obs.record_llm_call(
                    model=self.config.model,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    duration_ms=llm_ms,
                    task_id=task_id,
                    agent_id=self.config.agent_id,
                )
                total_tokens += response.tokens_in + response.tokens_out
                total_cost   += cost

                # ── Token budget enforcement ──────────────────────────────────
                if self.config.token_budget_usd and total_cost >= self.config.token_budget_usd:
                    budget_msg = (f"Token budget ${self.config.token_budget_usd:.4f} reached "
                                  f"(spent ${total_cost:.4f}). Stopping.")
                    print(f"   ⚠️  BUDGET CAP: {budget_msg}")
                    errors.append(budget_msg)
                    if all_tool_results:
                        final_output  = max(all_tool_results.values(), key=len)
                        task_complete = True
                    self.obs.end_span(step_span, status="error", error=budget_msg)
                    break

                print(f"   🤖 {response.tokens_in}in/{response.tokens_out}out "
                      f"| ${cost:.5f} | {llm_ms:.0f}ms")

                # Show model's raw thought (first 200 chars)
                preview = response.content[:200].replace('\n', ' ').strip()
                if preview:
                    print(f"   💬 {preview}")

                # ── 3. Parse tool calls ───────────────────────────────────
                tool_calls = response.tool_calls
                if not tool_calls and response.content:
                    tool_calls = parse_tool_calls(response.content)
                    if tool_calls:
                        print(f"   🔍 parsed from text: {tool_calls[0]['name']}")

                # ── 4. If no tool call — check if model gave final answer ─
                if not tool_calls:
                    content = response.content.strip()
                    print(f"   [NO TOOL PARSED]")

                    # For ReAct models: extract Bash/Done from raw text manually
                    if self._is_react:
                        extracted = _extract_react(content)
                        if extracted:
                            tool_calls = [extracted]
                            print(f"   🔍 manual extract: {extracted['name']}")

                    # For all models: if content looks like a final answer, use it
                    if not tool_calls and content:
                        done_signals = [
                            "here are", "found the following", "python files:",
                            "the files are", "results:", "output:", "i found",
                            ".py\n", ".py ",
                        ]
                        if any(s in content.lower() for s in done_signals):
                            final_output  = content
                            task_complete = True
                            print(f"   ✅ Final answer detected in text")
                            self.obs.end_span(step_span)
                            break

                    # No progress — inject hint and continue
                    if not tool_calls:
                        if self._is_react:
                            hint = (
                                f"[No command detected. Reply with EXACTLY:\n"
                                f"Thought: I need to find python files\n"
                                f"Bash: find . -name \"*.py\" -not -path \"*/__pycache__/*\"]"
                            )
                        else:
                            hint = ("[No tool call detected. Use: "
                                    "<tool>bash</tool>\\n<args>{\"command\":\"find . -name '*.py'\"}]</args>]")
                        self.ctx.record_turn(current_input, hint)
                        self.obs.end_span(step_span)
                        continue

                # ── 5. Execute tool calls ─────────────────────────────────
                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc.get("args", {})

                    if "_parse_error" in tool_args:
                        self.ctx.record_turn(current_input,
                            "[Parse error. Write the command more simply.]")
                        continue

                    print(f"   🔧 {tool_name}({json.dumps(tool_args)[:100]})")

                    # ── task_done ─────────────────────────────────────────
                    if tool_name == "task_done":
                        raw = (tool_args.get("output") or
                               tool_args.get("summary") or "Task completed.")
                        # If the model passed a shell command as the answer instead of
                        # the actual result, replace it with the best real tool output.
                        _looks_like_cmd = bool(re.match(
                            r'^(find|ls|grep|cat|echo|python|pip|git|cd|bash|sh)',
                            raw.strip()))
                        if all_tool_results and (_looks_like_cmd or len(raw.strip()) < 80):
                            raw = max(all_tool_results.values(), key=len, default=raw)
                        final_output  = raw
                        task_complete = True
                        print(f"   ✅ task_done")
                        print(f"\n   📋 OUTPUT:\n{final_output[:600]}\n")
                        self.memory.update_task_status(task_id, "DONE", raw[:80])
                        self.obs.end_span(step_span)
                        break

                    # ── Loop detection ────────────────────────────────────
                    loop_r = self.obs.check_loop(tool_name, tool_args)
                    if loop_r.stuck:
                        loop_breaks += 1
                        print(f"   🔄 LOOP [{loop_r.level}] count={loop_r.count}")
                        if loop_r.level == "critical":
                            if all_tool_results:
                                final_output  = max(all_tool_results.values(), key=len)
                                task_complete = True
                                print(f"   ⛔ Hard kill → force complete")
                                self.obs.end_span(step_span)
                                break
                            else:
                                self.ctx.record_turn(current_input,
                                    f"[BLOCKED: {tool_name} called {loop_r.count}x same args. "
                                    f"Try a completely different command.]")
                                break
                        else:
                            self.ctx.record_turn(current_input,
                                f"[WARNING: {tool_name} repeated. "
                                f"Use the output you already have, or run a different command.]")
                            break

                    # ── Approval ──────────────────────────────────────────
                    if self._requires_approval(tool_name, tool_args):
                        if not self._request_approval(tool_name, tool_args, task_id):
                            self.ctx.record_turn(current_input,
                                f"[DENIED: {tool_name}. Try another approach.]")
                            continue

                    # ── Execute ───────────────────────────────────────────
                    tool_span = self.obs.start_span(
                        f"tool.{tool_name}", trace_id, step_span.span_id,
                        self.config.agent_id, task_id)

                    tool_result: ToolResult = self.tools.execute(tool_name, tool_args)
                    self.obs.record_tool(tool_name, tool_args, tool_result.output)
                    self.obs.end_span(tool_span,
                                      status="ok" if tool_result.success else "error",
                                      error=tool_result.error or "")

                    if tool_result.success and tool_result.output:
                        # Key by tool_name + step so repeated bash calls don't overwrite
                        result_key = tool_name if tool_name != "bash" else f"bash_{step}"
                        all_tool_results[result_key] = tool_result.output
                        # Also keep the canonical key updated (for intent detector compat)
                        all_tool_results[tool_name] = tool_result.output
                        self.obs.hallucination.update_verified_fact(
                            f"{tool_name}_last", tool_result.output[:200],
                            source=f"tool:{tool_name}")

                    icon = "✓" if tool_result.success else "✗"
                    print(f"   {icon} {tool_result.output[:200].replace(chr(10),' ')}")
                    if tool_result.error:
                        print(f"     ERR: {tool_result.error[:100]}")

                    # ── Feed observation back — this is the harness loop ──
                    # Model sees: what it ran + what happened
                    if self._is_react:
                        # ReAct format: Observation block
                        if tool_result.success:
                            observation = (
                                f"Observation:\n{tool_result.output[:600]}\n\n"
                                f"Now decide: is the task done? If yes write Done: <output>. "
                                f"If not, write your next Thought: and Bash: command."
                            )
                        else:
                            observation = (
                                f"Observation: ERROR — {tool_result.error or 'command failed'}\n"
                                f"Try a different command."
                            )
                    else:
                        # Tool format: structured result
                        if tool_result.success:
                            observation = (
                                f"[{tool_name} result]:\n{tool_result.output[:500]}\n\n"
                                f"Tools used so far: {list(all_tool_results.keys())}\n"
                                f"→ If this answers '{task[:60]}', call task_done with this output."
                            )
                        else:
                            observation = (
                                f"[{tool_name} FAILED]: {tool_result.error}\n"
                                f"→ Try a different approach."
                            )

                    self.ctx.record_turn(f"[ran {tool_name}]", observation)

                    # ── Immediate satisfaction check for tiny models ───────
                    if self._is_react and tool_result.success:
                        sat, sat_result = TaskIntentDetector.is_satisfied(
                            task, all_tool_results)
                        if sat:
                            # Don't wait for model — complete now
                            print(f"   ✅ Auto-complete: task satisfied by bash output")
                            final_output  = sat_result
                            task_complete = True
                            self.obs.end_span(step_span)
                            break

            except KeyboardInterrupt:
                errors.append("interrupted by user")
                self.obs.end_span(step_span, status="error", error="interrupted")
                break
            except Exception as e:
                err = f"step {step} — {type(e).__name__}: {e}"
                errors.append(err)
                print(f"   ❌ {err}")
                print(f"      {traceback.format_exc().splitlines()[-1]}")
                self.obs.end_span(step_span, status="error", error=err)
                self.memory.upsert(MemorySection.ERRORS_SEEN, err[:120],
                                   priority=MemoryPriority.HIGH, ttl_hours=1)
                # Fast-fail on fatal errors — no point retrying all 20 steps
                fatal_signals = [
                    "Failed to connect", "Connection refused", "ollama serve",
                    "api_key", "401", "403", "ImportError", "ModuleNotFoundError",
                ]
                if any(sig.lower() in str(e).lower() for sig in fatal_signals):
                    print(f"   🛑 Fatal error — stopping immediately (not retrying 20 steps)")
                    break
                continue

            self.obs.end_span(step_span)

            if task_complete:
                break

        # ── Wrap up ───────────────────────────────────────────────────────
        duration_ms = (time.time() - start_time) * 1000
        success     = task_complete and bool(final_output) and not errors

        self.obs.end_span(task_span, status="ok" if success else "error")
        self.obs.bus.emit_simple(
            EventType.TASK_COMPLETED if success else EventType.TASK_FAILED,
            trace_id=trace_id, agent_id=self.config.agent_id, task_id=task_id,
            success=success, duration_ms=duration_ms,
            metadata={"steps": step_count, "tokens": total_tokens, "cost": total_cost})

        if not task_complete:
            self.memory.update_task_status(task_id, "INCOMPLETE",
                                           f"hit max_steps={self.config.max_steps}")

        self.run_mem.save_result(
            task=task, output=final_output, success=success,
            steps=step_count, tokens=total_tokens,
            cost=total_cost, duration_ms=duration_ms)

        result = TaskResult(
            task_id=task_id, success=success, output=final_output,
            steps_taken=step_count, tokens_used=total_tokens,
            cost_usd=total_cost, duration_ms=duration_ms,
            trace_id=trace_id, errors=errors,
            hallucination_flags=hallucination_flags, loop_breaks=loop_breaks,
        )

        print(f"\n{'='*60}")
        icon = "✅ DONE" if success else ("⚠️  INCOMPLETE" if not errors else "❌ FAILED")
        print(f"{icon} | {step_count} steps | {total_tokens:,}tk | "
              f"${total_cost:.5f} | {duration_ms:.0f}ms")
        if errors:
            print(f"Errors: {'; '.join(errors[:2])}")
        print(f"{'='*60}\n")

        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    def _requires_approval(self, tool: str, args: dict) -> bool:
        if tool in self.config.require_approval_for:
            return True
        args_s = json.dumps(args).lower()
        return any(d in args_s for d in ["rm -rf /", "format c:", "drop table",
                                          "truncate", "--force --all"])

    def _request_approval(self, tool: str, args: dict, task_id: str) -> bool:
        if hasattr(self, "_approval_cb") and self._approval_cb:
            return self._approval_cb(tool, args)
        print(f"\n   ⚠️  APPROVAL REQUIRED: {tool}({json.dumps(args)[:100]})")
        try:
            return input("   Approve? (y/n): ").strip().lower() in ("y","yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def get_trace(self, trace_id: str) -> str:
        return self.obs.tracer.format_trace(trace_id)

    def dashboard(self):
        self.obs.print_dashboard()

    def memory_export(self) -> str:
        return self.memory.export_markdown()


# ─────────────────────────────────────────────────────────────────────────────
# ReAct extractor — last-resort manual parse for llama3.2:3b
# ─────────────────────────────────────────────────────────────────────────────

def _extract_react(content: str) -> Optional[dict]:
    """
    Manually extract Bash or Done from model text when the regex parser misses it.
    Handles llama3.2:3b's inconsistent output formatting.
    """
    lines = content.strip().splitlines()

    # Scan line by line for patterns
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Bash command
        if re.match(r'^(Bash|Action|Command)\s*:', stripped, re.IGNORECASE):
            cmd = re.sub(r'^(Bash|Action|Command)\s*:\s*', '', stripped, flags=re.IGNORECASE).strip()
            cmd = cmd.strip('`').strip()
            # Check next line for multi-line command (code block)
            if not cmd and i + 1 < len(lines):
                cmd = lines[i+1].strip().strip('`')
            if cmd:
                return {"name": "bash", "args": {"command": cmd}}

        # Done / Final answer
        if re.match(r'^(Done|Answer|Final|Result)\s*:', stripped, re.IGNORECASE):
            answer = re.sub(r'^(Done|Answer|Final|Result)\s*:\s*', '', stripped,
                            flags=re.IGNORECASE).strip()
            # Collect remaining lines as part of answer
            if i + 1 < len(lines):
                answer = answer + "\n" + "\n".join(lines[i+1:])
            answer = answer.strip()
            if answer:
                return {"name": "task_done",
                        "args": {"summary": answer[:100], "output": answer}}

    return None