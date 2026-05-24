"""
model_router.py — Model-aware routing for the Agent OS.

This module answers ONE question: given the model name and the task intake,
HOW should the agent behave?

Key decisions:
  - Is this an SLM (≤ 14B params, small context) or an LLM (>14B / large context)?
  - What is the available context budget?
  - How deep should planning go?
  - How many steps are allowed?
  - Which tools are available?
  - What prompt format should be used? (ReAct text vs XML tool calls)
  - How aggressive should summarization be?

Design: purely deterministic — no LLM calls, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.task_intake import TaskIntake, Complexity, TaskType


# ─────────────────────────────────────────────────────────────────────────────
# Model tier
# ─────────────────────────────────────────────────────────────────────────────

class ModelTier(str, Enum):
    SLM = "slm"   # Small Language Model: ≤14B params, ≤16K context
    LLM = "llm"   # Large Language Model: >14B or >32K context


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# Keyed on model name substring patterns (lowercase, in priority order).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    tier:          ModelTier
    context_tokens: int       # usable context window
    supports_tools: bool      # native tool calling
    prompt_format: str        # "react" | "xml" | "native"
    max_output:    int
    params_b:      float      # approximate parameter count (billions)

    # Context zone presets (tokens)
    system_tokens:      int = 384
    tool_def_tokens:    int = 768
    memory_tokens:      int = 512
    task_ctx_tokens:    int = 256
    history_tokens:     int = 2048
    input_tokens:       int = 1024
    safety_buffer:      int = 256


_MODEL_REGISTRY: list[tuple[str, ModelSpec]] = [
    # ── Tiny / embedded SLMs ─────────────────────────────────────────────────
    ("llama3.2:1b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=2048, supports_tools=False,
        prompt_format="react", max_output=256, params_b=1.0,
        system_tokens=128, tool_def_tokens=256, memory_tokens=128,
        task_ctx_tokens=64, history_tokens=512, input_tokens=256, safety_buffer=64,
    )),
    ("gemma2:2b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=4096, supports_tools=False,
        prompt_format="react", max_output=512, params_b=2.0,
        system_tokens=200, tool_def_tokens=400, memory_tokens=200,
        task_ctx_tokens=128, history_tokens=1024, input_tokens=512, safety_buffer=128,
    )),
    ("phi3:mini", ModelSpec(
        tier=ModelTier.SLM, context_tokens=4096, supports_tools=False,
        prompt_format="react", max_output=512, params_b=3.8,
        system_tokens=200, tool_def_tokens=400, memory_tokens=200,
        task_ctx_tokens=128, history_tokens=1024, input_tokens=512, safety_buffer=128,
    )),
    # ── Small SLMs (3B-8B) ────────────────────────────────────────────────────
    ("llama3.2:3b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=4096, supports_tools=False,
        prompt_format="react", max_output=512, params_b=3.0,
        system_tokens=200, tool_def_tokens=400, memory_tokens=200,
        task_ctx_tokens=128, history_tokens=1024, input_tokens=512, safety_buffer=128,
    )),
    ("qwen2.5:7b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=8192, supports_tools=True,
        prompt_format="react", max_output=1024, params_b=7.0,
        system_tokens=256, tool_def_tokens=512, memory_tokens=256,
        task_ctx_tokens=128, history_tokens=2048, input_tokens=768, safety_buffer=192,
    )),
    ("mistral:7b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=8192, supports_tools=False,
        prompt_format="react", max_output=1024, params_b=7.0,
        system_tokens=256, tool_def_tokens=512, memory_tokens=256,
        task_ctx_tokens=128, history_tokens=2048, input_tokens=768, safety_buffer=192,
    )),
    ("llama3.1:8b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=8192, supports_tools=True,
        prompt_format="xml", max_output=1024, params_b=8.0,
        system_tokens=256, tool_def_tokens=512, memory_tokens=256,
        task_ctx_tokens=128, history_tokens=2048, input_tokens=768, safety_buffer=192,
    )),
    ("llama-3.1-8b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=8192, supports_tools=True,
        prompt_format="xml", max_output=1024, params_b=8.0,
        system_tokens=256, tool_def_tokens=512, memory_tokens=256,
        task_ctx_tokens=128, history_tokens=2048, input_tokens=768, safety_buffer=192,
    )),
    ("gemma2-9b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=8192, supports_tools=False,
        prompt_format="react", max_output=1024, params_b=9.0,
        system_tokens=256, tool_def_tokens=512, memory_tokens=256,
        task_ctx_tokens=128, history_tokens=2048, input_tokens=768, safety_buffer=192,
    )),
    # ── Mid-size SLMs (14B) ───────────────────────────────────────────────────
    ("qwen2.5:14b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=16384, supports_tools=True,
        prompt_format="xml", max_output=2048, params_b=14.0,
        system_tokens=512, tool_def_tokens=1024, memory_tokens=512,
        task_ctx_tokens=256, history_tokens=4096, input_tokens=1536, safety_buffer=256,
    )),
    ("mixtral-8x7b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=32768, supports_tools=True,
        prompt_format="xml", max_output=2048, params_b=47.0,  # MoE ~47B active
        system_tokens=512, tool_def_tokens=1024, memory_tokens=512,
        task_ctx_tokens=256, history_tokens=8192, input_tokens=2048, safety_buffer=512,
    )),
    # ── LLMs ─────────────────────────────────────────────────────────────────
    ("llama-3.1-70b", ModelSpec(
        tier=ModelTier.LLM, context_tokens=32768, supports_tools=True,
        prompt_format="xml", max_output=4096, params_b=70.0,
        system_tokens=800, tool_def_tokens=1500, memory_tokens=1024,
        task_ctx_tokens=512, history_tokens=8192, input_tokens=2048, safety_buffer=512,
    )),
    ("llama-3.3-70b", ModelSpec(
        tier=ModelTier.LLM, context_tokens=32768, supports_tools=True,
        prompt_format="xml", max_output=4096, params_b=70.0,
        system_tokens=800, tool_def_tokens=1500, memory_tokens=1024,
        task_ctx_tokens=512, history_tokens=8192, input_tokens=2048, safety_buffer=512,
    )),
    ("llama3.3:70b", ModelSpec(
        tier=ModelTier.LLM, context_tokens=32768, supports_tools=True,
        prompt_format="xml", max_output=4096, params_b=70.0,
        system_tokens=800, tool_def_tokens=1500, memory_tokens=1024,
        task_ctx_tokens=512, history_tokens=8192, input_tokens=2048, safety_buffer=512,
    )),
    ("gpt-4o-mini", ModelSpec(
        tier=ModelTier.LLM, context_tokens=128000, supports_tools=True,
        prompt_format="native", max_output=4096, params_b=8.0,  # not disclosed
        system_tokens=1024, tool_def_tokens=2048, memory_tokens=2048,
        task_ctx_tokens=512, history_tokens=32768, input_tokens=4096, safety_buffer=1024,
    )),
    ("gpt-4o", ModelSpec(
        tier=ModelTier.LLM, context_tokens=128000, supports_tools=True,
        prompt_format="native", max_output=4096, params_b=200.0,
        system_tokens=1024, tool_def_tokens=2048, memory_tokens=4096,
        task_ctx_tokens=1024, history_tokens=65536, input_tokens=8192, safety_buffer=2048,
    )),
    ("claude-3-haiku", ModelSpec(
        tier=ModelTier.LLM, context_tokens=200000, supports_tools=True,
        prompt_format="native", max_output=4096, params_b=20.0,
        system_tokens=1024, tool_def_tokens=2048, memory_tokens=2048,
        task_ctx_tokens=512, history_tokens=65536, input_tokens=4096, safety_buffer=1024,
    )),
    ("claude-3-5-sonnet", ModelSpec(
        tier=ModelTier.LLM, context_tokens=200000, supports_tools=True,
        prompt_format="native", max_output=8192, params_b=100.0,
        system_tokens=2048, tool_def_tokens=4096, memory_tokens=8192,
        task_ctx_tokens=2048, history_tokens=100000, input_tokens=8192, safety_buffer=4096,
    )),
    ("claude-sonnet-4", ModelSpec(
        tier=ModelTier.LLM, context_tokens=200000, supports_tools=True,
        prompt_format="native", max_output=8192, params_b=100.0,
        system_tokens=2048, tool_def_tokens=4096, memory_tokens=8192,
        task_ctx_tokens=2048, history_tokens=100000, input_tokens=8192, safety_buffer=4096,
    )),
    ("claude-opus-4", ModelSpec(
        tier=ModelTier.LLM, context_tokens=200000, supports_tools=True,
        prompt_format="native", max_output=8192, params_b=200.0,
        system_tokens=2048, tool_def_tokens=4096, memory_tokens=8192,
        task_ctx_tokens=2048, history_tokens=100000, input_tokens=8192, safety_buffer=4096,
    )),
    # ── Additional Ollama community models ───────────────────────────────────
    ("nemotron-mini", ModelSpec(
        tier=ModelTier.SLM, context_tokens=4096, supports_tools=False,
        prompt_format="react", max_output=512, params_b=4.0,
        system_tokens=200, tool_def_tokens=400, memory_tokens=200,
        task_ctx_tokens=128, history_tokens=1024, input_tokens=512, safety_buffer=128,
    )),
    ("nemotron", ModelSpec(
        tier=ModelTier.SLM, context_tokens=4096, supports_tools=False,
        prompt_format="react", max_output=512, params_b=4.0,
        system_tokens=200, tool_def_tokens=400, memory_tokens=200,
        task_ctx_tokens=128, history_tokens=1024, input_tokens=512, safety_buffer=128,
    )),
    ("deepseek-r1:1.5b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=4096, supports_tools=False,
        prompt_format="react", max_output=512, params_b=1.5,
        system_tokens=150, tool_def_tokens=300, memory_tokens=150,
        task_ctx_tokens=64, history_tokens=768, input_tokens=384, safety_buffer=96,
    )),
    ("deepseek-r1:7b", ModelSpec(
        tier=ModelTier.SLM, context_tokens=8192, supports_tools=False,
        prompt_format="react", max_output=1024, params_b=7.0,
        system_tokens=256, tool_def_tokens=512, memory_tokens=256,
        task_ctx_tokens=128, history_tokens=2048, input_tokens=768, safety_buffer=192,
    )),
    ("phi4", ModelSpec(
        tier=ModelTier.SLM, context_tokens=16384, supports_tools=True,
        prompt_format="xml", max_output=2048, params_b=14.0,
        system_tokens=512, tool_def_tokens=1024, memory_tokens=512,
        task_ctx_tokens=256, history_tokens=4096, input_tokens=1536, safety_buffer=256,
    )),
    ("codellama", ModelSpec(
        tier=ModelTier.SLM, context_tokens=16384, supports_tools=False,
        prompt_format="react", max_output=2048, params_b=7.0,
        system_tokens=256, tool_def_tokens=512, memory_tokens=256,
        task_ctx_tokens=128, history_tokens=4096, input_tokens=1024, safety_buffer=256,
    )),
]

# Default spec for unknown models
_DEFAULT_SLM_SPEC = ModelSpec(
    tier=ModelTier.SLM, context_tokens=8192, supports_tools=True,
    prompt_format="react", max_output=1024, params_b=8.0,
    system_tokens=256, tool_def_tokens=512, memory_tokens=256,
    task_ctx_tokens=128, history_tokens=2048, input_tokens=768, safety_buffer=192,
)

_DEFAULT_LLM_SPEC = ModelSpec(
    tier=ModelTier.LLM, context_tokens=32768, supports_tools=True,
    prompt_format="xml", max_output=4096, params_b=70.0,
    system_tokens=800, tool_def_tokens=1500, memory_tokens=1024,
    task_ctx_tokens=512, history_tokens=8192, input_tokens=2048, safety_buffer=512,
)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime configuration produced by the router
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelRuntimeConfig:
    model:          str
    provider:       str
    tier:           ModelTier
    spec:           ModelSpec

    # Step budget
    max_steps:      int
    step_limit:     int          # hard stop (never exceeded)

    # Memory
    max_memory_chunks:   int     # how many memory chunks to inject
    memory_summary_mode: str     # "compact" | "full"

    # Skill
    max_skill_steps:     int     # max steps from a matched skill
    skill_adapt_depth:   str     # "shallow" | "deep"

    # Planning
    planning_depth:     str      # "minimal" | "standard" | "deep"
    max_plan_steps:     int      # max number of plan steps
    use_subagents:      bool     # allow orchestration

    # Tools
    allowed_tools:      list[str]
    max_tools_per_step: int

    # Context
    summarize_aggressively: bool
    compress_tool_results:  bool

    @property
    def is_slm(self) -> bool:
        return self.tier == ModelTier.SLM

    @property
    def context_budget(self) -> dict:
        s = self.spec
        return {
            "total_context":    s.context_tokens,
            "max_output":       s.max_output,
            "system_prompt":    s.system_tokens,
            "tool_definitions": s.tool_def_tokens,
            "active_memory":    s.memory_tokens,
            "task_context":     s.task_ctx_tokens,
            "conversation_history": s.history_tokens,
            "current_input":    s.input_tokens,
            "safety_buffer":    s.safety_buffer,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

# Tools restricted to SLMs (simpler, fewer choices)
_SLM_CORE_TOOLS = [
    "bash", "read_file", "write_file", "list_dir",
    "grep", "find_files", "task_done", "memory_read",
]

_LLM_FULL_TOOLS = [
    "bash", "read_file", "write_file", "append_file",
    "list_dir", "grep", "find_files", "web_search",
    "memory_read", "memory_write", "task_done", "ask_human",
]

# Task type → tool recommendation
_TASK_TOOL_BIAS: dict[str, list[str]] = {
    "coding":     ["bash", "read_file", "write_file", "grep", "find_files", "task_done"],
    "research":   ["grep", "find_files", "read_file", "bash", "web_search", "task_done"],
    "automation": ["bash", "find_files", "read_file", "write_file", "task_done"],
    "ops":        ["bash", "read_file", "write_file", "task_done"],
    "data":       ["bash", "read_file", "write_file", "grep", "task_done"],
    "creative":   ["read_file", "write_file", "bash", "task_done"],
    "mixed":      _LLM_FULL_TOOLS,
}


class ModelRouter:
    """
    Route a model name + task intake to a fully-specified runtime config.

    Usage:
        config = ModelRouter().route("llama3.2:3b", "groq", intake)
    """

    def lookup_spec(self, model: str) -> ModelSpec:
        """Find the closest spec for the given model name."""
        ml = model.lower()
        for pattern, spec in _MODEL_REGISTRY:
            if pattern in ml:
                return spec
        # Heuristic fallback: try to detect tier from name
        if any(x in ml for x in ("70b", "72b", "65b", "claude", "gpt-4", "o3", "o4")):
            return _DEFAULT_LLM_SPEC
        return _DEFAULT_SLM_SPEC

    def route(
        self,
        model:    str,
        provider: str,
        intake:   TaskIntake,
        override_max_steps: Optional[int] = None,
        override_allowed_tools: Optional[list[str]] = None,
    ) -> ModelRuntimeConfig:

        spec = self.lookup_spec(model)
        tier = spec.tier

        # ── Step budgets ──────────────────────────────────────────────────────
        if tier == ModelTier.SLM:
            base_max_steps = {
                "simple":   4,
                "moderate": 8,
                "complex":  12,
                "compound": 8,   # SLM shouldn't do compound alone
            }.get(intake.complexity.value, 8)
        else:
            base_max_steps = {
                "simple":   6,
                "moderate": 15,
                "complex":  25,
                "compound": 20,
            }.get(intake.complexity.value, 15)

        max_steps  = override_max_steps or base_max_steps
        step_limit = max_steps + 5   # hard stop margin

        # ── Memory ───────────────────────────────────────────────────────────
        max_memory_chunks = 3 if tier == ModelTier.SLM else 8
        memory_summary_mode = "compact" if tier == ModelTier.SLM else "full"

        # ── Skills ───────────────────────────────────────────────────────────
        max_skill_steps = 4 if tier == ModelTier.SLM else 8
        skill_adapt_depth = "shallow" if tier == ModelTier.SLM else "deep"

        # ── Planning ─────────────────────────────────────────────────────────
        if tier == ModelTier.SLM:
            planning_depth = "minimal"
            max_plan_steps = 4
            use_subagents  = False
        else:
            planning_depth = {
                "simple":   "minimal",
                "moderate": "standard",
                "complex":  "deep",
                "compound": "deep",
            }.get(intake.complexity.value, "standard")
            max_plan_steps = max_steps
            use_subagents  = intake.complexity.value == "compound"

        # ── Tools ────────────────────────────────────────────────────────────
        if override_allowed_tools:
            allowed_tools = override_allowed_tools
        elif tier == ModelTier.SLM:
            # SLM: core tools only, biased by task type
            base = _TASK_TOOL_BIAS.get(intake.task_type.value, _SLM_CORE_TOOLS)
            allowed_tools = [t for t in base if t in _SLM_CORE_TOOLS]
            if not allowed_tools:
                allowed_tools = _SLM_CORE_TOOLS
        else:
            allowed_tools = _TASK_TOOL_BIAS.get(
                intake.task_type.value, _LLM_FULL_TOOLS)

        max_tools_per_step = 2 if tier == ModelTier.SLM else 4

        return ModelRuntimeConfig(
            model=model,
            provider=provider,
            tier=tier,
            spec=spec,
            max_steps=max_steps,
            step_limit=step_limit,
            max_memory_chunks=max_memory_chunks,
            memory_summary_mode=memory_summary_mode,
            max_skill_steps=max_skill_steps,
            skill_adapt_depth=skill_adapt_depth,
            planning_depth=planning_depth,
            max_plan_steps=max_plan_steps,
            use_subagents=use_subagents,
            allowed_tools=allowed_tools,
            max_tools_per_step=max_tools_per_step,
            summarize_aggressively=(tier == ModelTier.SLM),
            compress_tool_results=True,
        )

    def describe(self, config: ModelRuntimeConfig) -> str:
        """Human-readable summary of routing decision."""
        return (
            f"[{config.tier.value.upper()}] {config.model} via {config.provider}\n"
            f"  context: {config.spec.context_tokens:,}tk | "
            f"max_steps: {config.max_steps} | "
            f"planning: {config.planning_depth}\n"
            f"  tools: {', '.join(config.allowed_tools[:6])}"
            f"{'...' if len(config.allowed_tools) > 6 else ''}\n"
            f"  memory: {config.max_memory_chunks} chunks ({config.memory_summary_mode}) | "
            f"subagents: {config.use_subagents}"
        )


# Module-level singleton
_router = ModelRouter()

def route(model: str, provider: str, intake: TaskIntake, **kwargs) -> ModelRuntimeConfig:
    return _router.route(model, provider, intake, **kwargs)

def lookup_spec(model: str) -> ModelSpec:
    return _router.lookup_spec(model)