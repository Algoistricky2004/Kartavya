"""
context_manager.py — The most critical module in the entire harness.
Engineered for small context windows (llama-3.1-8b-instant: 8192 tokens).
If this is correct, everything else works. If this breaks, the agent dies.

Design principles:
  1. Every byte is architecture — no wasted tokens
  2. Step-scoped tool loading — never inject tools you won't use this turn
  3. Rolling summarization — sliding window with summaries, not truncation
  4. RAG memory — retrieve relevant chunks, never dump full memory
  5. Tool result compression — extract signal, strip noise
  6. Hard budget enforcement — never exceed a zone's allocation
  7. Full observability — every token decision is logged + traceable
"""

import re
import time
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from enum import Enum

# ─────────────────────────────────────────────────────────────────────────────
# Token Estimation
# (Without the model's exact tokenizer, we use a calibrated estimator.
#  Llama uses BPE tokenization; English text averages ~4 chars/token.)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Conservative token estimator for Llama-based models.
    We over-estimate by ~15% to build in safety margin.
    Real tokenizer: use tiktoken cl100k_base as proxy if available.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return int(len(enc.encode(text)) * 1.05)  # 5% safety
    except ImportError:
        # Fallback: character-based with calibration
        char_count = len(text)
        # Code tends to tokenize differently
        code_chars = len(re.findall(r'[{}()\[\];=<>+\-*/\\|&^%$#@!~`]', text))
        base = char_count / 3.8  # Llama tokenizer is slightly more efficient
        code_penalty = code_chars * 0.1
        return int((base + code_penalty) * 1.15)  # 15% over-estimate


# ─────────────────────────────────────────────────────────────────────────────
# Context Budget Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextBudget:
    """
    Token budget for each zone. Tune per model.
    Zones are enforced strictly — exceeding a zone is a hard error, not a warning.
    """
    # Model config
    total_context: int = 8192
    max_output: int = 1024

    # Zone allocations (tokens)
    system_prompt: int = 384
    tool_definitions: int = 768   # per-step tools only
    active_memory: int = 512      # RAG-retrieved chunks
    task_context: int = 256       # current step + plan position
    conversation_history: int = 2048
    current_input: int = 1024     # tool results + user message
    safety_buffer: int = 256      # tokenizer inaccuracy buffer

    @property
    def total_input_budget(self) -> int:
        return (self.system_prompt + self.tool_definitions + self.active_memory +
                self.task_context + self.conversation_history + self.current_input +
                self.safety_buffer)

    @property
    def utilization_pct(self) -> float:
        return self.total_input_budget / (self.total_context - self.max_output) * 100

    def validate(self):
        used = self.total_input_budget + self.max_output
        assert used <= self.total_context, (
            f"Budget overflow: {used} > {self.total_context}. "
            f"Reduce zone allocations."
        )

# Model-specific presets
BUDGET_PRESETS = {
    "llama-3.1-8b-instant":  ContextBudget(total_context=8192,   max_output=1024),
    "llama-3.1-70b-versatile": ContextBudget(total_context=32768, max_output=2048,
                                             system_prompt=800, tool_definitions=1500,
                                             active_memory=1024, task_context=512,
                                             conversation_history=8192, current_input=2048,
                                             safety_buffer=512),
    "llama-3.3-70b-versatile": ContextBudget(total_context=32768, max_output=2048,
                                             system_prompt=800, tool_definitions=1500,
                                             active_memory=1024, task_context=512,
                                             conversation_history=8192, current_input=2048,
                                             safety_buffer=512),
    "mixtral-8x7b-32768":    ContextBudget(total_context=32768, max_output=2048,
                                            system_prompt=800, tool_definitions=1500,
                                            active_memory=1024, task_context=512,
                                            conversation_history=8192, current_input=2048,
                                            safety_buffer=512),
    "gpt-4o-mini":           ContextBudget(total_context=128000, max_output=4096,
                                            system_prompt=1024, tool_definitions=2048,
                                            active_memory=2048, task_context=512,
                                            conversation_history=32768, current_input=4096,
                                            safety_buffer=1024),
    "claude-3-haiku-20240307": ContextBudget(total_context=200000, max_output=4096,
                                              system_prompt=1024, tool_definitions=2048,
                                              active_memory=2048, task_context=512,
                                              conversation_history=65536, current_input=4096,
                                              safety_buffer=1024),
}


# ─────────────────────────────────────────────────────────────────────────────
# Message types
# ─────────────────────────────────────────────────────────────────────────────

class Role(str, Enum):
    SYSTEM = "system"
    USER   = "user"
    ASSISTANT = "assistant"

@dataclass
class Turn:
    role: Role
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    is_summary: bool = False

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = estimate_tokens(self.content or "")

    def to_api_message(self) -> dict:
        return {"role": self.role.value, "content": self.content}

@dataclass
class ContextSnapshot:
    """What was actually assembled for a given LLM call — full audit trail."""
    call_id: str
    timestamp: float
    model: str
    zones: dict[str, int]      # zone_name → tokens used
    total_tokens: int
    headroom_tokens: int
    messages: list[dict]
    truncations: list[str]     # what was truncated and why
    summaries_used: int        # how many summaries replaced raw turns
    memory_chunks_injected: int
    tools_loaded: list[str]
    budget: dict


# ─────────────────────────────────────────────────────────────────────────────
# Tool Result Compressor
# ─────────────────────────────────────────────────────────────────────────────

class ToolResultCompressor:
    """
    Compress tool results before injecting into context.
    This is the single biggest lever for keeping small models functional.
    """

    MAX_BASH_OUTPUT = 500     # chars
    MAX_FILE_LINES  = 40      # lines
    MAX_JSON_DEPTH  = 2       # levels
    MAX_ERROR_LINES = 3

    def compress(self, tool_name: str, raw_result: str, task_hint: str = "") -> tuple[str, str]:
        """
        Returns (compressed_result, compression_note).
        compression_note describes what was removed (for observability).
        """
        if not raw_result:
            return "", ""

        original_tokens = estimate_tokens(raw_result)
        result = raw_result
        notes = []

        if tool_name in ("bash", "exec", "shell", "terminal"):
            result, n = self._compress_bash(raw_result)
            notes.extend(n)

        elif tool_name in ("read_file", "read", "cat"):
            result, n = self._compress_file(raw_result, task_hint)
            notes.extend(n)

        elif tool_name in ("search", "grep", "find"):
            result, n = self._compress_search(raw_result)
            notes.extend(n)

        elif tool_name in ("web_fetch", "web_search"):
            result, n = self._compress_web(raw_result)
            notes.extend(n)

        else:
            result, n = self._compress_generic(raw_result)
            notes.extend(n)

        compressed_tokens = estimate_tokens(result)
        reduction_pct = (1 - compressed_tokens / original_tokens) * 100 if original_tokens > 0 else 0

        note = (
            f"[compressed {tool_name}: {original_tokens}→{compressed_tokens}tk "
            f"(-{reduction_pct:.0f}%)" + (f" | {'; '.join(notes)}" if notes else "") + "]"
            if notes or reduction_pct > 5 else ""
        )

        return result, note

    def _compress_bash(self, output: str) -> tuple[str, list[str]]:
        lines = output.strip().split("\n")
        notes = []
        if len(output) <= self.MAX_BASH_OUTPUT:
            return output, notes
        # Keep head + tail with separator
        head_lines = lines[:10]
        tail_lines = lines[-5:]
        skipped = len(lines) - 15
        result = "\n".join(head_lines)
        if skipped > 0:
            result += f"\n... [{skipped} lines omitted] ...\n"
            result += "\n".join(tail_lines)
            notes.append(f"bash output truncated ({skipped} lines omitted)")
        return result, notes

    def _compress_file(self, content: str, task_hint: str) -> tuple[str, list[str]]:
        lines = content.split("\n")
        notes = []
        if len(lines) <= self.MAX_FILE_LINES:
            return content, notes
        # If task_hint has a line number, extract ±20 lines around it
        line_match = re.search(r'line[: ]+(\d+)', task_hint, re.I)
        if line_match:
            target = int(line_match.group(1)) - 1
            start = max(0, target - 20)
            end   = min(len(lines), target + 20)
            result = f"[Lines {start+1}-{end+1} of {len(lines)} total]\n"
            result += "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[start:end], start))
            notes.append(f"file windowed to lines {start+1}-{end+1}")
            return result, notes
        # Otherwise: head + tail
        head = lines[:20]
        tail = lines[-10:]
        skipped = len(lines) - 30
        result = "\n".join(head)
        if skipped > 0:
            result += f"\n... [{skipped} lines omitted] ...\n"
            result += "\n".join(tail)
            notes.append(f"file truncated ({skipped} lines omitted)")
        return result, notes

    def _compress_search(self, output: str) -> tuple[str, list[str]]:
        lines = output.strip().split("\n")
        notes = []
        if len(lines) <= 30:
            return output, notes
        # Keep first 30 matches
        result = "\n".join(lines[:30])
        result += f"\n... [{len(lines)-30} more matches omitted]"
        notes.append(f"search results capped at 30 of {len(lines)}")
        return result, notes

    def _compress_web(self, output: str) -> tuple[str, list[str]]:
        # Web content: keep first 1000 chars
        notes = []
        if len(output) <= 1000:
            return output, notes
        result = output[:1000] + "... [truncated]"
        notes.append("web content truncated to 1000 chars")
        return result, notes

    def _compress_generic(self, output: str) -> tuple[str, list[str]]:
        notes = []
        # Try JSON compression
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                keys = list(parsed.keys())
                if len(keys) > 10:
                    summary = {k: parsed[k] for k in keys[:10]}
                    summary["..."] = f"[{len(keys)-10} more keys]"
                    result = json.dumps(summary, indent=2)
                    notes.append(f"JSON compressed from {len(keys)} to 10 keys")
                    return result, notes
        except (json.JSONDecodeError, TypeError):
            pass
        # Plain text: truncate
        if len(output) > 800:
            result = output[:800] + "... [truncated]"
            notes.append("output truncated to 800 chars")
            return result, notes
        return output, notes


# ─────────────────────────────────────────────────────────────────────────────
# Conversation History Manager
# ─────────────────────────────────────────────────────────────────────────────

class ConversationHistoryManager:
    """
    Manages sliding window of conversation turns.
    Older turns are replaced with 1-sentence summaries.

    Strategy:
    - Always keep last KEEP_VERBATIM turns verbatim
    - Turns older than that: compress to 1-sentence summary
    - Total history budget is hard-capped
    """

    KEEP_VERBATIM = 4         # how many recent turns to keep in full
    SUMMARY_MAX_CHARS = 120   # max chars per summary

    def __init__(self, budget_tokens: int):
        self.budget = budget_tokens
        self.turns: list[Turn] = []
        self.summaries: list[str] = []  # summaries of old turns

    def add_turn(self, turn: Turn):
        self.turns.append(turn)
        # Summarize oldest turns when we exceed verbatim limit
        while len(self.turns) > self.KEEP_VERBATIM:
            old = self.turns.pop(0)
            summary = self._summarize_turn(old)
            self.summaries.append(summary)

    def _summarize_turn(self, turn: Turn) -> str:
        """One-line summary of a turn. Used by LLM in production; simple here."""
        content = (turn.content or "")[:self.SUMMARY_MAX_CHARS]
        role_tag = "Human" if turn.role == Role.USER else "Agent"
        if turn.tool_name:
            return f"[{role_tag} called {turn.tool_name}: {content[:60]}]"
        return f"[{role_tag}: {content[:80]}...]"

    def build_messages(self) -> tuple[list[dict], int, int]:
        """
        Build messages list within budget.
        Returns (messages, tokens_used, summaries_used).
        """
        messages = []
        tokens_used = 0
        summaries_used = 0

        # First: add compressed summaries of old turns
        if self.summaries:
            recent_summaries = self.summaries[-20:]  # keep last 20 summaries
            summary_text = " | ".join(recent_summaries)
            summary_tokens = estimate_tokens(summary_text)
            # Budget: use at most 30% of history budget for summaries
            max_summary_tokens = int(self.budget * 0.3)
            if summary_tokens > max_summary_tokens:
                # Trim from front
                while recent_summaries and estimate_tokens(" | ".join(recent_summaries)) > max_summary_tokens:
                    recent_summaries.pop(0)
                summary_text = " | ".join(recent_summaries)
                summary_tokens = estimate_tokens(summary_text)
            if summary_text:
                messages.append({
                    "role": "user",
                    "content": f"[Earlier conversation summary: {summary_text}]"
                })
                messages.append({
                    "role": "assistant",
                    "content": "Understood, continuing from where we left off."
                })
                tokens_used += summary_tokens + 10
                summaries_used = len(recent_summaries)

        # Then: add verbatim turns from newest to oldest (reverse, then flip)
        remaining = self.budget - tokens_used
        turns_to_add = []
        for turn in reversed(self.turns):
            t = estimate_tokens(turn.content or "")
            if t > remaining:
                break
            turns_to_add.insert(0, turn.to_api_message())
            remaining -= t
            tokens_used += t

        messages.extend(turns_to_add)
        return messages, tokens_used, summaries_used


# ─────────────────────────────────────────────────────────────────────────────
# Main Context Manager
# ─────────────────────────────────────────────────────────────────────────────

class ContextManager:
    """
    The central context assembler. Called before every LLM API call.

    Usage:
        ctx = ContextManager(model="llama-3.1-8b-instant")
        messages, snapshot = ctx.build(
            system_prompt=...,
            task_context=...,
            memory_chunks=...,
            tool_definitions=...,
            current_input=...,
            step_tools=["bash", "read_file"],  # only tools for this step
        )
        response = call_llm(messages)
        ctx.record_turn(user_msg, assistant_response)
    """

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model  = model
        self.budget = BUDGET_PRESETS.get(model, ContextBudget())
        self.budget.validate()
        self.history = ConversationHistoryManager(self.budget.conversation_history)
        self.compressor = ToolResultCompressor()
        self._call_count = 0
        self._snapshots: list[ContextSnapshot] = []

    def build(
        self,
        system_prompt: str,
        current_input: str,
        task_context: str         = "",
        memory_chunks: list[str]  = None,
        tool_definitions: list[dict] = None,
        tool_results: list[tuple[str, str]] = None,  # [(tool_name, raw_result)]
        step_hint: str            = "",
    ) -> tuple[list[dict], ContextSnapshot]:
        """
        Assemble the full message list for one LLM call.
        Returns (messages, snapshot) — snapshot is full observability audit.
        """
        import uuid
        self._call_count += 1
        call_id = f"ctx_{self._call_count}_{uuid.uuid4().hex[:6]}"
        truncations = []
        zones = {}

        # ── Zone 1: System Prompt ─────────────────────────────────────────
        sys_text = self._fit_zone(system_prompt, self.budget.system_prompt,
                                   "system_prompt", truncations)
        zones["system_prompt"] = estimate_tokens(sys_text)

        # ── Zone 2: Tool Definitions (step-scoped) ────────────────────────
        tool_text = ""
        tools_loaded = []
        if tool_definitions:
            tool_text, tools_loaded = self._build_tool_defs(
                tool_definitions, self.budget.tool_definitions, truncations
            )
        zones["tool_definitions"] = estimate_tokens(tool_text)

        # ── Zone 3: Active Memory (RAG chunks) ───────────────────────────
        mem_text = ""
        mem_chunks_injected = 0
        if memory_chunks:
            mem_text, mem_chunks_injected = self._build_memory(
                memory_chunks, self.budget.active_memory, truncations
            )
        zones["active_memory"] = estimate_tokens(mem_text)

        # ── Zone 4: Task Context ──────────────────────────────────────────
        task_text = self._fit_zone(task_context, self.budget.task_context,
                                    "task_context", truncations)
        zones["task_context"] = estimate_tokens(task_text)

        # ── Zone 5: Compress tool results BEFORE adding to current input ──
        compressed_results = []
        if tool_results:
            for tool_name, raw in tool_results:
                compressed, note = self.compressor.compress(tool_name, raw, step_hint)
                compressed_results.append(compressed)
                if note:
                    truncations.append(note)

        # ── Zone 6: Current Input ─────────────────────────────────────────
        full_input = current_input
        if compressed_results:
            results_block = "\n\n".join(
                f"[Tool result]\n{r}" for r in compressed_results if r
            )
            full_input = f"{results_block}\n\n{current_input}" if current_input else results_block

        current_text = self._fit_zone(full_input, self.budget.current_input,
                                       "current_input", truncations)
        zones["current_input"] = estimate_tokens(current_text)

        # ── Zone 7: Conversation History ──────────────────────────────────
        hist_messages, hist_tokens, summaries_used = self.history.build_messages()
        zones["conversation_history"] = hist_tokens

        # ── Assemble system prompt with injections ────────────────────────
        full_system = sys_text
        if tool_text:
            full_system += f"\n\n## Available Tools\n{tool_text}"
        if mem_text:
            full_system += f"\n\n## Relevant Memory\n{mem_text}"
        if task_text:
            full_system += f"\n\n## Current Task Context\n{task_text}"

        # Build final message list
        messages = [{"role": "system", "content": full_system}]
        messages.extend(hist_messages)
        messages.append({"role": "user", "content": current_text})

        # ── Safety check ──────────────────────────────────────────────────
        total = sum(zones.values())
        headroom = (self.budget.total_context - self.budget.max_output
                    - self.budget.safety_buffer - total)

        if headroom < 0:
            # Emergency: trim history further
            truncations.append(
                f"EMERGENCY TRIM: headroom={headroom}, trimming history by 30%"
            )
            self.history.budget = int(self.history.budget * 0.7)
            hist_messages, hist_tokens, summaries_used = self.history.build_messages()
            zones["conversation_history"] = hist_tokens
            # Rebuild without history
            messages = [{"role": "system", "content": full_system}]
            messages.extend(hist_messages)
            messages.append({"role": "user", "content": current_text})
            total = sum(zones.values())
            headroom = (self.budget.total_context - self.budget.max_output
                        - self.budget.safety_buffer - total)

        snapshot = ContextSnapshot(
            call_id=call_id,
            timestamp=time.time(),
            model=self.model,
            zones=zones,
            total_tokens=total,
            headroom_tokens=headroom,
            messages=messages,
            truncations=truncations,
            summaries_used=summaries_used,
            memory_chunks_injected=mem_chunks_injected,
            tools_loaded=tools_loaded,
            budget=asdict(self.budget),
        )
        self._snapshots.append(snapshot)
        return messages, snapshot

    def record_turn(self, user_content: str, assistant_content: str,
                    tool_name: str = None, tool_args: dict = None):
        """Call after each LLM response to update history."""
        self.history.add_turn(Turn(role=Role.USER, content=user_content))
        self.history.add_turn(Turn(
            role=Role.ASSISTANT, content=assistant_content,
            tool_name=tool_name, tool_args=tool_args
        ))

    def get_latest_snapshot(self) -> Optional[ContextSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def get_stats(self) -> dict:
        if not self._snapshots:
            return {}
        recent = self._snapshots[-10:]
        avg_tokens = sum(s.total_tokens for s in recent) / len(recent)
        avg_headroom = sum(s.headroom_tokens for s in recent) / len(recent)
        total_truncations = sum(len(s.truncations) for s in recent)
        return {
            "total_calls": self._call_count,
            "avg_tokens_per_call": round(avg_tokens),
            "avg_headroom_tokens": round(avg_headroom),
            "total_truncations_last_10": total_truncations,
            "model": self.model,
            "budget_utilization_pct": round(self.budget.utilization_pct, 1),
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _fit_zone(self, text: str, budget: int, zone_name: str,
                   truncations: list[str]) -> str:
        """Fit text into zone budget. Truncate at sentence/line boundaries."""
        if not text:
            return ""
        tokens = estimate_tokens(text)
        if tokens <= budget:
            return text
        # Truncate at char level (approximate): budget * 4 chars
        max_chars = budget * 4
        truncated = text[:max_chars]
        # Try to end at sentence boundary
        last_period = truncated.rfind(". ")
        if last_period > max_chars * 0.7:
            truncated = truncated[:last_period + 1]
        truncations.append(
            f"{zone_name}: {tokens}→~{budget}tk (truncated {tokens-budget}tk)"
        )
        return truncated + " [truncated]"

    def _build_tool_defs(self, tools: list[dict], budget: int,
                          truncations: list[str]) -> tuple[str, list[str]]:
        """
        Build compressed tool definitions within budget.
        Uses terse format instead of full JSON schema.
        """
        lines = []
        loaded = []
        tokens_used = 0
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")[:100]
            params = tool.get("parameters", {}).get("properties", {})
            param_summary = ", ".join(
                f"{k}({'req' if k in (tool.get('parameters',{}).get('required',[]) or []) else 'opt'})"
                for k in list(params.keys())[:6]
            )
            line = f"- {name}({param_summary}): {desc}"
            t = estimate_tokens(line)
            if tokens_used + t > budget:
                truncations.append(f"tool_defs: stopped at {len(loaded)}/{len(tools)} tools (budget)")
                break
            lines.append(line)
            loaded.append(name)
            tokens_used += t
        return "\n".join(lines), loaded

    def _build_memory(self, chunks: list[str], budget: int,
                       truncations: list[str]) -> tuple[str, int]:
        """Inject memory chunks within budget. Highest-priority chunks first."""
        lines = []
        tokens_used = 0
        injected = 0
        for chunk in chunks:
            t = estimate_tokens(chunk)
            if tokens_used + t > budget:
                truncations.append(
                    f"memory: {injected}/{len(chunks)} chunks injected (budget full)"
                )
                break
            lines.append(f"• {chunk}")
            tokens_used += t
            injected += 1
        return "\n".join(lines), injected
