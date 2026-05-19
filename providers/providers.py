"""
providers.py — Multi-provider LLM abstraction.
Primary: Groq (llama-3.1-8b-instant)
Optional: OpenAI, Anthropic (drop-in swap)

For small context windows:
  - Streaming by default (faster to first token)
  - Automatic retry with exponential backoff
  - Token counting on every response
  - Provider-specific error handling
"""

import time
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class LLMResponse:
    content: str
    tokens_in: int
    tokens_out: int
    model: str
    provider: str
    duration_ms: float
    finish_reason: str      # "stop", "length", "tool_calls"
    tool_calls: list[dict]  # parsed tool calls if any
    raw: dict               # raw API response for debugging


class LLMProvider(ABC):
    """Abstract provider. All providers speak the same interface."""

    @abstractmethod
    def complete(self, messages: list[dict], model: str,
                 max_tokens: int = 1024, temperature: float = 0.7,
                 stream: bool = True, tools: list[dict] = None) -> LLMResponse:
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        ...

    def complete_with_retry(self, messages: list[dict], model: str,
                             max_retries: int = 3, **kwargs) -> LLMResponse:
        """Retry with exponential backoff on rate limit / server errors."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.complete(messages, model, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Retryable: rate limit, server error
                if any(x in error_str for x in ("rate_limit", "429", "503", "500")):
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    print(f"[provider] retry {attempt+1}/{max_retries} in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise  # Non-retryable: auth error, invalid request
        raise last_error


class GroqProvider(LLMProvider):
    """
    Groq provider using llama-3.1-8b-instant.
    Fast inference, great for agentic loops.
    """

    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: str = None):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("pip install groq")
        import os
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY not set")
        self._client = Groq(api_key=key)

    def complete(self, messages: list[dict], model: str = None,
                 max_tokens: int = 1024, temperature: float = 0.7,
                 stream: bool = True, tools: list[dict] = None) -> LLMResponse:
        model = model or self.DEFAULT_MODEL
        start = time.time()

        # Count input tokens (estimate)
        input_text = " ".join(m.get("content", "") or "" for m in messages)
        tokens_in_est = max(1, len(input_text) // 4)

        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        # Groq tool calling (when model supports it)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if stream:
            return self._stream_complete(kwargs, model, tokens_in_est, start)
        else:
            return self._batch_complete(kwargs, model, tokens_in_est, start)

    def _stream_complete(self, kwargs: dict, model: str,
                          tokens_in_est: int, start: float) -> LLMResponse:
        completion = self._client.chat.completions.create(**kwargs)
        chunks = []
        tokens_out = 0
        finish_reason = "stop"

        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                chunks.append(delta.content)
                tokens_out += len(delta.content) // 4
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        content = "".join(chunks)
        duration_ms = (time.time() - start) * 1000

        # Try to get real usage from last chunk
        try:
            usage = chunk.usage
            if usage:
                tokens_in_est = usage.prompt_tokens or tokens_in_est
                tokens_out = usage.completion_tokens or tokens_out
        except Exception:
            pass

        # Parse tool calls from content if model uses text-based tool calling
        tool_calls = self._parse_tool_calls(content)

        return LLMResponse(
            content=content,
            tokens_in=tokens_in_est,
            tokens_out=tokens_out,
            model=model,
            provider="groq",
            duration_ms=round(duration_ms, 1),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw={"chunks": len(chunks)},
        )

    def _batch_complete(self, kwargs: dict, model: str,
                         tokens_in_est: int, start: float) -> LLMResponse:
        kwargs_copy = {**kwargs, "stream": False}
        response = self._client.chat.completions.create(**kwargs_copy)
        content = response.choices[0].message.content or ""
        duration_ms = (time.time() - start) * 1000

        usage = response.usage
        tokens_in = (usage.prompt_tokens if usage else tokens_in_est)
        tokens_out = (usage.completion_tokens if usage else max(1, len(content) // 4))
        finish_reason = response.choices[0].finish_reason or "stop"

        # Parse native tool calls if any
        tool_calls = []
        msg = response.choices[0].message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments or "{}"),
                })
        # Fallback: parse from text
        if not tool_calls:
            tool_calls = self._parse_tool_calls(content)

        return LLMResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            provider="groq",
            duration_ms=round(duration_ms, 1),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw={"finish_reason": finish_reason},
        )

    def _parse_tool_calls(self, content: str) -> list[dict]:
        """
        Parse tool calls from text responses.
        Models without native tool calling emit JSON tool blocks.
        We use a structured format: <tool>name</tool><args>{...}</args>
        """
        tool_calls = []
        # Pattern 1: XML-style tool blocks
        pattern = r'<tool>(.*?)</tool>\s*<args>(.*?)</args>'
        for match in re.finditer(pattern, content, re.DOTALL):
            tool_name = match.group(1).strip()
            args_raw  = match.group(2).strip()
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"raw": args_raw}
            tool_calls.append({"name": tool_name, "args": args})

        # Pattern 2: JSON function call block
        if not tool_calls:
            json_pattern = r'```(?:json)?\s*\{"tool":\s*"([^"]+)",\s*"args":\s*(\{[^`]+\})\s*\}\s*```'
            for match in re.finditer(json_pattern, content, re.DOTALL):
                tool_name = match.group(1)
                try:
                    args = json.loads(match.group(2))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"name": tool_name, "args": args})

        return tool_calls

    def list_models(self) -> list[str]:
        return [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama-3.3-70b-versatile",
            "llama-3.2-90b-vision-preview",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]


class OpenAIProvider(LLMProvider):
    """OpenAI provider — drop-in swap for Groq."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str = None):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")
        import os
        self._openai = openai
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def complete(self, messages: list[dict], model: str = None,
                 max_tokens: int = 1024, temperature: float = 0.7,
                 stream: bool = False, tools: list[dict] = None) -> LLMResponse:
        import openai
        model = model or self.DEFAULT_MODEL
        start = time.time()

        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = openai.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        duration_ms = (time.time() - start) * 1000
        usage = response.usage

        tool_calls = []
        msg = response.choices[0].message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments or "{}"),
                })

        return LLMResponse(
            content=content,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            model=model,
            provider="openai",
            duration_ms=round(duration_ms, 1),
            finish_reason=response.choices[0].finish_reason,
            tool_calls=tool_calls,
            raw={},
        )

    def list_models(self) -> list[str]:
        return ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]


class AnthropicProvider(LLMProvider):
    """Anthropic provider — Claude models."""

    DEFAULT_MODEL = "claude-3-haiku-20240307"

    def __init__(self, api_key: str = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        import os
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )

    def complete(self, messages: list[dict], model: str = None,
                 max_tokens: int = 1024, temperature: float = 0.7,
                 stream: bool = False, tools: list[dict] = None) -> LLMResponse:
        model = model or self.DEFAULT_MODEL
        start = time.time()

        # Anthropic uses system prompt separately
        system = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                api_messages.append(m)

        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=api_messages,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system
        if tools:
            # Convert to Anthropic tool format
            kwargs["tools"] = [
                {"name": t["name"], "description": t.get("description",""),
                 "input_schema": t.get("parameters", {"type":"object","properties":{}})}
                for t in tools
            ]

        response = self._client.messages.create(**kwargs)
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "args": block.input})

        duration_ms = (time.time() - start) * 1000
        return LLMResponse(
            content=content,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            model=model,
            provider="anthropic",
            duration_ms=round(duration_ms, 1),
            finish_reason=response.stop_reason or "stop",
            tool_calls=tool_calls,
            raw={},
        )

    def list_models(self) -> list[str]:
        return [
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-5",
        ]


# ── Provider factory ──────────────────────────────────────────────────────────

def get_provider(name: str, api_key: str = None) -> LLMProvider:
    providers = {
        "groq":      GroqProvider,
        "openai":    OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Choose from {list(providers.keys())}")
    return providers[name](api_key=api_key)
