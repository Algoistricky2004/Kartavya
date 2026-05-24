"""
providers.py — Multi-provider LLM abstraction.

Providers: Groq, Ollama, OpenAI, Anthropic

KEY FIXES vs previous:
  - OllamaProvider: token counts now actually captured (prompt_eval_count / eval_count)
  - OllamaProvider: stream=False forced (streaming breaks token capture on 3b models)
  - OllamaProvider: native tool calling disabled for tiny models — they don't use it reliably;
    ReAct text format (Thought/Bash/Done) is used instead and parsed here
  - parse_tool_calls: extended to parse ReAct format (Bash: <cmd>, Done: <answer>)
  - GroqProvider: streaming delta.tool_calls properly accumulated
"""

import time, json, re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    content:       str
    tokens_in:     int
    tokens_out:    int
    model:         str
    provider:      str
    duration_ms:   float
    finish_reason: str
    tool_calls:    list   # [{"name": str, "args": dict}]
    raw:           dict


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages, model, max_tokens=1024, temperature=0.7,
                 stream=True, tools=None) -> LLMResponse: ...

    @abstractmethod
    def list_models(self) -> list[str]: ...

    def complete_with_retry(self, messages, model, max_retries=3, **kwargs) -> LLMResponse:
        """
        Retry wrapper with exponential backoff for rate limits (429) and transient errors.
        Groq SDK's own retry is disabled (max_retries=0) so we fully control timing here.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.complete(messages, model, **kwargs)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # Retryable: rate limit, server errors, connection issues
                if any(x in err_str for x in ("rate_limit", "429", "503", "500",
                                               "connection", "timeout", "overloaded")):
                    wait = min(30, 2 ** attempt + 1)  # cap at 30s
                    print(f"   ⟳ retry {attempt+1}/{max_retries} in {wait}s ({type(e).__name__})")
                    time.sleep(wait)
                else:
                    # Non-retryable: auth errors, bad requests, etc.
                    raise
        raise last_error


# ─────────────────────────────────────────────────────────────────────────────
# Tool call parser — handles XML format + ReAct format
# ─────────────────────────────────────────────────────────────────────────────

def parse_tool_calls(content: str) -> list[dict]:
    """
    Parse tool calls from model text output.

    Priority: Done/Answer > Bash block > Bash inline > XML > JSON

    Critical fix for llama3.2:3b: it writes "Bash:\n  Done: answer" or
    "Bash:\n  Done:" (empty). Both must resolve to task_done, not bash.
    """
    c = content.strip()

    # ── PRIORITY 1: Done/Answer signal — always checked first ────────────────
    done_m = re.search(
        r'(?:^|\n)\s*(?:Done|Answer|Final|Result)\s*:\s*(.*)',
        c, re.IGNORECASE | re.DOTALL
    )
    if done_m:
        answer = done_m.group(1).strip()
        # Empty Done: → extract Thought content as the answer
        if not answer:
            thought_m = re.search(r'(?:Thought|Observation)\s*:\s*(.+)',
                                   c[:done_m.start()], re.DOTALL | re.IGNORECASE)
            before = c[:done_m.start()].strip()
            answer = thought_m.group(1).strip() if thought_m else before
        # Reject if answer is another ReAct token
        if answer and not re.match(r'^(Bash|Action|Command|Thought)\s*:', answer, re.I):
            # Trim trailing ReAct lines that got captured by DOTALL
            answer = re.split(r'\n(?:Bash|Action|Thought|Command)\s*:', answer)[0].strip()
            if answer:
                return [{"name": "task_done",
                         "args": {"output": answer, "summary": answer[:120]}}]

    # ── Format 1: Bash code block ─────────────────────────────────────────────
    m = re.search(
        r'(?:Bash|Action|Command)\s*:\s*\n```(?:bash|shell|sh|cmd)?\s*\n(.*?)\n```',
        c, re.DOTALL | re.IGNORECASE
    )
    if m:
        cmd = m.group(1).strip()
        if cmd and not re.match(r'^(Done|Answer|Final|Thought|Result)\s*:', cmd, re.I):
            return [{"name": "bash", "args": {"command": cmd}}]

    # ── Format 2: Bash inline (single line) ───────────────────────────────────
    m = re.search(r'^(?:Bash|Action|Command)\s*:\s*(.+)$', c,
                  re.MULTILINE | re.IGNORECASE)
    if m:
        cmd = m.group(1).strip().strip('`')
        if (cmd and not cmd.startswith('{')
                and not re.match(r'^(Done|Answer|Final|Result|Thought)\s*:', cmd, re.I)):
            return [{"name": "bash", "args": {"command": cmd}}]

    # ── Format 3: XML <tool>/<args> ───────────────────────────────────────────
    m = re.search(
        r'<tool>\s*([\w_]+)\s*</tool>\s*(?:<args>)?\s*(\{.*?\})\s*(?:</args>)?',
        c, re.DOTALL
    )
    if m:
        return [{"name": m.group(1).strip(), "args": _safe_json(m.group(2))}]

    # ── Format 4: Raw JSON tool call ──────────────────────────────────────────
    m = re.search(
        r'\{"name"\s*:\s*"([\w_]+)"\s*,\s*"(?:args|arguments|parameters)"\s*:\s*(\{.*?\})\s*\}',
        c, re.DOTALL
    )
    if m:
        return [{"name": m.group(1), "args": _safe_json(m.group(2))}]

    return []


def _safe_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = raw.replace("'", '"')
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        try:
            return json.loads(fixed)
        except Exception:
            return {"_parse_error": raw[:100]}


# ─────────────────────────────────────────────────────────────────────────────
# Groq Provider
# ─────────────────────────────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: str = None):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("pip install groq")
        import os
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY not set.")
        # max_retries=0 disables Groq SDK's own retry/sleep loop.
        # Without this, a 429 causes Groq to sleep internally for up to 60s,
        # which feels like a hang. We handle retries ourselves in complete_with_retry().
        self._client = Groq(api_key=key, max_retries=0)

    def complete(self, messages, model=None, max_tokens=1024, temperature=0.7,
                 stream=True, tools=None) -> LLMResponse:
        model = model or self.DEFAULT_MODEL
        start = time.time()

        kwargs = dict(model=model, messages=messages,
                      max_tokens=max_tokens, temperature=temperature)
        if tools:
            kwargs["tools"]       = _to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        content    = ""
        tool_calls = []

        if stream:
            resp = self._client.chat.completions.create(**kwargs, stream=True)
            tc_acc = {}   # id → {name, args_str}
            for chunk in resp:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue
                if delta.content:
                    content += delta.content
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for dtc in delta.tool_calls:
                        idx = dtc.index
                        if idx not in tc_acc:
                            tc_acc[idx] = {"name": "", "args_str": ""}
                        if dtc.function.name:
                            tc_acc[idx]["name"] = dtc.function.name
                        if dtc.function.arguments:
                            tc_acc[idx]["args_str"] += dtc.function.arguments
            for _, tc in sorted(tc_acc.items()):
                if tc["name"]:
                    tool_calls.append({"name": tc["name"],
                                       "args": _safe_json(tc["args_str"])})
            # token counts from last chunk
            tokens_in  = 0
            tokens_out = 0
        else:
            resp    = self._client.chat.completions.create(**kwargs)
            msg     = resp.choices[0].message
            content = msg.content or ""
            usage   = resp.usage
            tokens_in  = usage.prompt_tokens     if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc.function.name,
                        "args": _safe_json(tc.function.arguments or "{}"),
                    })

        # Fallback text parser
        if not tool_calls and content:
            tool_calls = parse_tool_calls(content)

        # Token fallback for streaming
        if tokens_in == 0 and stream:
            tokens_in  = max(1, sum(len(str(m.get("content","") or "")) for m in messages) // 4)
            tokens_out = max(1, len(content) // 4)

        return LLMResponse(
            content=content, tokens_in=tokens_in, tokens_out=tokens_out,
            model=model, provider="groq",
            duration_ms=round((time.time()-start)*1000, 1),
            finish_reason="stop", tool_calls=tool_calls, raw={},
        )

    def list_models(self):
        return ["llama-3.1-8b-instant", "llama-3.1-70b-versatile",
                "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Provider  ← main fix
# ─────────────────────────────────────────────────────────────────────────────

# Models that should NOT use native Ollama tool calling.
# These models respond in plain text regardless of the tools= parameter.
# We use ReAct (Thought/Bash/Done) format for them instead.
REACT_ONLY_MODELS = {
    "llama3.2:3b", "llama3.2:1b", "llama3.1:8b",
    "gemma2:2b", "gemma:2b", "phi3:mini", "phi3.5:mini",
    "qwen2:1.5b", "mistral:7b", "mistral-nemo",
}


class OllamaProvider(LLMProvider):
    DEFAULT_MODEL = "llama3.2:3b"
    DEFAULT_HOST  = "http://localhost:11434"

    def __init__(self, host: str = None):
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError:
            raise ImportError("pip install ollama  &&  ollama pull llama3.2:3b")
        self._host = host or self.DEFAULT_HOST

    def _client(self):
        return self._ollama.Client(host=self._host)

    def complete(self, messages, model=None, max_tokens=512, temperature=0.1,
                 stream=False, tools=None) -> LLMResponse:
        model = model or self.DEFAULT_MODEL
        start = time.time()

        # Always non-streaming for Ollama — streaming doesn't return token counts
        # reliably on small models, and the speed gain is negligible locally.
        stream = False

        # Convert messages
        api_msgs = [{"role": m["role"], "content": m.get("content", "") or ""}
                    for m in messages]

        use_native_tools = (tools and model not in REACT_ONLY_MODELS)

        kwargs = dict(
            model=model,
            messages=api_msgs,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        if use_native_tools:
            kwargs["tools"] = _to_openai_tools(tools)

        try:
            client   = self._client()
            response = client.chat(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Ollama error: {e}\n"
                f"  → Make sure ollama is running: ollama serve\n"
                f"  → Model pulled: ollama pull {model}"
            )

        duration_ms = (time.time() - start) * 1000

        # ── Extract content ───────────────────────────────────────────────
        content = ""
        try:
            content = response.message.content or ""
        except AttributeError:
            try:
                content = response["message"]["content"] or ""
            except (KeyError, TypeError):
                content = str(response)

        # ── Token counts — Ollama puts these on the response root ─────────
        # Fields: prompt_eval_count (input), eval_count (output)
        tokens_in  = 0
        tokens_out = 0
        for attr in ("prompt_eval_count",):
            v = getattr(response, attr, None)
            if v is not None:
                tokens_in = int(v)
                break
        for attr in ("eval_count",):
            v = getattr(response, attr, None)
            if v is not None:
                tokens_out = int(v)
                break
        # If still 0, estimate from content length
        if tokens_in == 0:
            tokens_in  = max(1, sum(len(m.get("content","") or "") for m in api_msgs) // 4)
        if tokens_out == 0:
            tokens_out = max(1, len(content) // 4)

        # ── Tool calls ────────────────────────────────────────────────────
        tool_calls = []

        # Path A: native tool calling (only for capable models)
        if use_native_tools:
            try:
                msg = response.message
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        fn   = tc.function
                        args = fn.arguments if isinstance(fn.arguments, dict) \
                               else _safe_json(str(fn.arguments))
                        tool_calls.append({"name": fn.name, "args": args})
            except Exception:
                pass

        # Path B: text parsing (ReAct format + XML fallback)
        if not tool_calls and content:
            tool_calls = parse_tool_calls(content)

        return LLMResponse(
            content=content, tokens_in=tokens_in, tokens_out=tokens_out,
            model=model, provider="ollama",
            duration_ms=round(duration_ms, 1),
            finish_reason="stop", tool_calls=tool_calls,
            raw={"host": self._host},
        )

    def list_models(self):
        try:
            resp = self._client().list()
            return [m.model for m in resp.models]
        except Exception:
            return ["llama3.2:3b", "llama3.2:1b", "llama3.1:8b",
                    "gemma2:2b", "phi3:mini", "mistral:7b"]


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Provider
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str = None):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("pip install openai")
        import os
        _openai.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._openai = _openai

    def complete(self, messages, model=None, max_tokens=1024, temperature=0.7,
                 stream=False, tools=None) -> LLMResponse:
        model = model or self.DEFAULT_MODEL
        start = time.time()
        kwargs = dict(model=model, messages=messages,
                      max_tokens=max_tokens, temperature=temperature)
        if tools:
            kwargs["tools"]       = _to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        resp = self._openai.chat.completions.create(**kwargs)
        msg  = resp.choices[0].message
        content    = msg.content or ""
        usage      = resp.usage
        tokens_in  = usage.prompt_tokens     if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        tool_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"name": tc.function.name,
                                   "args": _safe_json(tc.function.arguments or "{}")})
        if not tool_calls:
            tool_calls = parse_tool_calls(content)
        return LLMResponse(content=content, tokens_in=tokens_in, tokens_out=tokens_out,
                           model=model, provider="openai",
                           duration_ms=round((time.time()-start)*1000, 1),
                           finish_reason=resp.choices[0].finish_reason or "stop",
                           tool_calls=tool_calls, raw={})

    def list_models(self):
        return ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Provider
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        import os
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))

    def complete(self, messages, model=None, max_tokens=1024, temperature=0.7,
                 stream=False, tools=None) -> LLMResponse:
        model  = model or self.DEFAULT_MODEL
        start  = time.time()
        system = ""
        api_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                api_msgs.append(m)
        kwargs = dict(model=model, max_tokens=max_tokens,
                      messages=api_msgs, temperature=temperature)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t.get("description",""),
                 "input_schema": t.get("parameters", {"type":"object","properties":{}})}
                for t in tools
            ]
        resp = self._client.messages.create(**kwargs)
        content    = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "args": block.input})
        return LLMResponse(content=content,
                           tokens_in=resp.usage.input_tokens,
                           tokens_out=resp.usage.output_tokens,
                           model=model, provider="anthropic",
                           duration_ms=round((time.time()-start)*1000, 1),
                           finish_reason=resp.stop_reason or "stop",
                           tool_calls=tool_calls, raw={})

    def list_models(self):
        return ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_openai_tools(tools: list[dict]) -> list[dict]:
    result = []
    for t in tools:
        # get_definition() already wraps as {"type":"function","function":{...}}
        # handle both pre-wrapped and flat formats
        fn = t["function"] if "function" in t else t
        result.append({"type": "function", "function": {
            "name": fn.get("name", ""),
            "description": fn.get("description", "")[:200],
            "parameters": fn.get("parameters", {"type": "object", "properties": {}, "required": []}),
        }})
    return result


def get_provider(name: str, api_key: str = None, host: str = None) -> LLMProvider:
    if name == "groq":
        return GroqProvider(api_key=api_key)
    elif name == "ollama":
        return OllamaProvider(host=host)
    elif name == "openai":
        return OpenAIProvider(api_key=api_key)
    elif name == "anthropic":
        return AnthropicProvider(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider '{name}'. Choose: groq | ollama | openai | anthropic")