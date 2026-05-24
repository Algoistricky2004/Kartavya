"""
tools.py — Tool registry with execution, permission checking, and result compression.

Tools available:
  bash          — Execute shell commands (with safety checks)
  read_file     — Read file contents (with smart windowing)
  write_file    — Write/overwrite file contents
  append_file   — Append to file
  list_dir      — List directory contents
  grep          — Search file contents
  find_files    — Find files by name/pattern
  web_search    — Simple web search (DuckDuckGo)
  memory_read   — Read from agent memory
  memory_write  — Write to agent memory
  task_done     — Signal task completion
  ask_human     — Request human input
"""

import os
import re
import json
import time
import subprocess
import fnmatch
import platform
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Directories to always skip when searching files (avoids 700+ venv results)
_EXCLUDED_DIRS = frozenset({
    '__pycache__', '.git', '.hg', '.svn',
    'node_modules', 'bower_components',
    'venv', '.venv', 'env', 'virtualenv',
    'site-packages', 'dist-packages',
    'dist', 'build', 'target', '.eggs', '.tox',
    '.pytest_cache', '.mypy_cache', '.ruff_cache',
    '.idea', '.vscode',
})

_IS_WINDOWS = platform.system() == "Windows"


@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        self.metadata = self.metadata or {}


class ToolRegistry:
    """
    Registry of available tools with:
    - Permission enforcement
    - Automatic result compression for context window
    - Execution timeouts
    - Safe defaults
    """

    # Tools that need extra caution — ask before running
    DANGEROUS_TOOLS  = {"bash", "write_file", "delete_file"}

    # Maximum output lengths (chars) per tool
    MAX_OUTPUT = {
        "bash":       2000,
        "read_file":  3000,
        "grep":       1500,
        "list_dir":   1000,
        "find_files": 800,
        "web_search": 1500,
        "memory_read": 800,
    }

    def __init__(self, allowed: list[str] = None, require_approval: list[str] = None):
        """
        allowed: list of tool names to allow. None = all.
        require_approval: list of tool names that need human approval.
        """
        self._allowed          = set(allowed) if allowed else None
        self._require_approval = set(require_approval or [])
        self._tool_handlers    = {
            "bash":         self._bash,
            "read_file":    self._read_file,
            "write_file":   self._write_file,
            "append_file":  self._append_file,
            "list_dir":     self._list_dir,
            "grep":         self._grep,
            "find_files":   self._find_files,
            "web_search":   self._web_search,
            "memory_read":  self._memory_read,
            "memory_write": self._memory_write,
            "task_done":    self._task_done,
            "ask_human":    self._ask_human,
        }

    def is_allowed(self, tool_name: str) -> bool:
        if self._allowed is None:
            return True
        return tool_name in self._allowed

    def all_tool_names(self) -> list[str]:
        return list(self._tool_handlers.keys())

    def requires_approval(self, tool_name: str) -> bool:
        return tool_name in self._require_approval

    def execute(self, tool_name: str, args: dict) -> ToolResult:
        """Execute a tool and return compressed result."""
        if not self.is_allowed(tool_name):
            return ToolResult(success=False, output="",
                              error=f"Tool '{tool_name}' is not allowed for this agent")

        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return ToolResult(success=False, output="",
                              error=f"Unknown tool: {tool_name}. Available: {list(self._tool_handlers.keys())}")

        try:
            result = handler(args)
            # Compress output to fit context window
            max_len = self.MAX_OUTPUT.get(tool_name, 1000)
            if len(result.output) > max_len:
                truncated = result.output[:max_len]
                result.output = truncated + f"\n... [output truncated at {max_len} chars]"
            return result
        except Exception as e:
            return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")

    def get_definitions_for_step(self, step: int, task: str) -> list[dict]:
        """
        Return only the tools likely needed for this step.
        This is the KEY optimization for small context windows:
        never inject all 12 tools when you only need 2-3.
        """
        task_lower = task.lower()
        step_tools = set()

        # Always include: task_done (to signal completion)
        step_tools.add("task_done")

        # File reading tools (almost always needed)
        step_tools.update(["read_file", "grep", "find_files", "list_dir"])

        # Writing tools (only if task involves writing)
        write_keywords = ["fix", "write", "create", "update", "modify", "edit",
                          "add", "implement", "change", "refactor", "correct"]
        if any(kw in task_lower for kw in write_keywords):
            step_tools.update(["write_file", "append_file"])

        # Bash (only if task involves running commands)
        bash_keywords = ["run", "execute", "test", "install", "build", "compile",
                         "start", "deploy", "check", "verify", "debug"]
        if any(kw in task_lower for kw in bash_keywords):
            step_tools.add("bash")

        # Web search (only if research needed)
        search_keywords = ["search", "find", "look up", "research", "web", "internet"]
        if any(kw in task_lower for kw in search_keywords):
            step_tools.add("web_search")

        # Memory tools (always available but rarely needed in context)
        if step <= 2:  # Only on early steps when context is being established
            step_tools.add("memory_read")

        # Human help (only on later steps when stuck)
        if step > self._calculate_max_steps(task) * 0.7:
            step_tools.add("ask_human")

        # Filter by allowed
        if self._allowed:
            step_tools = step_tools & self._allowed

        return [self._get_definition(t) for t in step_tools
                if t in self._tool_handlers]

    def _calculate_max_steps(self, task: str) -> int:
        """Estimate task complexity from description."""
        words = len(task.split())
        if words < 10: return 5
        if words < 30: return 10
        return 20

    def _get_definition(self, tool_name: str) -> dict:
        """Terse tool definition (saves tokens vs full JSON schema)."""
        definitions = {
            "bash": {
                "name": "bash",
                "description": "Run shell command. Use for: tests, builds, file operations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
                        "cwd": {"type": "string", "description": "Working directory"},
                    },
                    "required": ["command"],
                },
            },
            "read_file": {
                "name": "read_file",
                "description": "Read file contents. Supports line range for large files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "description": "Start line (optional)"},
                        "end_line":   {"type": "integer", "description": "End line (optional)"},
                    },
                    "required": ["path"],
                },
            },
            "write_file": {
                "name": "write_file",
                "description": "Write content to file (creates or overwrites).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path":    {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            "append_file": {
                "name": "append_file",
                "description": "Append content to end of file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            },
            "list_dir": {
                "name": "list_dir",
                "description": "List directory contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["path"],
                },
            },
            "grep": {
                "name": "grep",
                "description": "Search for pattern in files. Returns matching lines with line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path":    {"type": "string"},
                        "case_sensitive": {"type": "boolean", "default": False},
                    },
                    "required": ["pattern", "path"],
                },
            },
            "find_files": {
                "name": "find_files",
                "description": "Find files matching a name pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "e.g. '*.py', 'test_*.py'"},
                        "directory": {"type": "string", "default": "."},
                        "max_results": {"type": "integer", "default": 20},
                    },
                    "required": ["pattern"],
                },
            },
            "web_search": {
                "name": "web_search",
                "description": "Search the web via DuckDuckGo.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            "memory_read": {
                "name": "memory_read",
                "description": "Read from agent memory. Returns relevant stored facts.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            "memory_write": {
                "name": "memory_write",
                "description": "Store a fact to memory for later retrieval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content":  {"type": "string"},
                        "section":  {"type": "string", "enum": ["project_facts","decisions","errors_seen"]},
                        "priority": {"type": "string", "enum": ["high","normal","low"]},
                    },
                    "required": ["content", "section"],
                },
            },
            "task_done": {
                "name": "task_done",
                "description": "Signal that the task is complete. Include final answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "What was accomplished"},
                        "output":  {"type": "string", "description": "Final output/answer"},
                    },
                    "required": ["summary"],
                },
            },
            "ask_human": {
                "name": "ask_human",
                "description": "Ask the human user a question when stuck.",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        }
        return definitions.get(tool_name, {"name": tool_name, "description": "", "parameters": {}})
    def get_definition(self, allowed=None) -> list[dict]:
        """
        Return tool schemas formatted for LLM tool/function calling.
        """

        tool_names = list(self._tool_handlers.keys())

        # Filter allowed tools
        if allowed:
            tool_names = [
                name for name in tool_names
                if name in allowed
            ]

        defs = []

        for name in tool_names:
            tool_def = self._get_definition(name)

            if tool_def:
                defs.append({
                    "type": "function",
                    "function": tool_def,
                })

        return defs

    # ── Tool implementations ───────────────────────────────────────────────

    @staticmethod
    def _win_translate(command: str) -> str:
        """Translate common Unix commands to Windows equivalents."""
        if not _IS_WINDOWS:
            return command
        cmd = command.strip()
        # python3 -> python (most important on Windows)
        cmd = re.sub(r'\bpython3\b', 'python', cmd)
        # ls -> dir
        if re.match(r'^ls(-la?|-al?)?\s*$', cmd):
            return 'dir'
        # pwd -> cd
        if cmd == 'pwd':
            return 'cd'
        # cat file -> type file
        m = re.match(r'^cat\s+(.+)', cmd)
        if m:
            return f'type {m.group(1)}'
        # which cmd -> where cmd
        m = re.match(r'^which\s+(\S+)', cmd)
        if m:
            return f'where {m.group(1)}'
        return cmd

    def _bash(self, args: dict) -> ToolResult:
        command = args.get("command", "")
        if not command:
            return ToolResult(success=False, output="", error="No command provided")

        timeout = int(args.get("timeout", 30))
        cwd     = args.get("cwd", os.getcwd())

        # Translate common Unix commands to Windows equivalents
        command = self._win_translate(command)

        # Safety: block most destructive commands
        dangerous = ["rm -rf /", "format", "mkfs", ":(){:|:&};:"]
        for d in dangerous:
            if d in command:
                return ToolResult(success=False, output="",
                                  error=f"Blocked dangerous command: {d}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            output = (result.stdout + result.stderr).strip()
            if not output:
                output = f"[Command completed with exit code {result.returncode}]"
            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=result.stderr if result.returncode != 0 else None,
                metadata={"exit_code": result.returncode, "command": command},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="",
                              error=f"Command timed out after {timeout}s: {command}")

    def _read_file(self, args: dict) -> ToolResult:
        path = args.get("path", "")
        if not path:
            return ToolResult(success=False, output="", error="No path provided")

        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(success=False, output="",
                                  error=f"File not found: {path}")
            if not p.is_file():
                return ToolResult(success=False, output="",
                                  error=f"Not a file: {path}")

            content = p.read_text(errors="replace")
            lines   = content.split("\n")
            total   = len(lines)

            start = args.get("start_line")
            end   = args.get("end_line")

            if start is not None or end is not None:
                s = max(0, int(start or 1) - 1)
                e = min(total, int(end or total))
                selected = lines[s:e]
                numbered = "\n".join(f"{s+i+1}: {l}" for i, l in enumerate(selected))
                header   = f"[{path} — lines {s+1}-{e} of {total}]\n"
                return ToolResult(success=True, output=header + numbered,
                                  metadata={"path": path, "total_lines": total})

            # Return all with line numbers (will be truncated by registry)
            numbered = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines))
            return ToolResult(success=True, output=f"[{path} — {total} lines]\n" + numbered,
                              metadata={"path": path, "total_lines": total})

        except PermissionError:
            return ToolResult(success=False, output="", error=f"Permission denied: {path}")

    def _write_file(self, args: dict) -> ToolResult:
        path    = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return ToolResult(success=False, output="", error="No path")
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            lines = len(content.split("\n"))
            return ToolResult(success=True,
                              output=f"Written {lines} lines to {path} ({len(content)} bytes)")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _append_file(self, args: dict) -> ToolResult:
        path    = args.get("path", "")
        content = args.get("content", "")
        try:
            with open(path, "a") as f:
                f.write(content)
            return ToolResult(success=True, output=f"Appended {len(content)} chars to {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _list_dir(self, args: dict) -> ToolResult:
        path      = args.get("path", ".")
        recursive = args.get("recursive", False)
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(success=False, output="", error=f"Path not found: {path}")

            if recursive:
                entries = []
                for item in sorted(p.rglob("*"))[:100]:
                    rel = item.relative_to(p)
                    entries.append(f"{'[D]' if item.is_dir() else '[F]'} {rel}")
                return ToolResult(success=True, output="\n".join(entries))
            else:
                entries = []
                for item in sorted(p.iterdir()):
                    icon = "📁" if item.is_dir() else "📄"
                    size = f"({item.stat().st_size}b)" if item.is_file() else ""
                    entries.append(f"{icon} {item.name} {size}")
                return ToolResult(success=True, output=f"[{path}]\n" + "\n".join(entries))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _grep(self, args: dict) -> ToolResult:
        pattern = args.get("pattern", "")
        path    = args.get("path", ".")
        case    = args.get("case_sensitive", False)
        if not pattern:
            return ToolResult(success=False, output="", error="No pattern")

        flags = 0 if case else re.IGNORECASE
        matches = []

        try:
            p = Path(path)
            files = [p] if p.is_file() else list(p.rglob("*"))
            for f in files:
                if not f.is_file():
                    continue
                try:
                    text = f.read_text(errors="replace")
                    for i, line in enumerate(text.split("\n"), 1):
                        if re.search(pattern, line, flags):
                            rel = f.relative_to(Path(path)) if p.is_dir() else f
                            matches.append(f"{rel}:{i}: {line.strip()[:100]}")
                            if len(matches) >= 50:
                                break
                except (PermissionError, UnicodeDecodeError):
                    pass
                if len(matches) >= 50:
                    break

            if not matches:
                return ToolResult(success=True,
                                  output=f"No matches for '{pattern}' in {path}")
            return ToolResult(success=True,
                              output=f"{len(matches)} match(es) for '{pattern}':\n" + "\n".join(matches))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _find_files(self, args: dict) -> ToolResult:
        pattern     = args.get("pattern", "*")
        directory   = args.get("directory", ".")
        max_results = int(args.get("max_results", 20))
        try:
            p = Path(directory).resolve()
            matches = []
            for item in p.rglob("*"):
                if not item.is_file():
                    continue
                try:
                    rel = item.relative_to(p)
                except ValueError:
                    continue
                # Skip non-project dirs: venv, __pycache__, .git, node_modules…
                if any(part in _EXCLUDED_DIRS for part in rel.parts[:-1]):
                    continue
                if fnmatch.fnmatch(item.name, pattern):
                    matches.append(str(rel))
                    if len(matches) >= max_results:
                        break
            if not matches:
                return ToolResult(success=True,
                                  output=f"No files matching '{pattern}' in {directory}")
            return ToolResult(
                success=True,
                output=f"{len(matches)} file(s) matching '{pattern}':\n"
                       + "\n".join(sorted(matches)),
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _web_search(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        if not query:
            return ToolResult(success=False, output="", error="No query")
        try:
            import urllib.request, urllib.parse
            q = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
            req = urllib.request.Request(url, headers={"User-Agent": "AgentOS/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            results = []
            if data.get("AbstractText"):
                results.append(f"Summary: {data['AbstractText'][:300]}")
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"• {topic['Text'][:150]}")
            if not results:
                return ToolResult(success=True, output=f"No results for: {query}")
            return ToolResult(success=True, output="\n".join(results))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Search failed: {e}")

    def _memory_read(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        # Import lazily to avoid circular imports
        return ToolResult(success=True,
                          output=f"[memory_read: use memory.build_context_chunks('{query}')]")

    def _memory_write(self, args: dict) -> ToolResult:
        content  = args.get("content", "")
        section  = args.get("section", "project_facts")
        priority = args.get("priority", "normal")
        return ToolResult(success=True,
                          output=f"[memory_write: stored '{content[:50]}' to {section}]",
                          metadata={"section": section, "priority": priority, "content": content})

    def _task_done(self, args: dict) -> ToolResult:
        summary = args.get("summary", "Task completed")
        output  = args.get("output", "")
        return ToolResult(success=True,
                          output=f"TASK_DONE: {summary}" + (f"\n{output}" if output else ""),
                          metadata={"done": True, "summary": summary})

    def _ask_human(self, args: dict) -> ToolResult:
        question = args.get("question", "")
        print(f"\n❓ Agent asks: {question}")
        try:
            answer = input("Your answer: ").strip()
            return ToolResult(success=True, output=f"Human says: {answer}")
        except (EOFError, KeyboardInterrupt):
            return ToolResult(success=False, output="", error="No human input available")