"""
tools.py — Tool registry with execution, permission checking, and result compression.

str_replace implements opencode's full 9-strategy cascade:
  1.  SimpleReplacer               — exact string match
  2.  LineTrimmedReplacer          — trim each line, compare
  3.  BlockAnchorReplacer          — first+last line anchors + Levenshtein similarity
  4.  WhitespaceNormalizedReplacer — collapse all whitespace
  5.  IndentationFlexibleReplacer  — strip common indent, compare
  6.  EscapeNormalizedReplacer     — unescape \\n \\t etc.
  7.  TrimmedBoundaryReplacer      — strip leading/trailing whitespace from whole block
  8.  ContextAwareReplacer         — match on trimmed non-empty lines only
  9.  MultiOccurrenceReplacer      — last resort: first-occurrence match

After a match is found the matched block is used as old_str and the replacement
is applied at the exact file position. For the IndentationFlexibleReplacer we also
delta-adjust new_str to preserve the file's absolute indentation.

Python files: syntax is validated BEFORE writing. A patch that would introduce a
SyntaxError is rejected and the file is left unchanged.
"""

import os, re, json, time, ast, subprocess, fnmatch, platform
from dataclasses import dataclass
from typing import Optional, Generator, Callable
from pathlib import Path

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


# ─────────────────────────────────────────────────────────────────────────────
# Replacement strategies  (opencode edit.ts cascade, ported to Python)
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    matrix = [[j if i == 0 else i if j == 0 else 0
               for j in range(len(b) + 1)]
              for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + cost,
            )
    return matrix[len(a)][len(b)]


def _remove_indent(text: str) -> str:
    """Strip common leading whitespace from all non-empty lines."""
    lines = text.split('\n')
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return text
    min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    return '\n'.join(l[min_indent:] if l.strip() else l for l in lines)


def _get_min_indent(text: str) -> int:
    lines = text.split('\n')
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 0
    return min(len(l) - len(l.lstrip()) for l in non_empty)


def _block_at(orig_lines: list, offsets: list, i: int, n: int) -> tuple:
    """Return (start_char, end_char, block_str) for lines[i:i+n]."""
    block_str = '\n'.join(orig_lines[i:i + n])
    start = offsets[i]
    if i + n < len(orig_lines):
        end = offsets[i + n] - 1   # exclude the newline
    else:
        end = offsets[i + n - 1] + len(orig_lines[i + n - 1])
    return start, end, block_str


def _build_offsets(orig_lines: list) -> list:
    """Cumulative char offsets for each line."""
    offsets = [0]
    for l in orig_lines:
        offsets.append(offsets[-1] + len(l) + 1)
    return offsets


# ── 1. SimpleReplacer ──────────────────────────────────────────────────────

def _simple_replacer(content: str, find: str):
    """
    Yield find only when it appears at LINE BOUNDARIES.

    This prevents a stripped 'return x' from matching as a substring inside
    '    return x  # comment' which would corrupt indentation on replacement.
    The character before the match must be newline (or start-of-file),
    and the character after must be newline (or end-of-file).

    If there is no line-boundary match, yield nothing and let the cascade
    fall through to indent-flexible and other strategies.
    """
    slen = len(find)
    clen = len(content)
    start = 0
    while True:
        pos = content.find(find, start)
        if pos == -1:
            break
        left_ok  = (pos == 0 or content[pos - 1] == "\n")
        right_ok = (pos + slen == clen or content[pos + slen] == "\n")
        if left_ok and right_ok:
            yield find
        start = pos + 1


# ── 2. LineTrimmedReplacer ─────────────────────────────────────────────────

def _line_trimmed_replacer(content: str, find: str):
    orig_lines = content.split('\n')
    search_lines = find.split('\n')
    if search_lines and search_lines[-1] == '':
        search_lines.pop()
    n = len(search_lines)
    if n == 0:
        return
    offsets = _build_offsets(orig_lines)
    for i in range(len(orig_lines) - n + 1):
        if all(orig_lines[i+j].strip() == search_lines[j].strip()
               for j in range(n)):
            start, end, block = _block_at(orig_lines, offsets, i, n)
            yield block


# ── 3. BlockAnchorReplacer ─────────────────────────────────────────────────

def _block_anchor_replacer(content: str, find: str):
    orig_lines = content.split('\n')
    search_lines = find.split('\n')
    if search_lines and search_lines[-1] == '':
        search_lines.pop()
    if len(search_lines) < 3:
        return
    first_t = search_lines[0].strip()
    last_t  = search_lines[-1].strip()
    n_s     = len(search_lines)
    offsets = _build_offsets(orig_lines)

    candidates = []
    for i, line in enumerate(orig_lines):
        if line.strip() != first_t:
            continue
        for j in range(i + 2, len(orig_lines)):
            if orig_lines[j].strip() == last_t:
                candidates.append((i, j))
                break

    if not candidates:
        return

    SINGLE_THRESH   = 0.0
    MULTIPLE_THRESH = 0.3

    def similarity(start, end):
        block_size = end - start + 1
        middle_count = min(n_s - 2, block_size - 2)
        if middle_count <= 0:
            return 1.0
        total = 0.0
        for j in range(1, n_s - 1):
            if start + j > end:
                break
            ol = orig_lines[start + j].strip()
            sl = search_lines[j].strip()
            mx = max(len(ol), len(sl))
            if mx == 0:
                continue
            total += 1.0 - _levenshtein(ol, sl) / mx
        return total / middle_count

    if len(candidates) == 1:
        s, e = candidates[0]
        if similarity(s, e) >= SINGLE_THRESH:
            start_c, end_c, block = _block_at(orig_lines, offsets, s, e - s + 1)
            yield block
        return

    best, best_sim = None, -1
    for s, e in candidates:
        sim = similarity(s, e)
        if sim > best_sim:
            best_sim, best = sim, (s, e)
    if best_sim >= MULTIPLE_THRESH and best:
        s, e = best
        start_c, end_c, block = _block_at(orig_lines, offsets, s, e - s + 1)
        yield block


# ── 4. WhitespaceNormalizedReplacer ───────────────────────────────────────

def _whitespace_normalized_replacer(content: str, find: str):
    norm = lambda t: re.sub(r'\s+', ' ', t).strip()
    find_n = norm(find)
    orig_lines = content.split('\n')
    find_lines = find.split('\n')
    offsets = _build_offsets(orig_lines)

    # Single-line
    for i, line in enumerate(orig_lines):
        if norm(line) == find_n:
            yield line

    # Multi-line
    if len(find_lines) > 1:
        n = len(find_lines)
        for i in range(len(orig_lines) - n + 1):
            block = '\n'.join(orig_lines[i:i+n])
            if norm(block) == find_n:
                yield block


# ── 5. IndentationFlexibleReplacer ────────────────────────────────────────

def _indent_flexible_replacer(content: str, find: str):
    """
    Strip common indent from find and each block; compare stripped versions.
    Yields the ORIGINAL (un-stripped) block so the caller knows the exact position.
    """
    norm_find = _remove_indent(find)
    orig_lines = content.split('\n')
    find_lines = find.split('\n')
    n = len(find_lines)
    for i in range(len(orig_lines) - n + 1):
        block = '\n'.join(orig_lines[i:i+n])
        if _remove_indent(block) == norm_find:
            yield block


# ── 6. EscapeNormalizedReplacer ───────────────────────────────────────────

def _unescape(s: str) -> str:
    return (s
        .replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        .replace("\\'", "'").replace('\\"', '"').replace('\\\\', '\\')
    )


def _escape_normalized_replacer(content: str, find: str):
    uf = _unescape(find)
    if content.find(uf) != -1:
        yield uf
    # Also block-level
    orig_lines = content.split('\n')
    find_lines = uf.split('\n')
    n = len(find_lines)
    for i in range(len(orig_lines) - n + 1):
        block = '\n'.join(orig_lines[i:i+n])
        if _unescape(block) == uf:
            yield block


# ── 7. TrimmedBoundaryReplacer ─────────────────────────────────────────────

def _trimmed_boundary_replacer(content: str, find: str):
    find_stripped = find.strip()
    if not find_stripped:
        return
    orig_lines = content.split('\n')
    find_lines = find_stripped.split('\n')
    n = len(find_lines)
    offsets = _build_offsets(orig_lines)
    for i in range(len(orig_lines) - n + 1):
        block = '\n'.join(orig_lines[i:i+n])
        if block.strip() == find_stripped:
            yield block


# ── 8. ContextAwareReplacer ───────────────────────────────────────────────

def _context_aware_replacer(content: str, find: str):
    """Match on trimmed, non-empty lines only."""
    orig_lines = content.split('\n')
    find_lines = [l for l in find.split('\n') if l.strip()]
    if not find_lines:
        return
    n = len(find_lines)
    offsets = _build_offsets(orig_lines)
    for i in range(len(orig_lines) - n + 1):
        # collect n non-empty lines starting from i
        block_orig = []
        pos = i
        while len(block_orig) < n and pos < len(orig_lines):
            if orig_lines[pos].strip():
                block_orig.append(orig_lines[pos])
            pos += 1
        if len(block_orig) < n:
            continue
        if all(block_orig[j].strip() == find_lines[j].strip() for j in range(n)):
            # yield a block spanning from i to pos-1
            end_line = pos - 1
            start_c, end_c, block = _block_at(orig_lines, offsets, i, end_line - i + 1)
            yield block


# ── 9. MultiOccurrenceReplacer ────────────────────────────────────────────

def _multi_occurrence_replacer(content: str, find: str):
    """Last resort: yield on first match anywhere (no boundary check)."""
    if find in content:
        yield find


# ─────────────────────────────────────────────────────────────────────────────
# Master replace() function — tries each strategy in order
# ─────────────────────────────────────────────────────────────────────────────

_REPLACERS = [
    ("simple",       _simple_replacer),
    ("line-trimmed", _line_trimmed_replacer),
    ("block-anchor", _block_anchor_replacer),
    ("ws-norm",      _whitespace_normalized_replacer),
    ("indent-flex",  _indent_flexible_replacer),
    ("escape-norm",  _escape_normalized_replacer),
    ("trim-bound",   _trimmed_boundary_replacer),
    ("ctx-aware",    _context_aware_replacer),
    ("multi-occur",  _multi_occurrence_replacer),
]


def _line_indent_at(content: str, idx: int) -> int:
    """
    Return the number of leading spaces on the line that contains position `idx`.
    This is the "absolute indent" of that line in the file.
    """
    # Find start of the line
    line_start = content.rfind('\n', 0, idx)
    line_start = line_start + 1 if line_start != -1 else 0
    line = content[line_start:content.find('\n', idx)]
    return len(line) - len(line.lstrip())


def _apply_replace(content: str, old_str: str, new_str: str) -> tuple:
    """
    Try each replacer in order. Returns (patched_content, method_name) or raises.

    Indent-delta logic (the critical SLM fix):
    ─────────────────────────────────────────
    SLMs often strip leading spaces from old_str. The cascade handles finding the
    right block (indent-flex, line-trimmed, etc.), but we must also fix new_str.

    After any match, we compute:
      file_indent  = indent of the line where the match starts in the file
      old_indent   = indent of old_str's first non-empty line (0 if SLM stripped it)
      delta        = file_indent - old_indent

    Then we apply delta to every line of new_str so the replacement lands at the
    correct indentation level, regardless of what the SLM wrote.

    For matched_block (from line-trimmed, indent-flex, etc.) the "file_indent" is
    taken from the matched block itself since it preserves the original file indentation.
    """
    not_found = True

    for method_name, replacer in _REPLACERS:
        for matched_block in replacer(content, old_str):
            idx = content.find(matched_block)
            if idx == -1:
                continue
            not_found = False

            # Check for multiple occurrences (ambiguous) — only for exact strategies
            if method_name in ("simple", "line-trimmed", "ws-norm"):
                last_idx = content.rfind(matched_block)
                if idx != last_idx:
                    count = content.count(matched_block)
                    ctx = _build_match_context(content, matched_block)
                    raise ValueError(
                        f"old_str matches {count} locations. "
                        f"Add more surrounding lines to make it unique.\n{ctx}"
                    )

            # ── Universal indent-delta adjustment ────────────────────────────
            # Compute the absolute indent in the file vs the indent in old_str.
            # If they differ, the SLM stripped (or added) indentation; fix new_str.
            file_indent = _get_min_indent(matched_block)
            old_indent  = _get_min_indent(old_str)
            delta       = file_indent - old_indent
            new_indent  = _get_min_indent(new_str) if new_str.strip() else 0

            actual_new = new_str
            if delta != 0 and new_indent < file_indent:
                # Only adjust when new_str doesn't already carry the right indent
                actual_new = _delta_indent(new_str, delta)

            patched = content[:idx] + actual_new + content[idx + len(matched_block):]
            return patched, method_name

    if not_found:
        raise LookupError("old_str not found")
    raise ValueError("Found multiple matches for old_str but no unique one")


def _delta_indent(text: str, delta: int) -> str:
    """Add (delta>0) or remove (delta<0) `delta` spaces from every non-empty line."""
    if delta == 0:
        return text
    lines = text.split('\n')
    result = []
    for line in lines:
        if not line.strip():
            result.append(line)
        elif delta > 0:
            result.append(' ' * delta + line)
        else:  # delta < 0
            strip = -delta
            result.append(line[strip:] if len(line) - len(line.lstrip()) >= strip else line)
    return '\n'.join(result)


def _build_match_context(content: str, pattern: str) -> str:
    """Show up to 3 match contexts."""
    lines = content.splitlines()
    pat_lines = pattern.splitlines()
    hits = []
    for i, line in enumerate(lines):
        if pat_lines[0].strip() in line:
            hits.append(f"  Line {i+1}: {line.rstrip()}")
        if len(hits) >= 3:
            break
    return '\n'.join(hits)


def _check_python_syntax(content: str) -> tuple:
    """Return (ok: bool, error_msg: str). Uses ast.parse — no execution."""
    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


# ─────────────────────────────────────────────────────────────────────────────
# ToolRegistry
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    DANGEROUS_TOOLS = {"bash", "write_file", "delete_file"}

    MAX_OUTPUT = {
        "bash":        3000,
        "read_file":   4000,
        "grep":        2000,
        "list_dir":    1200,
        "find_files":  1000,
        "web_search":  1500,
        "memory_read": 800,
        "str_replace": 600,
        "create_file": 400,
        "run_python":  3000,
    }

    def __init__(self, allowed: list = None, require_approval: list = None):
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
            "str_replace":  self._str_replace,
            "create_file":  self._create_file,
            "run_python":   self._run_python,
        }

    def is_allowed(self, tool_name: str) -> bool:
        return self._allowed is None or tool_name in self._allowed

    def all_tool_names(self) -> list:
        return list(self._tool_handlers.keys())

    def requires_approval(self, tool_name: str) -> bool:
        return tool_name in self._require_approval

    def execute(self, tool_name: str, args: dict) -> ToolResult:
        if not self.is_allowed(tool_name):
            return ToolResult(success=False, output="",
                              error=f"Tool '{tool_name}' is not allowed for this agent")
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return ToolResult(success=False, output="",
                              error=f"Unknown tool: {tool_name}. Available: {list(self._tool_handlers.keys())}")
        try:
            result = handler(args)
            max_len = self.MAX_OUTPUT.get(tool_name, 1200)
            if len(result.output) > max_len:
                result.output = result.output[:max_len] + f"\n... [output truncated at {max_len} chars]"
            return result
        except Exception as e:
            return ToolResult(success=False, output="", error=f"{type(e).__name__}: {e}")

    def get_definitions_for_step(self, step: int, task: str) -> list:
        task_lower = task.lower()
        step_tools = {"task_done", "read_file", "grep", "find_files", "list_dir"}
        write_kw = ["fix","write","create","update","modify","edit","add","implement","change","refactor","correct"]
        if any(kw in task_lower for kw in write_kw):
            step_tools.update(["write_file", "append_file", "str_replace"])
        bash_kw = ["run","execute","test","install","build","compile","start","deploy","check","verify","debug"]
        if any(kw in task_lower for kw in bash_kw):
            step_tools.add("bash")
        search_kw = ["search","find","look up","research","web","internet"]
        if any(kw in task_lower for kw in search_kw):
            step_tools.add("web_search")
        if step <= 2:
            step_tools.add("memory_read")
        if self._allowed:
            step_tools = step_tools & self._allowed
        return [self._get_definition(t) for t in step_tools if t in self._tool_handlers]

    def _get_definition(self, tool_name: str) -> dict:
        definitions = {
            "bash": {
                "name": "bash",
                "description": "Run shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "cwd":     {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
            "read_file": {
                "name": "read_file",
                "description": "Read file contents with line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path":       {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line":   {"type": "integer"},
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
                        "path":      {"type": "string"},
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
                        "pattern":        {"type": "string"},
                        "path":           {"type": "string"},
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
                        "pattern":     {"type": "string"},
                        "directory":   {"type": "string", "default": "."},
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
                "description": "Read from agent memory.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            "memory_write": {
                "name": "memory_write",
                "description": "Store a fact to memory.",
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
            "str_replace": {
                "name": "str_replace",
                "description": (
                    "Targeted file patch. Find EXACT text and replace it. "
                    "Safer than write_file for bug fixes. "
                    "old_str must be copied VERBATIM from the file including all indentation. "
                    "For multi-line new_str: use the SAME indentation level as old_str's first line."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path":    {"type": "string", "description": "File path to patch"},
                        "old_str": {"type": "string", "description": "Exact text to find — copy from file verbatim"},
                        "new_str": {"type": "string", "description": "Replacement text with correct indentation"},
                    },
                    "required": ["path", "old_str", "new_str"],
                },
            },
            "run_python": {
                "name": "run_python",
                "description": "Run a Python file. Auto-uses pytest for test files. Shows [PASSED]/[FAILED].",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path":    {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
            "create_file": {
                "name": "create_file",
                "description": "Create a new file with given content. Fails if file already exists.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path":    {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            "task_done": {
                "name": "task_done",
                "description": "Signal that the task is complete.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "output":  {"type": "string"},
                    },
                    "required": ["summary"],
                },
            },
            "ask_human": {
                "name": "ask_human",
                "description": "Ask the human a question when stuck.",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        }
        return definitions.get(tool_name, {"name": tool_name, "description": "", "parameters": {}})

    def get_definition(self, allowed=None) -> list:
        tool_names = list(self._tool_handlers.keys())
        if allowed:
            tool_names = [n for n in tool_names if n in allowed]
        return [
            {"type": "function", "function": self._get_definition(n)}
            for n in tool_names
            if self._get_definition(n)
        ]

    # ── Tool implementations ──────────────────────────────────────────────────

    @staticmethod
    def _win_translate(command: str) -> str:
        if not _IS_WINDOWS:
            return command
        cmd = command.strip()
        cmd = re.sub(r'\bpython3\b', 'python', cmd)
        if re.match(r'^ls(-la?|-al?)?\s*$', cmd):
            return 'dir'
        if cmd == 'pwd':
            return 'cd'
        m = re.match(r'^cat\s+(.+)', cmd)
        if m:
            return f'type {m.group(1)}'
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
        command = self._win_translate(command)
        dangerous = ["rm -rf /", "format", "mkfs", ":(){:|:&};:"]
        for d in dangerous:
            if d in command:
                return ToolResult(success=False, output="", error=f"Blocked dangerous command: {d}")
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd,
            )
            output = (result.stdout + result.stderr).strip()
            if not output:
                output = f"[Command completed with exit code {result.returncode}]"
            return ToolResult(
                success=result.returncode == 0, output=output,
                error=result.stderr if result.returncode != 0 else None,
                metadata={"exit_code": result.returncode, "command": command},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error=f"Command timed out after {timeout}s")

    def _read_file(self, args: dict) -> ToolResult:
        path = args.get("path", "")
        if not path:
            return ToolResult(success=False, output="", error="No path provided")
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")
            if not p.is_file():
                return ToolResult(success=False, output="", error=f"Not a file: {path}")
            content = p.read_text(errors="replace")
            lines   = content.split("\n")
            total   = len(lines)
            start   = args.get("start_line")
            end     = args.get("end_line")
            if start is not None or end is not None:
                s = max(0, int(start or 1) - 1)
                e = min(total, int(end or total))
                selected = lines[s:e]
                numbered = "\n".join(f"{s+i+1}: {l}" for i, l in enumerate(selected))
                return ToolResult(success=True,
                                  output=f"[{path} — lines {s+1}-{e} of {total}]\n" + numbered,
                                  metadata={"path": path, "total_lines": total})
            numbered = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines))
            return ToolResult(success=True,
                              output=f"[{path} — {total} lines]\n" + numbered,
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
            return ToolResult(success=True, output=f"Written {lines} lines to {path} ({len(content)} bytes)")
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
                            matches.append(f"{rel}:{i}: {line.strip()[:120]}")
                            if len(matches) >= 60:
                                break
                except (PermissionError, UnicodeDecodeError):
                    pass
                if len(matches) >= 60:
                    break
            if not matches:
                return ToolResult(success=True, output=f"No matches for '{pattern}' in {path}")
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
                if any(part in _EXCLUDED_DIRS for part in rel.parts[:-1]):
                    continue
                if fnmatch.fnmatch(item.name, pattern):
                    matches.append(str(rel))
                    if len(matches) >= max_results:
                        break
            if not matches:
                return ToolResult(success=True, output=f"No files matching '{pattern}' in {directory}")
            return ToolResult(success=True,
                              output=f"{len(matches)} file(s) matching '{pattern}':\n" + "\n".join(sorted(matches)))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _web_search(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        if not query:
            return ToolResult(success=False, output="", error="No query")
        try:
            import urllib.request, urllib.parse
            q   = urllib.parse.quote(query)
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
        return ToolResult(success=True, output=f"[memory_read: '{query}' — no memories yet]")

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

    def _str_replace(self, args: dict) -> ToolResult:
        """
        Apply a targeted find-and-replace using opencode's 9-strategy cascade.

        Tries strategies in order from strictest (exact match) to most lenient
        (fuzzy matching). The first strategy that finds a unique match wins.

        Python files: syntax is validated after patching.
        A syntactically invalid patch is rejected, file left unchanged.
        """
        path    = args.get("path", "")
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")

        if not path:
            return ToolResult(success=False, output="", error="path is required")
        if old_str is None:
            return ToolResult(success=False, output="", error="old_str is required")
        if old_str == new_str:
            return ToolResult(success=False, output="",
                              error="old_str and new_str are identical — no change needed")

        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(success=False, output="", error=f"File not found: {path}")

            original = p.read_text(encoding="utf-8", errors="replace")

            try:
                patched, method = _apply_replace(original, old_str, new_str)
            except LookupError:
                # Build a helpful hint
                hint_lines = []
                first_old_trimmed = (old_str.strip().splitlines() or [""])[0][:40]
                for i, line in enumerate(original.splitlines(), 1):
                    if first_old_trimmed and first_old_trimmed in line:
                        hint_lines.append(f"  Line {i}: {repr(line)}")
                hint = ("\nHint — similar lines:\n" + "\n".join(hint_lines[:4])) if hint_lines else ""
                return ToolResult(success=False, output="",
                                  error=f"old_str not found in {path}. "
                                        f"Read the file and copy the exact text.{hint}")
            except ValueError as e:
                return ToolResult(success=False, output="", error=str(e))

            # ── Syntax validation for Python files ───────────────────────────
            if path.endswith(".py"):
                ok, err = _check_python_syntax(patched)
                if not ok:
                    return ToolResult(success=False, output="",
                                      error=f"Patch rejected — would introduce {err}. "
                                            f"Check indentation in new_str.")

            p.write_text(patched, encoding="utf-8")

            # Compact diff preview
            old_lines = old_str.splitlines()
            new_lines = new_str.splitlines()
            diff = ""
            for l in old_lines[:3]:
                diff += f"  - {l}\n"
            for l in new_lines[:3]:
                diff += f"  + {l}\n"
            if len(old_lines) > 3:
                diff += f"  ... ({len(old_lines)-3} more lines)\n"

            return ToolResult(
                success=True,
                output=f"str_replace applied to {path} ({method})\n{diff.strip()}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _create_file(self, args: dict) -> ToolResult:
        path    = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return ToolResult(success=False, output="", error="path required")
        try:
            p = Path(path)
            if p.exists():
                return ToolResult(success=False, output="",
                                  error=f"{path} already exists. Use write_file to overwrite.")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Created {path} ({len(content.splitlines())} lines)")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _run_python(self, args: dict) -> ToolResult:
        """
        Run a Python file. Auto-detects test files and uses pytest.
        First checks syntax to give a clear error before attempting execution.
        """
        path    = args.get("path", "")
        timeout = int(args.get("timeout", 30))
        extra   = args.get("args", "")
        if not path:
            return ToolResult(success=False, output="", error="path required")

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                src = f.read()
            ok, err = _check_python_syntax(src)
            if not ok:
                return ToolResult(success=False, output=f"[SYNTAX ERROR]\n{err}", error=err)
        except Exception:
            pass

        is_test = "test" in Path(path).name.lower() or "spec" in Path(path).name.lower()
        if is_test:
            cmd = f"python -m pytest {path} -v --tb=short 2>&1"
        else:
            cmd = f"python {path} {extra}".strip()

        if _IS_WINDOWS:
            env_cmd = f"set PYTHONIOENCODING=utf-8 && {cmd}"
        else:
            env_cmd = f"PYTHONIOENCODING=utf-8 {cmd}"

        result = self._bash({"command": env_cmd, "timeout": timeout})
        out    = result.output or ""
        passed = result.success and all(
            x not in out.lower()
            for x in ("traceback", "error:", "assertionerror", "syntaxerror",
                       "nameerror", "typeerror", "failed")
        )
        banner = "[PASSED]" if passed else "[FAILED]"
        return ToolResult(
            success=result.success,
            output=f"{banner}\n{out}",
            error=result.error,
        )

    def _ask_human(self, args: dict) -> ToolResult:
        question = args.get("question", "")
        print(f"\n❓ Agent asks: {question}")
        try:
            answer = input("Your answer: ").strip()
            return ToolResult(success=True, output=f"Human says: {answer}")
        except (EOFError, KeyboardInterrupt):
            return ToolResult(success=False, output="", error="No human input available")
