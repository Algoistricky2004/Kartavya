# Kartavya — SLM-First Agent Harness: System Design

## The Problem

Kartavya was built for powerful frontier LLMs (GPT-4, Claude 3) but fails silently with Small Language Models (SLMs like `qwen2.5-coder:7b`, `deepseek-coder:6.7b`, `llama3.1:8b`). The demo video shows the agent identifying all 3 bugs correctly yet leaving the file in a broken, un-parseable state after patching.

**Root cause (str_replace):**  
SLMs copy `old_str` from `read_file` output which adds `"21: "` line-number prefixes, OR they simply strip the 4-space indentation when writing JSON. The previous implementation used `str.count(old_str)` — a substring match — which would find the stripped string INSIDE the indented line. Then `str.replace()` replaced only the substring, orphaning the leading 4 spaces in front of the multi-line replacement. Python's parser raised `IndentationError`. The agent reported "patch applied" but the file was destroyed.

---

## Architecture: The Three-Phase Fix

```
┌─────────────────────────────────────────────────────────────────┐
│                     TASK: "fix bugs in file.py"                  │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
          ┌─────────────────────────▼──────────────────────────┐
          │              Phase 0: Pre-Bootstrap                  │
          │   (deterministic, no LLM)                            │
          │   1. find_files("*buggy*.py")                        │
          │   2. Path.read_text() → RAW content (no line nums)   │
          │   3. run_python() → capture current error            │
          └─────────────────────────┬──────────────────────────┘
                                    │  raw_content + run_error
          ┌─────────────────────────▼──────────────────────────┐
          │              Phase 1: Plan (1 LLM call)              │
          │   System: "Return JSON array of ALL patches"         │
          │   User:   <raw file> + <current error>               │
          │   SLM→    [{path,old_str,new_str}, ...]              │
          │   Parse:  9-strategy JSON extraction + validation     │
          └─────────────────────────┬──────────────────────────┘
                                    │  patches[]
          ┌─────────────────────────▼──────────────────────────┐
          │           Phase 2: Execute (deterministic)           │
          │   For each patch:                                    │
          │     str_replace → 9-strategy cascade → syntax check  │
          │     run_python → [PASSED] / [FAILED]                 │
          │     Early exit on [PASSED]                           │
          │   Fallback: strip line-number prefixes, retry        │
          └─────────────────────────────────────────────────────┘
```

---

## The str_replace Cascade (9 Strategies)

Ported from [opencode's `edit.ts`](https://github.com/opencode-ai/opencode) with Python adaptations. Each strategy tries to find `old_str` in the file content; the first one that succeeds wins. The **key insight**: when a match is found, the matched block carries the file's ORIGINAL indentation, so we can compute `delta = file_indent - old_str_indent` and apply it to `new_str`.

```
Strategy 1: SimpleReplacer
  - Exact match at LINE BOUNDARIES only (char before = '\n' or start-of-file)
  - Prevents mid-line substring matches that corrupt indentation
  - Most common path for correctly-indented SLM output

Strategy 2: LineTrimmedReplacer
  - Trims each line before comparing
  - Handles trailing whitespace, SLM-stripped leading spaces
  - Most common fallback for SLMs (they often strip leading spaces)
  → DELTA applied: file indentation restored via delta adjustment

Strategy 3: BlockAnchorReplacer
  - Matches multi-line blocks by first+last line anchors
  - Uses Levenshtein similarity for middle lines
  - Handles SLMs that slightly paraphrase the code
  → Only triggers for blocks ≥ 3 lines

Strategy 4: WhitespaceNormalizedReplacer
  - Collapses all whitespace sequences to single space
  - Handles extra/missing spaces anywhere

Strategy 5: IndentationFlexibleReplacer (opencode original)
  - Strips common indent from both old_str and each candidate block
  - Compares the de-indented versions

Strategy 6: EscapeNormalizedReplacer
  - Unescapes \n \t \r sequences
  - Handles JSON-escaped strings from SLM output

Strategy 7: TrimmedBoundaryReplacer
  - Strips leading/trailing whitespace from the whole block
  - Last resort for blocks with inconsistent outer whitespace

Strategy 8: ContextAwareReplacer
  - Matches on non-empty trimmed lines only (skips blank lines)
  - Handles when SLM omits blank lines between statements

Strategy 9: MultiOccurrenceReplacer
  - Last resort: yields on first match anywhere
  - Only used when all else fails
```

After any match: **syntax validation** (`ast.parse`) — if the patch would introduce a `SyntaxError`, the file is left unchanged and the error is reported.

---

## Why Raw Content for Planning

The planner uses `Path.read_text()` directly (no line numbers) because:

```
What read_file returns:     What old_str must be:
  21:     return a + b  →       return a + b     (stripped)
  OR:     return a + b  →       return a + b     (stripped)
vs:
  21:     return a + b  →   21:     return a + b  (BROKEN — has "21:" prefix)
```

When the model sees `21:     return a + b` it might copy the prefix `21:` into `old_str`. The `_strip_line_number_prefixes` fallback handles this, but showing raw content avoids the problem entirely.

---

## The Delta-Indent Mechanism

This is the core fix for SLM outputs:

```python
# SLM says: old_str = "return a / b  # BUG"  (0-space indent)
# File has:           "    return a / b  # BUG"  (4-space indent)
# SLM says: new_str = "if b == 0:\n    return None\nreturn a / b"  (0, 4, 0 relative)

file_indent = _get_min_indent(matched_block)  # = 4
old_indent  = _get_min_indent(old_str)        # = 0
delta       = file_indent - old_indent        # = 4

adjusted_new = _delta_indent(new_str, delta)
# = "    if b == 0:\n        return None\n    return a / b"  (4, 8, 4)
# ✓ Correct Python indentation
```

---

## Planning Prompt Design for SLMs

Two critical rules that make SLMs produce better patches:

**Rule 1: Show raw file content, not numbered content**  
The file is shown verbatim. The model reads `    return a + b  # BUG` and should copy those exact 4 spaces into `old_str`. The system prompt reinforces this with a worked example showing correct vs wrong.

**Rule 2: Constrain output format strictly**  
```
Return ONLY this JSON array, nothing else:
[{"path": "file.py", "old_str": "exact buggy text", "new_str": "fixed text"}]

RULES:
1. old_str must be COPIED EXACTLY — same spaces, same comments
2. new_str must have SAME indent as old_str's first line
3. Return ONLY the JSON. No explanations, no markdown fences.
```

Plus robust JSON extraction handles common SLM formatting failures:
- `<think>...</think>` blocks (DeepSeek-R1, Qwen3)
- ` ```json ... ``` ` markdown fences
- Single-quoted JSON (`'` → `"`)
- Trailing commas in arrays
- Single-object `{}` instead of array `[{}]`

---

## What Was Changed vs Original Kartavya

| Component | Before | After |
|---|---|---|
| `tools/tools.py` `_str_replace` | 1 strategy (substring count) | 9-strategy cascade with delta-indent |
| `tools/tools.py` `_run_python` | No syntax pre-check | Syntax check before exec → `[SYNTAX ERROR]` |
| `core/agent_os.py` `_code_bootstrap_raw` | New method | Returns raw file (no `"21:"` prefixes) |
| `core/agent_os.py` `_plan_all_patches` | Basic JSON extraction | Robust extraction + think-block stripping + worked examples |
| `core/agent_os.py` `_code_correction_execute` | Single-attempt patch | Plan→Execute with fallback (strip prefixes, retry) |
| `core/agent_os.py` `_strip_line_number_prefixes` | Not present | Static method for `"21: line"` → `"line"` |
| `demo/buggy_calculator.py` | Unknown state from prior runs | Reset to clean 3-bug state |
| `tests/test_harness.py` | Empty | 49 tests covering all above |

---

## How to Run

```bash
# 1. Verify it's broken (shows AssertionError: subtract(5,3))
python demo/buggy_calculator.py

# 2. Fix with Kartavya + Ollama
python cli/main.py run \
  "fix the bugs in demo/buggy_calculator.py" \
  --provider ollama \
  --model qwen2.5-coder:7b

# 3. Verify it's fixed
python demo/buggy_calculator.py

# 4. Run the test suite
python -m pytest tests/test_harness.py -v
```

---

## Lessons from opencode

opencode (`opencode-ai/opencode`) taught us three things:

1. **Cascade strategies, not one**: Real code editors use 9+ fallback strategies because no single strategy handles all model outputs. Each strategy targets a specific failure mode.

2. **The `IndentationFlexibleReplacer` is not enough alone**: It handles stripped indentation but not escaped strings, whitespace normalization, or block-anchor matching. All are needed.

3. **Separate tool phase from reasoning phase**: opencode's architecture is: deterministic tool calls first (read files), then single LLM reasoning call over actual content. Never ask the LLM to guess what's in a file — read it first. This is the `_pre_bootstrap` → `_direct_answer` pattern.
