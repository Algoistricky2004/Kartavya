# Kartavya Agent OS — Complete End-to-End Guide

> **What this is:** A context-engineered agent harness that runs on local SLMs (3B–14B) and cloud LLMs.  
> Proves that smart context management — not model size — determines output quality.

---

## Quick Start

```cmd
:: Activate environment
Kartavya_env\Scripts\activate

:: Run a code fix task on a local model
python cli\main.py run "fix the bugs in demo/buggy_calculator.py" --provider ollama --model qwen2.5-coder:7b

:: Run a research task
python cli\main.py run "explain the observability module" --provider ollama --model llama3.2:3b

:: Compare models on same task
python cli\main.py run "fix the bugs in demo/buggy_calculator.py" --provider ollama --model llama3.2:3b
python cli\main.py run "fix the bugs in demo/buggy_calculator.py" --provider ollama --model deepseek-r1:7b
python cli\main.py run "fix the bugs in demo/buggy_calculator.py" --provider ollama --model qwen2.5-coder:7b
```

---

## Architecture Overview

```
Your task
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│                    AgentOS  (agent_os.py)                    │
│                                                             │
│  [1] INTAKE → [2] ROUTE → [3] SKILL → [4] MEMORY          │
│     ↓              ↓           ↓            ↓              │
│  [5] KNOWLEDGE → [6] PLAN → [7] EXECUTE → [8] VERIFY      │
│                                    ↓            ↓          │
│                              [9] PERSIST ←──────┘          │
└─────────────────────────────────────────────────────────────┘
         │                │                │
    ToolRegistry     LLMProvider      ObsManager
    (tools.py)    (providers.py)  (observability.py)
         │                │
    str_replace       Ollama/Groq/
    run_python        Anthropic API
    read_file
    find_files
    bash
```

---

## Stage-by-Stage Walkthrough

### Command
```cmd
python cli\main.py run "fix the bugs in demo/buggy_calculator.py" --provider ollama --model qwen2.5-coder:7b
```

---

### Stage 1 — INTAKE: Understand the task

**What happens:**  
`core/task_intake.py` analyzes the task text and extracts structured metadata.

**Output:**
```
[1] INTAKE  type=coding  complexity=moderate  risk=medium  scope=local  action_verbs=['fix']
```

**How it works:**
- `type=coding` — detected "fix" + filename with `.py` extension
- `complexity=moderate` — 7 words, references a specific file, has an action verb
- `risk=medium` — modifies files on disk (str_replace writes to your filesystem)
- `scope=local` — no web URLs, no external services
- `action_verbs=['fix']` — triggers coding task routing

**Code location:** `core/task_intake.py` → `process_task()`, `TaskIntake` dataclass

---

### Stage 2 — ROUTE: Choose model configuration

**What happens:**  
`core/model_router.py` looks up the model spec and computes context budgets.

**Output:**
```
[2] ROUTE  tier=slm  steps=8  planning=minimal
```

**For qwen2.5-coder:7b specifically:**
```
context_tokens:  32,768  (full context window)
supports_tools:  True    (native JSON tool calling)
prompt_format:   xml     (structured tool schemas injected)
max_output:      4,096   (generous for multi-line patches)
history_tokens:  8,192   (keeps 6+ turns of fix attempts)
```

**Context budget allocation (32,768 total):**
```
System prompt:    512 tk
Tool definitions: 1,024 tk
Memory:           512 tk
Task context:     256 tk
History:          8,192 tk  ← large: keeps full patch history
Current input:    2,048 tk
Output buffer:    4,096 tk
Safety margin:    512 tk
─────────────────────────
Used:            ~17,152 tk  (leaving 15,616 tk headroom)
```

**Code location:** `core/model_router.py` → `_MODEL_REGISTRY`, `route()`

---

### Stage 3 — SKILL: Match the right workflow

**What happens:**  
`skills/skill_registry.py` scores all available skills against the task text and picks the best match.

**Output:**
```
[3] SKILL  matched=patch_and_run  score=0.47
```

**How scoring works:**
```python
# patch_and_run.yaml triggers:
tags: [fix, bug, patch, debug, error, broken, failing, correct, repair, crash]

# Task: "fix the bugs in demo/buggy_calculator.py"
# Matched tags: "fix", "bug", "bugs"
# Score: tag_hits / total_tags * weight
```

**What the skill provides:**
- 4-step workflow: Read & Diagnose → Apply Patch → Run & Verify → Iterate
- Tool allowlist: `[str_replace, run_python, read_file, find_files, bash]`
- Step-level prompt hints: what to do at each step
- Success criteria: `["tests pass", "PASSED", "code runs without error"]`

**Skills available:**
```
skills/builtin/skills.yaml    ← original skills (research, debug, test_and_fix)
skills/builtin/patch_and_run.yaml  ← new code correction skill
```

**Code location:** `skills/skill_registry.py`, `skills/builtin/`

---

### Stage 4 — MEMORY: Recall past context

**What happens:**  
`core/memory_engine.py` searches 3 memory stores for relevant past runs.

**Output:**
```
[4] MEMORY  0 chunks from {task: 0, project: 0, user: 0, run: 0}
```

First run = no memory. After a successful run, the harness stores:
- What skill worked for this task type
- Which files were involved
- How many steps it took

Next run on a similar task retrieves these chunks and injects them into context.

**Code location:** `core/memory_engine.py`, `.agent_os/memory/`

---

### Stage 5 — KNOWLEDGE: External docs

**What happens:**  
Checks if the task needs external knowledge (README, docs, API reference).

**Output:**
```
[5] KNOWLEDGE  none needed
```

For coding tasks on local files, this is always skipped. For tasks like "fix the rate limit handling using Groq's docs", it would fetch the Groq API documentation.

**Code location:** `core/knowledge_retriever.py`

---

### Stage 6 — PLAN: Build execution steps

**What happens:**  
`core/planner.py` generates a minimal plan matched to the skill and model tier.

**Output:**
```
[6] PLAN  4 steps | depth=minimal | skill=patch_and_run
```

**What the plan looks like internally:**
```
Step 1: Read & Diagnose
  tools: [find_files, read_file, run_python]
  hint: "Find and read the target file. Run it to capture the current error."

Step 2: Apply Patch  
  tools: [str_replace]
  hint: "Use str_replace to fix ONLY the broken lines. Match indentation exactly."

Step 3: Run & Verify
  tools: [run_python]
  hint: "Run the file. [PASSED] → done. [FAILED] → read the new error, patch again."

Step 4: Iterate
  tools: [str_replace, run_python, read_file]
  hint: "Fix the next error. Repeat until [PASSED]."
```

For SLMs: `depth=minimal` (3–4 steps, short hints)  
For LLMs: `depth=standard` (5–8 steps, full context hints)

**Code location:** `core/planner.py`

---

### Stage 7 — EXECUTE: The core loop

**What happens:**  
This is where all the actual work happens. For coding tasks, `_code_correction_execute()` runs.

**Output (condensed):**
```
[7] EXECUTE  starting execution loop...
[CODE CORRECTION MODE]

[Pre-run demo\buggy_calculator.py]: [FAILED]
AssertionError: subtract(5,3): expected 2, got 8

── Code Step 1 ──
str_replace: subtract a+b → a-b
[RUN] [FAILED] (divide bug next)

── Code Step 2 ──
str_replace: divide guard
[RUN] [FAILED] (factorial bug next)

── Code Step 3 ──
str_replace: factorial base case 0 → 1
[RUN] [PASSED] ← done
```

**The actual flow:**

```
                    ┌─────────────────────────────────────┐
                    │       _code_correction_execute()     │
                    │                                     │
                    │  1. _code_bootstrap()               │
                    │     ├── find_files("*calculator*")  │
                    │     ├── read_file(calculator.py)    │
                    │     └── run_python → [FAILED] msg   │
                    │                                     │
                    │  2. Build system prompt (JSON format)│
                    │                                     │
                    │  3. First user message:             │
                    │     "Task: fix bugs...              │
                    │      FILE CONTENT: [actual code]    │
                    │      CURRENT ERROR: [actual error]  │
                    │      Use str_replace to fix."       │
                    │                                     │
                    │  4. LLM → JSON tool call            │
                    │     → str_replace(path, old, new)   │
                    │                                     │
                    │  5. AUTO-RUN run_python             │
                    │     [PASSED]? → return success      │
                    │     [FAILED]? → new error → step 4  │
                    └─────────────────────────────────────┘
```

**Key tools used:**

| Tool | What it does |
|------|-------------|
| `str_replace` | Targeted patch: find exact text, replace it. Indentation-aware. |
| `run_python` | Runs `.py` file, prepends `[PASSED]`/`[FAILED]` banner |
| `read_file` | Read current file state |
| `find_files` | Locate files by glob pattern, excludes venv/caches |

**str_replace indentation fix:**
```python
# Model sends (no leading spaces):
old_str = "return a / b  # BUG: no zero guard"
new_str = "if b == 0:\n    return None\nelse:\n    return a / b"

# Tool detects base indent = "    " (4 spaces before old_str in file)
# Applies it to continuation lines:
# Result in file:
#     if b == 0:
#         return None        ← 4+4=8 spaces ✓
#     else:                  ← 4 spaces ✓
#         return a / b       ← 4+4=8 spaces ✓
```

**Loop detection:**  
If the same `str_replace` fails twice in a row, the harness injects:
```
"You tried that EXACT str_replace 2 times. It fails because: [error].
 Include the function def line to make old_str unique. Example: ..."
```

**Code location:** `core/agent_os.py` → `_code_correction_execute()`, `_code_bootstrap()`, `_build_code_fix_system()`

---

### Stage 8 — VERIFY: Check the result

**What happens:**  
`core/verifier.py` checks the combined output against success signals and criteria.

**Output:**
```
[8] VERIFY  ✅ PASS
```

**Success signals checked:**
```python
_SUCCESS_PATTERNS = [
    r"\b(all tests? pass(ed)?|tests? (passed|succeed))\b",
    r"\[passed\]",      # run_python banner
    r"\[pass\]",        # individual test line
    r"all tests passed",
    r"\bpassed\b",
    r"\bsuccess(fully)?\b",
    r"\b(done|completed|finished)\b",
]
```

**Failure patterns also checked:**
```python
_FAILURE_PATTERNS = [
    r"\btraceback \(most recent\b",   → python_traceback (high)
    r"\bsyntax error\b",              → syntax_error (high)
    r"\berror: \b",                   → generic_error (medium)
    r"\bfailed\b.*\bwith\b.*\bexit\b" → non_zero_exit (high)
]
```

If verify fails: one retry with a focused sub-task.

**Code location:** `core/verifier.py`

---

### Stage 9 — PERSIST: Save to memory

**What happens:**  
Saves what worked to memory stores for future tasks.

**Output:**
```
[9] PERSIST  memory + skills updated
```

**What gets saved:**
- Task type + skill that worked
- Files modified
- Steps taken
- Token cost

Next time you run a similar task, Stage 4 retrieves this and the model starts with better context.

**Code location:** `core/memory_engine.py`, `.agent_os/memory/`

---

## Final Output

```
══════════════════════════════════════════════════════════════
OUTPUT
══════════════════════════════════════════════════════════════
Fixed! [PASSED]
==================================================
buggy_calculator.py self-test
==================================================
  [PASS] add(2, 3) = 5
  [PASS] subtract(5, 3) = 2
  [PASS] multiply(4, 3) = 12
  [PASS] divide(10, 2) = 5.0
  [PASS] divide(10, 0) = None
  [PASS] factorial(5) = 120
  [PASS] factorial(0) = 1
  [PASS] power(2, 8) = 256
All tests PASSED!
══════════════════════════════════════════════════════════════
steps=3  tokens=9,057  cost=$0.00  status=pass
skill_used=patch_and_run
```

---

## Model Comparison: Same Task, Different Models

```cmd
python cli\main.py run "fix the bugs in demo/buggy_calculator.py" --provider ollama --model llama3.2:3b
python cli\main.py run "fix the bugs in demo/buggy_calculator.py" --provider ollama --model deepseek-r1:7b
python cli\main.py run "fix the bugs in demo/buggy_calculator.py" --provider ollama --model qwen2.5-coder:7b
python cli\main.py run "fix the bugs in demo/buggy_calculator.py" --provider groq --model llama-3.3-70b-versatile
```

| Model | Steps | Tokens | Behavior |
|-------|-------|--------|----------|
| llama3.2:3b | 3–5 | ~6K | Needs ReAct format, no JSON tool calls |
| deepseek-r1:7b | 2–4 | ~15K | Uses `<think>` blocks, then clean JSON |
| qwen2.5-coder:7b | 3 | ~9K | Direct JSON, most reliable format |
| llama-3.3-70b | 2–3 | ~12K | Full LLM mode, rich planning |

**What context engineering does for the 3B model:**

Without it:
- Model answers from training data ("observability is like Prometheus...")
- Model runs `Done:` as a shell command
- `find_files` returns 781 venv files
- `str_replace` creates SyntaxErrors

With it:
- File content injected BEFORE first LLM call
- Tool call format shown explicitly per model
- Indentation auto-corrected on patches
- Loop detection breaks infinite retry at step 2
- Output truncation prevented via `PYTHONIOENCODING=utf-8`

---

## All CLI Commands

```cmd
:: Run a task
python cli\main.py run "TASK" --provider PROVIDER --model MODEL

:: Debug task classification
python cli\main.py intake "TASK"

:: Debug model routing
python cli\main.py route "TASK" --provider PROVIDER --model MODEL

:: List skills
python cli\main.py skills

:: Show memory
python cli\main.py memory

:: List Ollama models available
python cli\main.py models --provider ollama

:: Interactive chat
python cli\main.py chat --provider ollama --model qwen2.5-coder:7b

:: Dashboard
python cli\main.py dashboard

:: Run tests (no API key needed)
python tests\test_harness.py -v
```

---

## Supported Models

### Ollama (local, free, no rate limits)
```cmd
ollama pull llama3.2:3b          # fastest, 4K context
ollama pull deepseek-r1:7b       # reasoning model, thinks before acting
ollama pull qwen2.5-coder:7b     # best for code tasks, 32K context
ollama pull phi4                  # Microsoft, 14B, strong reasoning
ollama pull codellama             # Meta, code-focused
```

### Groq (cloud, fast inference, free tier available)
```
llama-3.1-8b-instant     # fast, free tier
llama-3.3-70b-versatile  # powerful
```

---

## File Structure

```
Kartavya/
├── cli/
│   └── main.py              ← Entry point, all CLI commands
├── core/
│   ├── agent_os.py          ← 9-stage pipeline, _code_correction_execute
│   ├── agent.py             ← Simple ReAct agent (used by orchestrator)
│   ├── task_intake.py       ← Task classification
│   ├── model_router.py      ← Model spec registry, context budgets
│   ├── skill_engine.py      ← Skill scoring and step config
│   ├── context_manager.py   ← Context window budget allocation
│   ├── planner.py           ← Plan generation
│   ├── memory_engine.py     ← Memory store (task/project/user/run)
│   ├── verifier.py          ← Output verification, success/fail detection
│   └── knowledge_retriever.py
├── tools/
│   └── tools.py             ← All tools: str_replace, run_python, bash, etc.
├── providers/
│   └── providers.py         ← LLM providers + parse_tool_calls
├── skills/
│   └── builtin/
│       ├── skills.yaml      ← Built-in skills
│       └── patch_and_run.yaml  ← Code correction skill
├── observability/
│   └── observability.py     ← Audit log, loop detection, hallucination check
├── demo/
│   └── buggy_calculator.py  ← Demo file with 3 intentional bugs
└── .agent_os/
    ├── memory/              ← Persisted memory stores
    └── audit/               ← Run audit logs (JSONL)
```

---

## Environment Setup

```cmd
:: Windows (PowerShell or cmd)
cd C:\Users\CHIRAG CHAWLA\Documents\Kartavya
Kartavya_env\Scripts\activate

:: Install deps
pip install groq anthropic openai pyyaml python-dotenv

:: .env file (create if missing)
echo GROQ_API_KEY=your_key_here > .env

:: Start Ollama (separate terminal)
ollama serve
```

---

## How Context Engineering Solves SLM Limitations

The central thesis of this project: **context quality beats model size.**

| Problem | Without CE | With CE |
|---------|-----------|---------|
| Model hallucinates file content | Answers from training data | File injected before first call |
| Model loops on same failed tool call | 13 identical retries | Caught at retry 2, new instruction sent |
| str_replace creates SyntaxError | Multi-line patch breaks indentation | Base indent auto-detected and applied |
| `find_files` returns 781 venv files | Floods context with noise | `_EXCLUDED_DIRS` filters non-project dirs |
| Model generates Unix commands on Windows | `find: not recognized` | PYTHONIOENCODING set, commands translated |
| `Done:` executed as shell command | `Done: is not recognized` | Parser checks Done before Bash in priority |
| deepseek-r1 thinks in `<think>` blocks | Parser misses the JSON | `<think>` stripped before parsing |
| Tool names/descriptions missing from context | Model doesn't know what tools exist | Unwrap `{"type":"function","function":{...}}` format |