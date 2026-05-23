# Kartavya — Agent OS

> A grounded, self-improving AI agent runtime that executes real tasks through a strict 9-stage pipeline — no hallucination, no guessing, every claim backed by a tool result.

---

## Table of Contents

- [What It Is](#what-it-is)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Architecture Overview](#architecture-overview)
- [The 9-Stage Pipeline](#the-9-stage-pipeline)
- [Orchestration & Multi-Agent Flow](#orchestration--multi-agent-flow)
- [File-by-File Reference](#file-by-file-reference)
  - [cli/main.py](#climainpy)
  - [core/agent_os.py](#coreagent_ospy)
  - [core/task_intake.py](#coretask_intakepy)
  - [core/model_router.py](#coremodel_routerpy)
  - [core/planner.py](#coreplannerpy)
  - [core/skill_engine.py](#coreskill_enginepy)
  - [core/memory_engine.py](#corememory_enginepy)
  - [core/knowledge_retriever.py](#coreknowledge_retrieverpy)
  - [core/verifier.py](#coreverifierpy)
  - [core/context_manager.py](#corecontext_managerpy)
  - [core/orchestrator.py](#coreorchestratorpy)
  - [core/agent.py](#coreagentpy)
  - [providers/providers.py](#providersproviderspy)
  - [tools/tools.py](#toolstoolspy)
  - [observability/observability.py](#observabilityobservabilitypy)
  - [skills/skill_registry.py](#skillsskill_registrypy)
  - [skills/builtin/skills.yaml](#skillsbuiltinSkillsyaml)
- [Skill System](#skill-system)
- [Memory System](#memory-system)
- [Supported Providers & Models](#supported-providers--models)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)

---

## What It Is

Kartavya is a local-first, provider-agnostic agent OS. Given any task in plain English, it:

1. **Classifies** the task — type, complexity, risk, scope — without touching a model
2. **Routes** to the right model tier (SLM vs LLM) and tool set for that task
3. **Retrieves** relevant skills, memory, and external knowledge before the first LLM call
4. **Plans** a concrete step list matched to the task type
5. **Executes** a grounded ReAct / native-tool-call loop where every claim must come from a tool result
6. **Verifies** the output against the plan's success criteria before accepting it
7. **Learns** — successful workflows are crystallized into reusable skills; facts and run outcomes land in a persistent memory store

It runs on Groq (free, fast), Ollama (local, offline), OpenAI, or Anthropic — swap with `--provider`.

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/your-repo/Kartavya.git
cd Kartavya
python -m venv Kartavya_env
# Windows:
Kartavya_env\Scripts\activate
# macOS/Linux:
source Kartavya_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
# Edit .env and paste your GROQ_API_KEY

# 4. Run a task
python cli/main.py run "explain the auth module" --provider groq --model llama-3.1-8b-instant
```

---

## CLI Reference

```
python cli/main.py <command> [options]
```

| Command | What it does |
|---------|-------------|
| `run "<task>"` | Execute a task through the full 9-stage pipeline |
| `chat` | Interactive multi-turn chat mode |
| `skills` | List all loaded skills with health and usage stats |
| `memory` | Print a JSON summary of all memory banks |
| `dashboard` | Show the observability dashboard (events, loop stats, cost) |
| `models --provider <p>` | List available models for a provider |
| `intake "<task>"` | Debug: show how the task intake processor classifies a task |
| `route "<task>"` | Debug: show the full model routing decision for a task |

**Common flags** (apply to all commands):

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `groq` | Provider: `groq`, `ollama`, `openai`, `anthropic` |
| `--model` | `llama-3.1-8b-instant` | Model name for the provider |
| `--max-steps` | `20` | Hard cap on execution loop steps |
| `--temperature` | `0.2` | Sampling temperature |
| `--no-stream` | off | Disable streaming responses |
| `--no-verify` | off | Skip output verification stage |
| `--no-skill-learn` | off | Skip writing new skills from successful runs |
| `--tools` | all | Comma-separated list of allowed tools |
| `--ollama-host` | `None` | Custom Ollama host (e.g. `http://remote:11434`) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  cli/main.py  — argument parsing, command dispatch                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ instantiates
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  core/agent_os.py  — AgentOS  (the single public entry point)       │
│                                                                     │
│  Stage 1  task_intake.py     ──► classify type / complexity / risk  │
│  Stage 2  model_router.py    ──► pick tier, tools, step budget      │
│  Stage 3  skill_engine.py    ──► retrieve matching skill template   │
│  Stage 4  memory_engine.py   ──► retrieve relevant past facts       │
│  Stage 5  knowledge_retriever.py ► pull repo context / web search  │
│  Stage 6  planner.py         ──► build concrete step plan           │
│  Stage 7  [execution loop]   ──► ReAct / native tool-call loop     │
│             ├── providers/providers.py  (LLM calls)                 │
│             ├── tools/tools.py          (tool execution)            │
│             └── observability/observability.py (loop detection)     │
│  Stage 8  verifier.py        ──► verify output vs success criteria  │
│  Stage 9  memory_engine.py   ──► persist run outcome + new skills   │
└─────────────────────────────────────────────────────────────────────┘
                           │ for compound tasks
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  core/orchestrator.py  — multi-agent coordinator                    │
│    ├── TaskDecomposer   ──► break task into subtasks with roles     │
│    ├── specialist Agent instances (researcher / coder / tester …)  │
│    └── ResultMerger     ──► combine outputs into final answer       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The 9-Stage Pipeline

Every call to `agent.run_task()` runs exactly these stages in order.

### Stage 1 — Task Intake (`task_intake.py`)

Pure heuristic, zero LLM calls. Parses the raw task string and outputs a `TaskIntake` dataclass that every downstream stage reads.

Extracts:
- **Type** — `coding | research | automation | ops | data | creative | mixed`
- **Complexity** — `simple | moderate | complex | compound`
- **Risk** — `low | medium | high | critical`
- **Scope** — `local | repo | system | external`
- Action verbs, file references, tech stack, named entities
- Planning flags: `needs_bash`, `needs_file_write`, `is_multi_step`, `needs_external_knowledge`
- `skill_query` — compact keyword string used for skill and memory retrieval

### Stage 2 — Model-Aware Routing (`model_router.py`)

Looks up the model in a registry of ~20 pre-configured `ModelSpec` entries and produces a `ModelRuntimeConfig` that controls the entire run:

- **Tier** — SLM (≤14B) vs LLM. SLMs get tighter step budgets, fewer tools, minimal planning depth, and a ReAct/XML prompt format instead of native tool calling.
- **Step budget** — varies by complexity: e.g. SLM + moderate → 8 steps; LLM + complex → 25 steps.
- **Allowed tools** — filtered per task type and tier. `task_done` is always included so the agent can signal completion.
- **Context budget** — per-zone token allocations fed to the context manager.
- **Prompt format** — `react` (Thought/Bash/Done text), `xml`, or `native` (provider tool-call API).

### Stage 3 — Skill Retrieval (`skill_engine.py`)

TF-IDF similarity search across all loaded skills. Returns the best matching `Skill` (if score ≥ 0.08) and adapts it into a prompt fragment (step list, tool hints, success criteria) sized to the model tier.

### Stage 4 — Memory Retrieval (`memory_engine.py`)

Queries five memory banks — `task`, `project`, `user`, `run`, `errors` — for chunks relevant to the current task. Returns ranked `MemEntry` objects formatted as context lines. SLMs receive fewer, shorter chunks.

### Stage 5 — External Knowledge (`knowledge_retriever.py`)

Triggered only when `intake.needs_external_knowledge` is `True`. Checks repo files (README, git log) first; falls back to a DuckDuckGo search if the task needs current or web-based information.

### Stage 6 — Planning (`planner.py`)

Generates a `Plan` dataclass with a concrete step list and success criteria. SLMs get a minimal 2–4 step plan; LLMs get task-type-specific plans (coding, research, automation, ops, data). If a skill was matched, the plan is seeded from the skill's step list. Prior run summaries from memory are used to avoid repeating known mistakes.

### Stage 7 — Execution Loop (`agent_os.py`)

The core ReAct / tool-call loop:

```
while step < step_limit:
    1. Build context (system prompt + history + current input) within token budget
    2. Call LLM via provider
    3. Parse response — check for Done: signal or tool calls
    4. If Done: → capture output and break
    5. For each tool call:
       a. Loop-detection check (warn / break on repeat)
       b. Execute tool via ToolRegistry
       c. Quick-verify tool output for hallucination markers
       d. Store result in run_state
    6. Compress tool results into next step's input
    7. Record turn in context history (using original pre-mutation input)
```

Loop safety: four-level loop detector (`info → warn → error → critical`) aborts or redirects when it detects repeated identical tool calls or ping-pong patterns.

### Stage 8 — Verification (`verifier.py`)

Checks the final output against the plan's success criteria, tool outputs (grounding check), step count, and confidence estimate. Returns a `VerificationResult` with a `PASS | PARTIAL | FAIL` status and a retry strategy if it fails. If `retry_viable` is `True`, the agent runs once more with the retry hint prepended.

### Stage 9 — Memory + Skill Update (`memory_engine.py`, `skill_engine.py`)

- Run outcome (success, summary, duration, tokens) is saved to the `run` memory bank.
- File contents seen during the run are saved as `project` facts.
- Any errors are saved for future avoidance.
- On success, if the run used ≥ 2 tool calls, a new `Skill` is crystallized from the tool call sequence and saved to `.agent_os/skills/`.

---

## Orchestration & Multi-Agent Flow

Activated automatically when `intake.complexity == COMPOUND` (or manually via `core/orchestrator.py`).

```
Orchestrator.coordinate(task)
    │
    ├─► TaskDecomposer.decompose_heuristic(task)
    │       Detects pattern: fix+test, implement+verify, research+code, or generic 2-phase
    │       Returns a DecompositionResult with a list of SubTask objects
    │       Each SubTask has: role, description, depends_on, allowed_tools, max_steps
    │
    ├─► For each SubTask (sequential):
    │       Orchestrator._create_agent(role)  →  specialist Agent instance
    │       Agent.run_task(subtask.description)
    │       SubTask.result = TaskResult
    │
    ├─► ResultMerger.merge(task, subtasks)
    │       Conflict detection across subtask outputs
    │       Human-in-the-loop gate if conflicts found (optional)
    │       Synthesizes final answer
    │
    └─► OrchestratorResult (final_output, success, subtask_results, cost_usd, …)
```

**Specialist roles and their tool sets:**

| Role | Allowed tools | Responsibility |
|------|--------------|----------------|
| `researcher` | read_file, grep, find_files, list_dir, web_search | Gather facts only, no writes |
| `coder` | read_file, write_file, append_file, grep, find_files | Write and edit code |
| `tester` | bash, read_file, grep, find_files | Run commands, verify results |
| `reviewer` | read_file, grep, find_files | Review correctness, give PASS/NEEDS_FIX |
| `general` | all tools | Fallback for unclassified subtasks |

---

## File-by-File Reference

---

### `cli/main.py`

CLI entry point. Parses arguments, dispatches to command functions, instantiates `AgentOS`.

| Function | Summary |
|----------|---------|
| `_api_key(provider)` | Reads the correct API key env var for a given provider name. Returns `None` if not set. |
| `_make_agent(args)` | Builds an `AgentOSConfig` from parsed CLI args and instantiates `AgentOS`. |
| `cmd_run(args)` | Runs a single task through the full pipeline, prints output, step count, token count, and cost. |
| `cmd_chat(args)` | Starts an interactive REPL. Supports inline commands: `dashboard`, `memory`, `skills`, `exit`. |
| `cmd_skills(args)` | Lists all loaded skills with version, task type, usage count, success rate, and tags. |
| `cmd_memory(args)` | Prints a JSON export of all memory bank summaries. |
| `cmd_dashboard(args)` | Renders the observability dashboard and memory/skill counts. |
| `cmd_models(args)` | Calls the provider's `list_models()` and prints available model names. |
| `cmd_intake(args)` | Debug command: runs task intake only and prints the full `TaskIntake` as JSON. |
| `cmd_route(args)` | Debug command: runs intake + routing and prints the `ModelRuntimeConfig` description. |
| `_add_model_args(p)` | Attaches the standard model/provider/behaviour flags to any sub-parser. |
| `build_parser()` | Constructs the full `argparse` parser tree with all subcommands. |
| `main()` | Entry point: parses args, dispatches to the correct `cmd_*` function. |

---

### `core/agent_os.py`

The central runtime. Owns all subsystems and runs the full 9-stage pipeline for every task.

**Dataclasses**

| Class | Summary |
|-------|---------|
| `AgentOSConfig` | All configuration for one agent instance: model, provider, directories, behaviour flags, step/budget limits. |
| `TaskResult` | Output of a completed task: output text, steps taken, tokens used, cost, errors, skill used, verify status. |
| `RunState` | Mutable per-run state: intake, route config, plan, step counter, token counts, tool history, tool outputs, final output, done flag. |

**Module-level functions**

| Function | Summary |
|----------|---------|
| `_build_system_prompt(config, plan, skill_hint, memory_chunks, knowledge_chunks, role)` | Assembles the full system prompt for a run. Tier-aware: SLMs get a stripped 4-rule prompt; LLMs get full operating principles, memory, knowledge, skill hint, and plan. |
| `_build_env_context(max_chars)` | Generates a compact environment snapshot (OS, CWD, top-level files, git status) using cross-platform `Path.rglob`. Injected into the system prompt. |

**`AgentOS` methods**

| Method | Summary |
|--------|---------|
| `__init__(config, api_key, ollama_host)` | Initialises all subsystems: provider, skill engine, memory engine, knowledge retriever, planner, verifier, observability manager, tool registry, context manager, run log. |
| `run_task(raw_task, task_id)` | **Public entry point.** Runs the full 9-stage pipeline and returns a `TaskResult`. |
| `_stage_intake(raw_task)` | Stage 1: delegates to `process_task()`. |
| `_stage_route(intake)` | Stage 2: calls `router_route()`, enforces config's `max_steps` ceiling. |
| `_stage_skill_retrieval(intake, config)` | Stage 3: retrieves top-3 skill matches, returns best if score ≥ 0.08, plus the prompt hint. |
| `_stage_memory_retrieval(intake, config)` | Stage 4: clears task memory, retrieves relevant chunks across all banks, returns formatted context lines. |
| `_stage_knowledge(intake)` | Stage 5: calls `knowledge_retriever.retrieve_if_needed()`, returns formatted context blocks. |
| `_stage_plan(intake, config, skill_match)` | Stage 6: passes intake + routing config + skill match + prior run summary to the planner. |
| `_stage_execute(run_state, task_id, trace_id, skill_hint, memory_chunks, knowledge_chunks, retry_hint)` | Stage 7: the full execution loop — initialises context manager, builds system prompt, loops over LLM calls and tool executions until `Done:` or step limit. |
| `_stage_retry(run_state, …)` | Resets `run_state.done` and calls `_stage_execute` again with a retry hint prepended to the system prompt. |
| `_stage_verify(run_state, result, plan)` | Stage 8: calls `verifier.verify()` with the output, tool outputs, success criteria, and step stats. |
| `_stage_update_memory(run_state, result, intake)` | Stage 9a: saves run outcome, file contents seen, and any errors to memory banks. |
| `_stage_update_skills(run_state, result, intake)` | Stage 9b: if run succeeded and used ≥ 2 tools, crystallises a new skill from the tool history. |
| `_extract_best_output(last_content, run_state)` | Fallback output extractor when `task_done` was never called — uses last LLM content or last tool output. |
| `_print_header(task, task_id)` | Prints the styled run header with model, provider, task, and ID. |
| `_print_stage(n, name, detail)` | Prints a single stage progress line. |
| `_print_footer(result)` | Prints the styled run footer with step count, tokens, cost, time, and errors. |
| `dashboard()` | Prints the observability dashboard plus memory stats and skill count. |
| `memory_export()` | Returns a JSON string of the full memory export summary. |
| `list_skills()` | Returns a list of skill summary dicts from the skill engine. |

---

### `core/task_intake.py`

Stage 1: pure heuristic task classification. No LLM calls, deterministic, runs in <1 ms.

**Enums:** `TaskType`, `Complexity`, `RiskLevel`

**Dataclass:** `TaskIntake` — the full classification output passed to every downstream stage.

| Method | Summary |
|--------|---------|
| `TaskIntake.summary()` | Returns a one-line string: `type=X complexity=Y risk=Z scope=W`. |
| `TaskIntakeProcessor.process(raw_task)` | Main entry: normalises → classifies → assesses risk/complexity → extracts signals → builds skill query. Returns a fully populated `TaskIntake`. |
| `_normalize(task)` | Strips extra whitespace and trailing punctuation. Preserves case. |
| `_classify_type(tl)` | Scores the task against 5 pattern tables (coding, research, automation, ops, data). Returns `MIXED` if top two scores are within 70% of each other; returns the winner otherwise. |
| `_score(text, patterns)` | Counts regex pattern hits and returns a normalised fraction (0–1). |
| `_assess_risk(tl)` | Checks high-risk (delete, production, secrets) then medium-risk (writes, git push) patterns. Read-only tasks get `LOW`. |
| `_assess_complexity(tl)` | Checks compound signals (multi-phase language), complex signals (broad scope), then falls back to word count. Short single-verb tasks get `SIMPLE`. |
| `_extract_action_verbs(tl)` | Regex-extracts up to 6 deduplicated action verbs (write, fix, run, deploy, …). |
| `_extract_file_refs(task)` | Extracts file paths by extension and quoted path patterns. Returns up to 8. |
| `_extract_tech_stack(tl)` | Matches against a dict of 10 technology patterns. Returns up to 5 matched names. |
| `_extract_entities(task)` | Extracts CamelCase identifiers, long snake_case names, and backtick-quoted strings. Returns up to 10. |
| `_assess_scope(tl)` | Returns `repo` (entire codebase), `system` (OS-level), `external` (API/HTTP), or `local` (default). |
| `_needs_external(tl)` | Returns `True` if the task references current dates, online docs, or web lookups. |
| `_needs_bash(tl, task_type)` | Returns `True` for automation/ops tasks or tasks mentioning run/execute/install/test. |
| `_needs_file_write(tl, task_type)` | Returns `True` for coding tasks that create/fix/update, or any task that mentions writing a file. |
| `_build_skill_query(intake)` | Combines action verbs + task type + tech stack + entities + first 8 words of the task into a single keyword string for TF-IDF retrieval. |
| `process_task(raw_task)` | Module-level convenience wrapper around the shared `TaskIntakeProcessor` singleton. |

---

### `core/model_router.py`

Stage 2: maps model name + task intake to a fully-specified `ModelRuntimeConfig`. No LLM calls.

**Key dataclasses/enums:** `ModelTier` (SLM/LLM), `ModelSpec` (per-model capabilities), `ModelRuntimeConfig` (per-run settings).

| Function/Method | Summary |
|----------------|---------|
| `ModelRouter.lookup_spec(model)` | Pattern-matches the model name against `_MODEL_REGISTRY` (~20 entries). Falls back to `_DEFAULT_SLM_SPEC` or `_DEFAULT_LLM_SPEC` based on name heuristics (70b, claude, gpt-4). |
| `ModelRouter.route(model, provider, intake, override_max_steps, override_allowed_tools)` | Core routing function. Determines tier, computes step budgets per complexity, selects allowed tools (filtered by `_TASK_TOOL_BIAS` and `_SLM_CORE_TOOLS`), sets planning depth, and returns a `ModelRuntimeConfig`. |
| `ModelRouter.describe(config)` | Returns a human-readable multi-line summary of the routing decision for the `route` debug command. |
| `route(model, provider, intake, …)` | Module-level convenience wrapper around `ModelRouter().route(…)`. |

**Key constants:**

- `_SLM_CORE_TOOLS` — the restricted tool set for small models: `bash, read_file, write_file, list_dir, grep, find_files, task_done, memory_read`
- `_LLM_FULL_TOOLS` — the full tool set for large models
- `_TASK_TOOL_BIAS` — per-task-type preferred tool lists (all include `task_done`)

---

### `core/planner.py`

Stage 6: generates a concrete `Plan` from intake + routing config + skill match.

**Dataclasses:** `PlanStep` (id, description, tools, prompt_hint), `Plan` (steps, success_criteria, depth, skill_name, token_estimate).

| Method | Summary |
|--------|---------|
| `Plan.to_prompt_fragment()` | Formats the plan as a numbered step list for injection into the system prompt. |
| `Plan.estimated_steps` | Property: returns `len(steps)`. |
| `Planner.generate(intake, config, skill_match, prior_run_summary)` | Main entry: selects `_plan_from_skill`, `_slm_plan`, or `_plan_heuristic` based on tier and skill match. |
| `_plan_from_skill(skill, intake, config)` | Converts a matched `Skill`'s steps into `PlanStep` objects. Truncates to `config.max_plan_steps`. |
| `_plan_heuristic(intake, config, prior_run_summary)` | Dispatches to a task-type-specific plan builder. |
| `_slm_plan(intake, config)` | Generates a minimal 2–4 step plan suitable for SLMs with small context windows. |
| `_coding_plan(intake, config, prior)` | 5-step plan: find files → read → identify → implement → verify. Adapts for bug-fix vs new feature. |
| `_research_plan(intake, config, prior)` | 4-step plan: find relevant files → read → synthesise → report. |
| `_automation_plan(intake, config, prior)` | 3-step plan: check environment → execute → verify output. |
| `_ops_plan(intake, config, prior)` | 4-step plan: assess state → plan change → apply → validate. |
| `_data_plan(intake, config, prior)` | 4-step plan: explore → query/process → validate → present. |
| `_generic_plan(intake, config, prior)` | Fallback 3-step plan: gather information → process → complete. |
| `_verify_tools(intake, config)` | Returns a list of tools that should be used for the verification step, matched to task type. |

---

### `core/skill_engine.py`

Stage 3 (retrieval) and Stage 9b (creation). Manages the full lifecycle of skills.

**Dataclasses:** `SkillStep`, `Skill` (name, description, tags, steps, success_criteria, usage_count, success_rate), `SkillMatch` (skill, score, explanation).

| Method | Summary |
|--------|---------|
| `Skill.success_rate` | Property: `success_count / use_count` or `1.0` if unused. |
| `Skill.is_healthy` | Property: `True` if success rate ≥ 30% or fewer than 3 uses. |
| `Skill.record_use(success)` | Increments use count and optionally success count. |
| `Skill.to_dict()` | Serialises the skill to a dict for YAML/JSON persistence. |
| `SkillEngine.__init__(skill_dirs, generated_dir)` | Loads skills from all provided directories and the generated skills directory. Builds TF-IDF document frequency table. |
| `_load_from_dir(directory)` | Walks a directory for `.yaml` skill files and registers each parsed skill. |
| `_from_dict(d, source_file)` | Parses a raw skill dict into a `Skill` dataclass with defaults and validation. |
| `_register(skill)` | Adds a skill to the registry, deduplicating by name (keeps highest version). |
| `_rebuild_df(skills)` | Rebuilds the TF-IDF document frequency table from all registered skills. |
| `_version_tuple(v)` | Parses a semver string into a comparable tuple. |
| `retrieve(query, task_type, top_k)` | TF-IDF + tag-boost retrieval. Tokenises query, scores all skills, applies a task-type tag bonus, returns top-k `SkillMatch` objects sorted by score. |
| `_explain_match(skill, query_tokens, score)` | Generates a human-readable explanation of why a skill was matched (shared tokens, tag overlap). |
| `adapt_for_prompt(skill, task, max_steps, depth)` | Renders a skill as a prompt fragment. `shallow` depth: just step names. `deep`: full step descriptions with tool lists and hints. Truncated to `max_steps`. |
| `create_from_workflow(task, tool_calls, outcome, task_type, success)` | Crystallises a successful tool-call sequence into a new `Skill`. Groups consecutive same-tool calls, generates a name and tags, estimates token cost. |
| `save_skill(skill)` | Serialises a skill to YAML in the generated skills directory. Returns `True` on success. |
| `bump_version(skill)` | Increments the patch version of a skill (e.g. `1.0.0` → `1.0.1`). |
| `_update_skill_stats(skill, success)` | Updates use count and success count for a skill after a run. |
| `record_use(skill_name, success)` | Looks up a skill by name and calls `_update_skill_stats`. |
| `_describe_tool_call(tool, args)` | Returns a short human-readable description of a tool call (e.g. `"grep for 'auth' in src/"`). |
| `_generate_skill_name(task)` | Generates a snake_case skill name from the first 4–5 meaningful words of the task. |
| `_generate_tags(task, tools)` | Extracts keyword tags from the task text and tool names used. |
| `list_skills()` | Returns a list of summary dicts for the `skills` CLI command. |
| `get_skill(name)` | Looks up a skill by exact name. Returns `None` if not found. |
| `count` | Property: number of registered skills. |

---

### `core/memory_engine.py`

Stage 4 (retrieval) and Stage 9a (persistence). Five-bank persistent memory store with TF-IDF retrieval.

**Enums:** `MemoryBank` (task, project, user, run, errors), `MemoryPriority` (high, normal, low)

**Dataclasses:** `MemEntry` (id, content, bank, priority, tags, created_at, ttl_hours, use_count)

| Method | Summary |
|--------|---------|
| `MemEntry.is_expired()` | Returns `True` if the entry has a TTL and it has elapsed. |
| `MemEntry.relevance(query)` | TF-IDF-style relevance score against a query string. Boosts high-priority entries. |
| `MemEntry.to_context_line(max_chars)` | Formats the entry as a compact context line: `[bank] content (tags)`. |
| `MemEntry.to_dict() / from_dict()` | Serialise/deserialise to/from a plain dict for JSON persistence. |
| `MemoryRetrieval.to_context_chunks(max_per_bank, max_chars_each)` | Formats retrieval results into a flat list of context strings, capped per bank. |
| `BankStore.__init__(bank, persist_path)` | Initialises one memory bank with an in-memory store and an optional JSON persistence path. Loads existing entries on init. |
| `BankStore.add(entry)` | Adds an entry, evicts the lowest-scoring entry if at capacity (500 entries). |
| `BankStore.retrieve(query, top_k, min_score)` | Scores all non-expired entries against the query, returns the top-k above `min_score`. |
| `BankStore.get_all_recent(n)` | Returns the `n` most recently created entries. |
| `BankStore.upsert(entry_id, content, …)` | Updates an existing entry's content and timestamp if it exists; creates it otherwise. |
| `BankStore.delete(entry_id)` | Removes an entry by ID. |
| `BankStore.clear()` | Removes all entries from the bank. |
| `BankStore._evict()` | Removes the single lowest-relevance entry when the bank is over capacity. |
| `BankStore._load() / _save()` | JSON persistence: load from disk on init, save after every write. |
| `BankStore.count` | Property: current number of entries. |
| `MemoryEngine.__init__(agent_id, memory_dir)` | Creates one `BankStore` per bank, wired to per-agent JSON files under `memory_dir`. |
| `save_task_fact(content, tags, priority)` | Saves a fact to the `task` bank with a 24-hour TTL. |
| `save_verified_fact(content, tags)` | Saves a grounded (tool-verified) fact to the `task` bank with high priority. |
| `save_project_fact(content, tags, priority)` | Saves a long-lived project-level fact (no TTL). |
| `save_user_pref(content, tags)` | Saves a user preference to the `user` bank with 720-hour TTL. |
| `save_run_outcome(task_id, task_text, success, summary, duration_ms, tokens_used, tags)` | Saves a run result summary to the `run` bank for future prior-run retrieval. |
| `save_error(error_msg, tool, tags)` | Saves an error to the `errors` bank for future avoidance. |
| `retrieve_for_context(query, max_chunks, max_per_bank)` | Queries all five banks and returns a `MemoryRetrieval` with ranked results per bank. |
| `get_prior_run(task_text, top_k)` | Retrieves the most similar past run summaries from the `run` bank. |
| `get_errors()` | Returns recent entries from the `errors` bank. |
| `get_user_prefs()` | Returns all entries from the `user` bank. |
| `clear_task_memory()` | Clears the `task` bank (called at the start of each new run). |
| `export_summary()` | Returns a dict summarising all banks: count, recent entries, errors. |
| `stats()` | Returns a per-bank dict of `{bank: entry_count}`. |

---

### `core/knowledge_retriever.py`

Stage 5: pulls grounding context from the repo or the web when the task needs current or external information.

**Dataclass:** `KnowledgeChunk` (source, content, relevance, chunk_type)

| Method | Summary |
|--------|---------|
| `KnowledgeChunk.to_context_block(max_chars)` | Formats a chunk as `[Source: X]\nContent…` for injection into the system prompt. |
| `KnowledgeRetriever.__init__(cwd)` | Sets the working directory for repo-based lookups. |
| `retrieve_if_needed(task, needs_external, tech_stack, max_chunks)` | Gate function: returns repo context if `needs_external` is `False`; tries repo first then falls back to web search if needed. |
| `_get_repo_context(task, max_chunks)` | Reads README, git log, and any file in the repo whose name is mentioned in the task. Returns relevant chunks. |
| `_find_readme()` | Looks for `README.md`, `README.rst`, or `README.txt` in the CWD. |
| `_read_file_safe(path, max_chars)` | Reads a file with error replacement encoding, truncated to `max_chars`. |
| `_get_git_log(max_lines)` | Runs `git log --oneline -N` and returns the output. Returns empty string on failure. |
| `_is_relevant(content, task)` | Checks if at least 2 words from the task appear in the content (case-insensitive). |
| `_web_search(task, tech_stack, max_chunks)` | Queries DuckDuckGo's JSON API with a constructed query. Parses `AbstractText` and `RelatedTopics` into `KnowledgeChunk` objects. |
| `_build_search_query(task, tech_stack)` | Builds a focused DuckDuckGo query from the task's first 10 words and tech stack. |

---

### `core/verifier.py`

Stage 8: validates the agent's output before it is accepted.

**Enums/Dataclasses:** `VerifyStatus` (PASS, PARTIAL, FAIL, SKIP), `VerifyFlag` (issue description + severity), `VerificationResult` (status, flags, confidence, retry_viable, retry_strategy)

| Method | Summary |
|--------|---------|
| `VerificationResult.passed` | Property: `True` if status is PASS or PARTIAL. |
| `VerificationResult.summary()` | Returns a one-line summary string for stage-progress printing. |
| `Verifier.verify(agent_response, tool_outputs, success_criteria, task_text, steps_taken, max_steps)` | Main verification entry. Checks: output non-empty, success criteria met, output grounded in tool outputs, step budget not exhausted, confidence threshold. Returns a `VerificationResult`. |
| `_criterion_met(criterion, text)` | Checks if a single success criterion phrase is satisfied by the output text using keyword matching. |
| `_estimate_confidence(response, tool_outputs, criteria_met, total_criteria)` | Heuristic confidence score (0–1) based on output length, tool grounding, and criteria coverage. |
| `_suggest_retry(flags, criteria_missed, steps_taken, max_steps)` | Generates a targeted retry instruction based on what failed: missing criteria, hallucination markers, empty output, etc. |
| `quick_check_tool_output(tool_name, output)` | Fast sanity check on a raw tool result. Flags hallucination-indicator phrases, error keywords in bash output, and suspiciously short results. Returns `(ok: bool, description: str)`. |

---

### `core/context_manager.py`

Manages token-budget-aware message assembly for every LLM call. Prevents context overflow across all six budget zones.

**Key classes:** `ContextBudget` (per-zone token allocations), `Turn` (a single user/assistant exchange), `ConversationHistory` (sliding-window history with summary compression), `ToolResultCompressor`, `ContextSnapshot` (per-call audit record), `ContextManager`

| Method | Summary |
|--------|---------|
| `ContextBudget.total_allocated` | Property: sum of all zone allocations. |
| `ContextBudget.utilization_pct` | Property: `total_allocated / total_context * 100`. |
| `Turn.to_api_message()` | Converts a `Turn` to an OpenAI-format `{"role": …, "content": …}` dict. |
| `ToolResultCompressor.compress(tool_name, raw_result, task_hint)` | Trims tool output per-tool: bash gets first/last lines + exit code; read_file gets first N lines; grep/search gets top matches. Returns `(compressed, truncation_note)`. |
| `ConversationHistory.add_turn(turn)` | Appends a turn; if the sliding window is full, summarises and evicts the oldest turn. |
| `ConversationHistory.build_messages()` | Builds the message list from summaries (oldest, compressed) + verbatim recent turns, within the history token budget. |
| `ContextManager.__init__(model)` | Loads the budget preset for the model from `BUDGET_PRESETS`, initialises history and compressor. |
| `ContextManager.build(system_prompt, current_input, task_context, memory_chunks, tool_definitions, tool_results, step_hint)` | **Core method.** Assembles the full message list for one LLM call across 7 budget zones: system prompt → tool defs → memory → task context → tool results → current input → history. Performs emergency history trim if headroom goes negative. Returns `(messages, ContextSnapshot)`. |
| `ContextManager.record_turn(user_content, assistant_content, tool_name, tool_args)` | Records a completed exchange into conversation history. Must be called with the **original** `current_input` (before it is mutated to tool results). |
| `ContextManager.get_latest_snapshot()` | Returns the `ContextSnapshot` from the most recent `build()` call. |
| `ContextManager.get_stats()` | Returns aggregate stats: total calls, average tokens, average headroom, total truncations. |

---

### `core/orchestrator.py`

Multi-agent coordinator for `COMPOUND` complexity tasks.

**Dataclasses:** `SubTask` (role, description, depends_on, result), `DecompositionResult`, `OrchestratorResult`, `OrchestratorConfig`

| Class/Method | Summary |
|-------------|---------|
| `SubTask.succeeded` | Property: `True` if `result` is set and `result.success` is `True`. |
| `TaskDecomposer.decompose_heuristic(task)` | Pattern-matches the task to a known multi-phase plan (fix+test, implement+verify, research+code, generic 2-phase). Falls back to a generic plan if no pattern matches. |
| `TaskDecomposer.decompose_with_llm(task, provider, model, api_key)` | Uses the LLM to decompose the task into subtasks with roles. Parses the structured output and falls back to `decompose_heuristic` on parse failure. |
| `TaskDecomposer._fix_and_test_plan(task)` | 2-subtask plan: coder fixes the bug; tester verifies the fix. |
| `TaskDecomposer._implement_and_verify_plan(task)` | 2-subtask plan: coder implements; reviewer verifies correctness. |
| `TaskDecomposer._research_and_code_plan(task)` | 2-subtask plan: researcher gathers context; coder implements using the research. |
| `TaskDecomposer._generic_two_phase_plan(task, reason)` | Fallback 2-phase plan: researcher then coder. |
| `ResultMerger.merge(task, subtasks)` | Combines subtask outputs into a final answer. Detects conflicts, gates on human approval if needed, synthesises a summary. |
| `ResultMerger.detect_conflicts(subtasks)` | Flags contradictions across subtask outputs (e.g. one says PASS, another says FAIL). |
| `Orchestrator.__init__(config, api_key, ollama_host)` | Initialises the orchestrator with config and a shared observability manager. |
| `Orchestrator.coordinate(task)` | Main entry: decomposes the task, runs each subtask sequentially in a specialist agent, merges results, returns `OrchestratorResult`. |
| `Orchestrator._create_agent(role, agent_id_suffix)` | Instantiates a specialist `Agent` with the role's tool restrictions, temperature, and step limit. |

---

### `core/agent.py`

The legacy single-agent runtime (pre–Agent OS). Kept for the orchestrator's specialist worker instances.

Contains `Agent`, `AgentConfig`, and a simpler execution loop that the orchestrator uses when spawning role-specialised sub-agents (researcher, coder, tester, reviewer).

---

### `providers/providers.py`

Unified LLM provider abstraction. All providers implement the same `LLMProvider` interface.

**Dataclass:** `LLMResponse` (content, tokens_in, tokens_out, finish_reason, tool_calls, raw)

| Function/Class | Summary |
|---------------|---------|
| `parse_tool_calls(content)` | Parses tool calls from model text output. Handles 5 formats in priority order: ReAct bash block, ReAct inline bash, Done/Answer/Final signal, XML `<tool>` tags, raw JSON `{"name":…,"arguments":…}`. |
| `_safe_json(raw)` | Attempts `json.loads`; on failure tries single→double quote fix and trailing-comma strip. Returns a dict with `_parse_error` key on complete failure. |
| `LLMProvider.complete_with_retry(messages, model, max_retries, **kwargs)` | Base class method: calls `complete()` with exponential backoff on transient errors (rate limits, server errors). Raises on non-retryable errors. |
| `GroqProvider.complete(messages, model, …)` | Groq API call with streaming support. Accumulates streaming deltas including `tool_calls`. Falls back to `parse_tool_calls()` on text content if no native tool calls. |
| `GroqProvider.list_models()` | Returns a list of model ID strings available on the Groq account. |
| `OllamaProvider.complete(messages, model, …)` | Ollama API call via HTTP (no SDK dependency). Handles the Ollama chat format and parses tool calls from text. |
| `OllamaProvider.list_models()` | Calls the Ollama `/api/tags` endpoint and returns model names. |
| `OpenAIProvider.complete(messages, model, …)` | OpenAI-compatible call. Extracts native function calls if present. |
| `OpenAIProvider.list_models()` | Returns available OpenAI model IDs. |
| `AnthropicProvider.complete(messages, model, …)` | Anthropic Messages API call. Maps between Anthropic's `tool_use` block format and the internal tool-call dict. |
| `AnthropicProvider.list_models()` | Returns a hardcoded list of current Claude model IDs. |
| `_to_openai_tools(tools)` | Converts the internal tool definition format (already wrapped as `{"type":"function","function":{…}}` or flat) to OpenAI-compatible tool schemas. Handles both pre-wrapped and flat input. |
| `get_provider(name, api_key, host)` | Factory: returns the correct `LLMProvider` subclass for `groq`, `ollama`, `openai`, or `anthropic`. |

---

### `tools/tools.py`

All tool implementations behind a single `ToolRegistry`. Every tool result is compressed to fit the context window.

**Dataclass:** `ToolResult` (success, output, error, metadata)

| Method | Summary |
|--------|---------|
| `ToolRegistry.__init__(allowed, require_approval)` | Sets up the tool handler map. `allowed=None` means all tools are permitted. |
| `ToolRegistry.is_allowed(tool_name)` | Returns `True` if the tool is in the allowed set (or no restriction is set). |
| `ToolRegistry.all_tool_names()` | Returns the list of all registered tool names. |
| `ToolRegistry.requires_approval(tool_name)` | Returns `True` if the tool requires human approval before execution. |
| `ToolRegistry.execute(tool_name, args)` | Dispatches to the handler, truncates output to `MAX_OUTPUT[tool_name]`, and returns a `ToolResult`. |
| `ToolRegistry.get_definitions_for_step(step, task)` | Returns only the tools likely needed for a given step, inferred from task keywords. Optimises context window usage. |
| `ToolRegistry.get_definition(allowed)` | Returns full OpenAI-format tool schemas for all (or filtered) tools. |
| `_bash(args)` | Runs a shell command with `subprocess.run`. Blocks a short list of destructive patterns. Returns stdout+stderr, exit code. |
| `_read_file(args)` | Reads a file with optional `start_line`/`end_line` windowing. Returns numbered lines. |
| `_write_file(args)` | Writes content to a file, creating parent directories as needed. |
| `_append_file(args)` | Appends content to an existing file. |
| `_list_dir(args)` | Lists directory contents. `recursive=True` uses `rglob`, capped at 100 entries. |
| `_grep(args)` | Regex search across a file or directory tree. Case-insensitive by default. Returns up to 50 matches with file path and line number. |
| `_find_files(args)` | Glob pattern search (`fnmatch`) across a directory tree. Returns up to `max_results` relative paths. |
| `_web_search(args)` | DuckDuckGo JSON API search. Returns abstract text and up to 5 related topics. |
| `_memory_read(args)` | Placeholder — returns a note directing the caller to use the memory engine directly. |
| `_memory_write(args)` | Placeholder — records metadata for the memory engine to process. |
| `_task_done(args)` | Signals task completion. Returns a `TASK_DONE` marker with the summary and output. |
| `_ask_human(args)` | Prints a question and reads a line from stdin. Used when the agent is stuck. |

---

### `observability/observability.py`

Full runtime observability: event bus, hallucination detection, loop detection, token tracking, and distributed tracing.

| Class/Method | Summary |
|-------------|---------|
| `EventBus.__init__(audit_dir)` | Initialises the event bus with a JSONL audit log writer. |
| `EventBus.subscribe(handler)` | Registers a callback to receive every emitted `AgentEvent`. |
| `EventBus.emit(event)` | Fires an event to all subscribers and writes it to the audit log. |
| `EventBus.emit_simple(event_type, **kwargs)` | Convenience wrapper: builds and emits an `AgentEvent` in one call. |
| `EventBus.get_recent(n, event_type)` | Returns the `n` most recent events, optionally filtered by type. |
| `EventBus.get_trace(trace_id)` | Returns all events belonging to a trace ID. |
| `EventBus.get_metrics_summary()` | Aggregates event counts, LLM call stats, tool call counts, and error rates. |
| `HallucinationDetector.update_verified_fact(key, value, source)` | Stores a grounded fact for later contradiction checking. |
| `HallucinationDetector.analyze(agent_response, tool_outputs, task_text)` | Runs 5 hallucination checks: confabulation phrases, specific claim verification, contradiction detection, impossible values, tool-output alignment. Returns a list of `HallucinationFlag` objects. |
| `_check_confabulation_phrases(text)` | Flags hedging phrases that signal made-up content ("I believe", "should work", "typically"). |
| `_check_specific_claims(text, tool_outputs, verified_facts)` | Checks that specific factual claims (line numbers, counts, file names) appear in tool outputs. |
| `_check_contradictions(text)` | Detects internal contradictions within the response (e.g. "works" and "doesn't work" in same output). |
| `_check_impossible_values(text)` | Flags statistically impossible numbers (negative lines, 100% coverage, zero latency). |
| `_check_tool_alignment(text, tool_outputs)` | Flags when the response describes an action but no corresponding tool result exists. |
| `LoopDetector.check(tool_name, args)` | Checks if the tool+args combination has been seen before. Returns a `LoopDetectionResult` with level `info/warn/error/critical` and a message. |
| `LoopDetector.record(tool_name, args, result)` | Records a tool call in the detection history after it executes. |
| `TokenTracker.record(model, tokens_in, tokens_out, duration_ms, cost_usd, task_id, agent_id)` | Records one LLM call's token usage and cost. |
| `TokenTracker.get_task_usage(task_id)` | Returns aggregated token and cost stats for a specific task. |
| `TokenTracker.get_session_total()` | Returns total tokens and cost across the entire session. |
| `Span.set_attribute(key, value)` | Attaches a key-value attribute to a tracing span. |
| `Span.add_event(name, attributes)` | Records a timestamped event within a span. |
| `Span.end(status, error)` | Closes the span and records its duration. |
| `Tracer.start_span(name, trace_id, parent_span_id, attributes)` | Creates and registers a new tracing span. |
| `Tracer.end_span(span, status, error)` | Closes a span and fires an event. |
| `Tracer.format_trace(trace_id)` | Returns a formatted multi-line string of all spans in a trace. |
| `ObservabilityManager.__init__(audit_dir)` | Wires together the event bus, hallucination detector, loop detector, token tracker, and tracer. |
| `ObservabilityManager.new_trace()` | Generates and returns a new trace ID. |
| `ObservabilityManager.record_llm_call(model, tokens_in, tokens_out, duration_ms, task_id, agent_id)` | Records an LLM call in the token tracker and emits an LLM_CALL event. Returns the cost in USD. |
| `ObservabilityManager.check_loop(tool_name, args)` | Delegates to the loop detector's `check()`. |
| `ObservabilityManager.record_tool(tool_name, args, result)` | Records a tool call in the loop detector and emits a TOOL_CALL event. |

---

### `skills/skill_registry.py`

Legacy skill registry (pre–`SkillEngine`). Used by the orchestrator's specialist agents for YAML skill loading.

| Method | Summary |
|--------|---------|
| `SkillRegistry.__init__(skills_dir)` | Loads YAML skill files from the given directory on init. |
| `SkillRegistry.load_all()` | Walks the skills directory and parses all `.yaml` files. |
| `SkillRegistry._parse_skill(doc, source)` | Converts a raw YAML dict into a `Skill` dataclass with defaults. |
| `SkillRegistry.match(task)` | Scores all skills against the task using keyword overlap. Returns `(best_skill, score)`. |
| `SkillRegistry.get_step_config(skill, step_idx)` | Returns the configuration dict for a specific step index. |
| `SkillRegistry.get_all_step_hints(skill)` | Returns a formatted multi-line string of all step hints for a skill. |
| `SkillRegistry.record_usage(skill_name, success)` | Updates usage stats for a skill and persists them. |
| `SkillRegistry.list_skills()` | Returns a list of all loaded skill summary dicts. |
| `SkillRegistry.get_skill(name)` | Looks up a skill by exact name. |
| `SkillRegistry.skill_count` | Property: number of loaded skills. |
| `SkillRegistry.format_skills_list()` | Returns a formatted multi-line string of all skills for display. |

---

### `skills/builtin/skills.yaml`

Three built-in skill templates loaded at startup:

| Skill | Trigger tags | Steps |
|-------|-------------|-------|
| `debug_workflow` | debug, fix, error, bug, failing, test | find_error → read_context → identify_cause → apply_fix → verify |
| `grep_then_edit` | refactor, rename, update, grep, find-replace | find_all → plan_changes → apply_changes → verify |
| `test_and_fix` | test, ci, failing, pytest, jest, mocha | run_tests → analyze_failures → fix_one → verify_one → repeat_or_done |

New skills are automatically generated and saved to `.agent_os/skills/` after successful runs.

---

## Skill System

Skills are YAML templates that encode a proven workflow for a class of tasks. The agent checks the skill registry before planning and, if a skill matches (TF-IDF score ≥ 0.08), uses its step list as the execution blueprint.

**Skill lifecycle:**
```
Built-in YAML skills  (loaded at startup from skills/builtin/)
         +
Generated skills      (loaded from .agent_os/skills/)
         │
         ▼
SkillEngine.retrieve()   ─── TF-IDF match + tag boost ──► SkillMatch
         │
         ▼
adapt_for_prompt()       ─── render to system prompt ────► prompt fragment
         │
         ▼
[task runs successfully with ≥ 2 tool calls]
         │
         ▼
create_from_workflow()   ─── crystallise tool sequence ──► new Skill
save_skill()             ─── write to .agent_os/skills/
```

---

## Memory System

Five persistent banks, each stored as a JSON file under `.agent_os/memory/<agent_id>/`:

| Bank | Content | TTL |
|------|---------|-----|
| `task` | Facts discovered during the current task | 24 hours |
| `project` | Long-lived project-level facts (file structure, APIs) | None |
| `user` | User preferences and working style | 30 days |
| `run` | Past run outcomes (task text, success, summary, tokens) | None |
| `errors` | Errors encountered and the tools that caused them | None |

Retrieval is TF-IDF with priority boosting. The `task` bank is cleared at the start of every new run.

---

## Supported Providers & Models

| Provider | Flag | Models |
|----------|------|--------|
| Groq | `--provider groq` | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, and others |
| Ollama (local) | `--provider ollama` | `llama3.2:3b`, `llama3.1:8b`, `gemma2:9b`, `phi3:mini`, and any Ollama-hosted model |
| OpenAI | `--provider openai` | `gpt-4o-mini`, `gpt-4o`, `o3-mini` |
| Anthropic | `--provider anthropic` | `claude-sonnet-4-20250514`, `claude-haiku-4-5`, `claude-opus-4` |

Models not in the registry are handled by a heuristic fallback: names containing `70b`, `claude`, or `gpt-4` → LLM tier; everything else → SLM tier.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes (for Groq) | — | Groq API key |
| `OPENAI_API_KEY` | For OpenAI | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Anthropic | — | Anthropic API key |
| `DEFAULT_MODEL` | No | `llama-3.1-8b-instant` | Default model name |
| `DEFAULT_PROVIDER` | No | `groq` | Default provider |
| `AGENT_MAX_STEPS` | No | `20` | Default max steps |
| `AGENT_TEMPERATURE` | No | `0.3` | Default temperature |
| `AGENT_MEMORY_DIR` | No | `.agent_os/memory` | Memory storage path |
| `AGENT_AUDIT_DIR` | No | `.agent_os/audit` | Audit log path |

---

## Project Structure

```
Kartavya/
├── cli/
│   └── main.py                  # CLI entry point and command dispatch
├── core/
│   ├── agent_os.py              # AgentOS: 9-stage pipeline orchestrator
│   ├── agent.py                 # Legacy single-agent runtime (used by orchestrator)
│   ├── task_intake.py           # Stage 1: task classification (heuristic, no LLM)
│   ├── model_router.py          # Stage 2: model-aware routing and tool selection
│   ├── planner.py               # Stage 6: plan generation per task type
│   ├── skill_engine.py          # Stage 3 + 9b: skill retrieval and creation
│   ├── memory_engine.py         # Stage 4 + 9a: five-bank persistent memory
│   ├── knowledge_retriever.py   # Stage 5: repo context and web search
│   ├── verifier.py              # Stage 8: output verification and retry strategy
│   ├── context_manager.py       # Token-budget-aware message assembly
│   └── orchestrator.py          # Multi-agent coordinator for compound tasks
├── providers/
│   └── providers.py             # LLM provider abstraction (Groq, Ollama, OpenAI, Anthropic)
├── tools/
│   └── tools.py                 # Tool registry and implementations (12 tools)
├── observability/
│   └── observability.py         # Event bus, loop detection, tracing, token tracking
├── skills/
│   ├── builtin/
│   │   └── skills.yaml          # Built-in skill templates (debug, refactor, test+fix)
│   └── skill_registry.py        # Legacy skill registry used by sub-agents
├── .agent_os/
│   ├── memory/                  # Per-agent JSON memory banks
│   ├── audit/                   # JSONL audit log of all events
│   ├── runs/                    # Session run log
│   └── skills/                  # Auto-generated skills from successful runs
├── .env.example                 # Environment variable template
└── requirements.txt             # Python dependencies
```