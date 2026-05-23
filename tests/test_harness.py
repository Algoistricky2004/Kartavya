"""
test_harness.py — Comprehensive test suite for Kartavya Agent OS.

Run:
    python tests/test_harness.py            # all tests
    python tests/test_harness.py -v         # verbose
    python tests/test_harness.py ToolTests  # one class

Test categories:
    Unit:        parser, context, memory, tokens, loop detection,
                 hallucination, tools, run_memory, task_intent
    Integration: skill_registry, orchestrator (heuristic), budget enforcement
    Live:        Groq API, Ollama (auto-skipped if unavailable)
    Safety:      dangerous commands, tool allowlists, budget caps

All unit tests run in < 3 seconds with no network or API key required.
"""

import sys
import os
import json
import time
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _groq_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))

def _ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Tool Call Parser Tests
# ─────────────────────────────────────────────────────────────────────────────

class ToolParserTests(unittest.TestCase):
    def setUp(self):
        from providers.providers import parse_tool_calls
        self.parse = parse_tool_calls

    def test_xml_with_args_block(self):
        text = '<tool>find_files</tool><args>{"pattern":"*.py"}</args>'
        result = self.parse(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "find_files")
        self.assertIn("pattern", result[0]["args"])

    def test_xml_without_args_block(self):
        text = '<tool>bash</tool>{"command":"ls -la"}'
        result = self.parse(text)
        self.assertEqual(result[0]["name"], "bash")
        self.assertIn("command", result[0]["args"])

    def test_react_bash_inline(self):
        text = "Thought: I need to list files\nBash: find . -name '*.py'"
        result = self.parse(text)
        self.assertEqual(result[0]["name"], "bash")
        self.assertIn("find", result[0]["args"]["command"])

    def test_react_bash_code_block(self):
        text = "Thought: Check files\nBash:\n```bash\nfind . -name '*.py'\n```"
        result = self.parse(text)
        self.assertEqual(result[0]["name"], "bash")
        self.assertIn("find", result[0]["args"]["command"])

    def test_react_done(self):
        text = "Thought: I have the answer\nDone: core/agent.py\ncli/main.py"
        result = self.parse(text)
        self.assertEqual(result[0]["name"], "task_done")
        self.assertIn("core/agent.py", result[0]["args"].get("output", ""))

    def test_react_action_alias(self):
        text = "Action: ls -la"
        result = self.parse(text)
        self.assertEqual(result[0]["name"], "bash")

    def test_empty_returns_empty(self):
        self.assertEqual(self.parse(""), [])
        self.assertEqual(self.parse("Just some text with no tool."), [])

    def test_json_raw_format(self):
        text = '{"name": "read_file", "arguments": {"path": "core/agent.py"}}'
        result = self.parse(text)
        if result:  # may or may not parse depending on format
            self.assertEqual(result[0]["name"], "read_file")


# ─────────────────────────────────────────────────────────────────────────────
# Context Manager Tests
# ─────────────────────────────────────────────────────────────────────────────

class ContextManagerTests(unittest.TestCase):
    def test_budget_validates_for_all_presets(self):
        from core.context_manager import BUDGET_PRESETS
        for model, budget in BUDGET_PRESETS.items():
            budget.validate()  # should not raise

    def test_build_returns_messages_and_snapshot(self):
        from core.context_manager import ContextManager
        ctx = ContextManager("llama-3.1-8b-instant")
        msgs, snap = ctx.build(
            system_prompt="You are an agent.",
            current_input="List python files.",
        )
        self.assertGreaterEqual(len(msgs), 2)
        self.assertGreater(snap.total_tokens, 0)
        self.assertGreaterEqual(snap.headroom_tokens, 0)

    def test_budget_not_exceeded_tiny_model(self):
        from core.context_manager import ContextManager
        ctx = ContextManager("llama3.2:3b")
        msgs, snap = ctx.build(
            system_prompt="Agent prompt.",
            current_input="Do something.",
            memory_chunks=["Fact A", "Fact B"],
        )
        self.assertGreaterEqual(snap.headroom_tokens, 0,
                                "Context budget should not be exceeded")

    def test_tool_definitions_truncated_to_budget(self):
        from core.context_manager import ContextManager
        ctx = ContextManager("llama3.2:3b")
        many_tools = [
            {"name": f"tool_{i}", "description": "Does something useful " * 5,
             "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}
            for i in range(20)
        ]
        msgs, snap = ctx.build(
            system_prompt="Agent.",
            current_input="Task.",
            tool_definitions=many_tools,
        )
        self.assertGreaterEqual(snap.headroom_tokens, 0)
        self.assertLessEqual(len(snap.tools_loaded), len(many_tools))

    def test_record_turn_updates_history(self):
        from core.context_manager import ContextManager
        ctx = ContextManager("llama-3.1-8b-instant")
        ctx.record_turn("user msg", "assistant response")
        snap = ctx.get_latest_snapshot()
        # Should have no snapshot yet (build not called), history should be updated
        self.assertEqual(len(ctx.history.turns), 2)

    def test_token_estimator_basic(self):
        from core.context_manager import estimate_tokens
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("Hello world"), 0)
        self.assertLess(estimate_tokens("short"), estimate_tokens("x" * 1000))


# ─────────────────────────────────────────────────────────────────────────────
# Memory Store Tests
# ─────────────────────────────────────────────────────────────────────────────

class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        from core.memory import MemoryStore, MemorySection, MemoryPriority
        self.store = MemoryStore("test_agent", self._tmpdir.name)
        self.MemorySection  = MemorySection
        self.MemoryPriority = MemoryPriority

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_upsert_and_retrieve(self):
        self.store.upsert(self.MemorySection.PROJECT_FACTS, "Python 3.11 project",
                          priority=self.MemoryPriority.HIGH)
        results = self.store.retrieve("python version")
        self.assertTrue(len(results) > 0)
        self.assertIn("Python", results[0].content)

    def test_deduplication(self):
        self.store.upsert(self.MemorySection.DECISIONS, "Use pytest")
        self.store.upsert(self.MemorySection.DECISIONS, "Use pytest")  # duplicate
        entries = self.store.get_section(self.MemorySection.DECISIONS)
        self.assertEqual(len(entries), 1)

    def test_high_priority_always_included(self):
        self.store.upsert(self.MemorySection.PROJECT_FACTS, "Critical info",
                          priority=self.MemoryPriority.HIGH)
        self.store.upsert(self.MemorySection.DECISIONS, "Low priority thing",
                          priority=self.MemoryPriority.LOW)
        high = self.store.get_all_high_priority()
        self.assertEqual(len(high), 1)
        self.assertIn("Critical info", high[0].content)

    def test_ttl_expiry(self):
        self.store.upsert(self.MemorySection.ERRORS_SEEN, "Expired error",
                          ttl_hours=0.00001)  # expires almost instantly
        time.sleep(0.5)
        entries = self.store.get_section(self.MemorySection.ERRORS_SEEN)
        self.assertEqual(len(entries), 0)

    def test_context_chunks_respect_token_budget(self):
        for i in range(20):
            self.store.upsert(self.MemorySection.PROJECT_FACTS,
                              f"Fact number {i}: " + "x" * 100)
        chunks = self.store.build_context_chunks("fact", max_chunks=5, max_tokens=200)
        self.assertLessEqual(len(chunks), 5)

    def test_persistence_across_instances(self):
        from core.memory import MemoryStore, MemorySection, MemoryPriority
        self.store.upsert(MemorySection.PROJECT_FACTS, "Persisted fact",
                          priority=MemoryPriority.HIGH)
        # New instance should load from disk
        store2 = MemoryStore("test_agent", self._tmpdir.name)
        entries = store2.get_all_high_priority()
        self.assertTrue(any("Persisted fact" in e.content for e in entries))

    def test_update_task_status(self):
        self.store.upsert(self.MemorySection.CURRENT_TASKS,
                          "[ACTIVE] task123: Do something",
                          priority=self.MemoryPriority.HIGH, tags=["task123"])
        self.store.update_task_status("task123", "DONE", "completed successfully")
        entries = self.store.get_section(self.MemorySection.CURRENT_TASKS)
        self.assertTrue(any("DONE" in e.content for e in entries))


# ─────────────────────────────────────────────────────────────────────────────
# Loop Detector Tests
# ─────────────────────────────────────────────────────────────────────────────

class LoopDetectorTests(unittest.TestCase):
    def setUp(self):
        from observability.observability import LoopDetector, LoopKind
        self.LoopDetector = LoopDetector
        self.LoopKind = LoopKind

    def test_no_loop_on_first_call(self):
        d = self.LoopDetector()
        result = d.check("bash", {"command": "ls"})
        self.assertFalse(result.stuck)

    def test_warning_on_repeated_tool(self):
        d = self.LoopDetector()
        d.WARN_THRESHOLD = 3
        for _ in range(3):
            d.record("bash", {"command": "ls"}, "output")
        result = d.check("bash", {"command": "ls"})
        self.assertTrue(result.stuck)
        self.assertEqual(result.level, "warning")

    def test_critical_on_many_repeats(self):
        d = self.LoopDetector()
        d.CRIT_THRESHOLD = 5
        for _ in range(5):
            d.record("bash", {"command": "ls"}, "output")
        result = d.check("bash", {"command": "ls"})
        self.assertTrue(result.stuck)
        self.assertEqual(result.level, "critical")

    def test_circuit_breaker_fires(self):
        d = self.LoopDetector()
        for i in range(d.BREAKER):
            d.record("bash", {"command": f"cmd_{i}"}, f"out_{i}")
        result = d.check("bash", {"command": "any"})
        self.assertTrue(result.stuck)
        self.assertEqual(result.kind, self.LoopKind.CIRCUIT_BREAKER)

    def test_different_args_no_loop(self):
        d = self.LoopDetector()
        d.WARN_THRESHOLD = 3
        for i in range(10):
            d.record("bash", {"command": f"cmd_{i}"}, f"result_{i}")
        result = d.check("bash", {"command": "new_cmd"})
        self.assertFalse(result.stuck)


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination Detector Tests
# ─────────────────────────────────────────────────────────────────────────────

class HallucinationDetectorTests(unittest.TestCase):
    def setUp(self):
        from observability.observability import HallucinationDetector
        self.detector = HallucinationDetector()

    def test_clean_response_low_score(self):
        score, flags = self.detector.analyze(
            "The file has 10 lines. Running the tests produces exit code 0.")
        self.assertLess(score, 0.4)

    def test_confabulation_language_detected(self):
        score, flags = self.detector.analyze(
            "I believe the file is probably util.py. I think it might work. "
            "I'm not sure about the exact line number.")
        self.assertGreater(score, 0.0)
        kinds = [f.kind for f in flags]
        self.assertIn("confabulation_language", kinds)

    def test_verified_fact_update(self):
        self.detector.update_verified_fact("line_count", "42", source="tool:read_file")
        self.assertIn("line_count", self.detector.verified_facts)

    def test_impossible_token_count(self):
        score, flags = self.detector.analyze(
            "The response used 500000 tokens in total.")
        kinds = [f.kind for f in flags]
        self.assertIn("impossible_value", kinds)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry Tests
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        from tools.tools import ToolRegistry
        self.registry = ToolRegistry()

    def test_bash_echo(self):
        result = self.registry.execute("bash", {"command": "echo kartavya_test"})
        self.assertTrue(result.success)
        self.assertIn("kartavya_test", result.output)

    def test_list_dir(self):
        result = self.registry.execute("list_dir", {"path": "."})
        self.assertTrue(result.success)
        self.assertGreater(len(result.output), 0)

    def test_find_files_finds_python(self):
        result = self.registry.execute("find_files",
                                       {"pattern": "*.py", "directory": "."})
        self.assertTrue(result.success)
        self.assertIn(".py", result.output)

    def test_grep_finds_pattern(self):
        result = self.registry.execute("grep",
                                       {"pattern": "import", "path": "core/agent.py"})
        self.assertTrue(result.success)
        self.assertGreater(len(result.output), 0)

    def test_task_done(self):
        result = self.registry.execute("task_done",
                                       {"summary": "done", "output": "42"})
        self.assertTrue(result.success)
        self.assertIn("TASK_DONE", result.output)

    def test_tool_allowlist_enforced(self):
        from tools.tools import ToolRegistry
        restricted = ToolRegistry(allowed=["read_file", "list_dir"])
        result = restricted.execute("bash", {"command": "echo x"})
        self.assertFalse(result.success)
        self.assertIn("not allowed", result.error)

    def test_dangerous_command_blocked(self):
        result = self.registry.execute("bash", {"command": "rm -rf /"})
        self.assertFalse(result.success)
        self.assertIn("Blocked", result.error)

    def test_unknown_tool_returns_error(self):
        result = self.registry.execute("nonexistent_tool", {})
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.error)

    def test_write_and_read_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.txt")
            write_result = self.registry.execute("write_file",
                                                  {"path": path, "content": "hello world"})
            self.assertTrue(write_result.success)
            read_result = self.registry.execute("read_file", {"path": path})
            self.assertTrue(read_result.success)
            self.assertIn("hello world", read_result.output)

    def test_step_tools_filtered_by_task(self):
        tools = self.registry.get_definitions_for_step(1, "list all python files")
        names = [t["name"] for t in tools]
        self.assertIn("find_files", names)
        self.assertIn("task_done", names)

    def test_write_tools_included_for_edit_task(self):
        tools = self.registry.get_definitions_for_step(1, "fix and write the bug in utils.py")
        names = [t["name"] for t in tools]
        self.assertIn("write_file", names)


# ─────────────────────────────────────────────────────────────────────────────
# Run Memory Tests
# ─────────────────────────────────────────────────────────────────────────────

class RunMemoryTests(unittest.TestCase):
    def test_save_and_retrieve(self):
        from core.agent import RunMemory
        with tempfile.TemporaryDirectory() as d:
            rm = RunMemory(d)
            rm.save_result("list python files", "core/agent.py\ntools/tools.py",
                           True, 2, 500, 0.001, 5000)
            similar = rm.get_similar("list all python files")
            self.assertGreater(len(similar), 0)

    def test_format_for_context(self):
        from core.agent import RunMemory
        with tempfile.TemporaryDirectory() as d:
            rm = RunMemory(d)
            rm.save_result("find config files", "config.yaml", True, 1, 200, 0.0005, 2000)
            ctx = rm.format_for_context("find configuration files")
            self.assertIn("config", ctx.lower())

    def test_no_similar_for_unrelated_task(self):
        from core.agent import RunMemory
        with tempfile.TemporaryDirectory() as d:
            rm = RunMemory(d)
            rm.save_result("list python files", "agent.py", True, 1, 100, 0.001, 1000)
            similar = rm.get_similar("deploy kubernetes cluster")
            self.assertEqual(len(similar), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Task Intent Detector Tests
# ─────────────────────────────────────────────────────────────────────────────

class TaskIntentTests(unittest.TestCase):
    def setUp(self):
        from core.agent import TaskIntentDetector
        self.detector = TaskIntentDetector

    def test_satisfied_after_find_files(self):
        satisfied, result = self.detector.is_satisfied(
            "list all python files",
            {"find_files": "core/agent.py\ntools/tools.py\ncli/main.py"}
        )
        self.assertTrue(satisfied)
        self.assertIn("core/agent.py", result)

    def test_not_satisfied_with_empty_results(self):
        satisfied, _ = self.detector.is_satisfied(
            "list all python files", {}
        )
        self.assertFalse(satisfied)

    def test_short_task_auto_satisfies(self):
        satisfied, _ = self.detector.is_satisfied(
            "show files",
            {"list_dir": "[.]\n📁 core\n📄 agent.py\n📄 tools.py"}
        )
        self.assertTrue(satisfied)

    def test_not_satisfied_with_tiny_result(self):
        satisfied, _ = self.detector.is_satisfied(
            "list all python files",
            {"find_files": "a.py"}  # only 4 chars — too short
        )
        self.assertFalse(satisfied)


# ─────────────────────────────────────────────────────────────────────────────
# Token Tracker Tests
# ─────────────────────────────────────────────────────────────────────────────

class TokenTrackerTests(unittest.TestCase):
    def setUp(self):
        from observability.observability import TokenTracker
        self.tracker = TokenTracker()

    def test_records_cost(self):
        cost = self.tracker.record("llama-3.1-8b-instant", 1000, 200,
                                    task_id="t1", agent_id="a1")
        self.assertGreater(cost, 0)

    def test_session_total(self):
        self.tracker.record("gpt-4o-mini", 500, 100, task_id="t1")
        self.tracker.record("gpt-4o-mini", 300, 50, task_id="t1")
        total = self.tracker.get_session_total()
        self.assertEqual(total["total_tokens_in"], 800)
        self.assertEqual(total["total_tokens_out"], 150)
        self.assertEqual(total["total_calls"], 2)

    def test_task_usage_tracked(self):
        self.tracker.record("gpt-4o-mini", 100, 50, task_id="task_abc")
        usage = self.tracker.get_task_usage("task_abc")
        self.assertEqual(usage["tokens_in"], 100)

    def test_unknown_model_uses_default(self):
        cost = self.tracker.record("unknown-model-xyz", 1000, 100)
        self.assertGreater(cost, 0)  # should not crash


# ─────────────────────────────────────────────────────────────────────────────
# Skill Registry Tests
# ─────────────────────────────────────────────────────────────────────────────

class SkillRegistryTests(unittest.TestCase):
    def setUp(self):
        from skills.skill_registry import SkillRegistry
        self.registry = SkillRegistry()

    def test_loads_builtin_skills(self):
        self.assertGreater(self.registry.skill_count(), 0,
                           "Should load at least 1 builtin skill")

    def test_matches_debug_task(self):
        skill, score = self.registry.match("fix the failing test in test_auth.py")
        self.assertGreater(score, 0)

    def test_matches_refactor_task(self):
        skill, score = self.registry.match("rename the function across the codebase")
        self.assertGreater(score, 0)

    def test_no_match_for_unrelated_task(self):
        # Should not strongly match any skill
        skill, score = self.registry.match("what is the weather today")
        # Either no match or very low score
        if skill:
            self.assertLess(score, 0.5)

    def test_get_step_config(self):
        skills = self.registry.list_skills()
        if skills:
            skill = self.registry.get_skill(skills[0]["name"])
            if skill and skill.steps:
                cfg = self.registry.get_step_config(skill, 0)
                self.assertIn("tools", cfg)
                self.assertIn("hint", cfg)

    def test_usage_tracking(self):
        skills = self.registry.list_skills()
        if skills:
            name = skills[0]["name"]
            skill = self.registry.get_skill(name)
            initial_count = skill.usage_count
            initial_success = skill.success_count
            self.registry.record_usage(name, success=True)
            self.assertEqual(skill.usage_count, initial_count + 1)
            self.assertEqual(skill.success_count, initial_success + 1)

    def test_format_skills_list_is_string(self):
        listing = self.registry.format_skills_list()
        self.assertIsInstance(listing, str)

    def test_get_all_step_hints(self):
        skills = self.registry.list_skills()
        if skills:
            skill = self.registry.get_skill(skills[0]["name"])
            if skill:
                hints = self.registry.get_all_step_hints(skill)
                self.assertIsInstance(hints, str)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Tests (no LLM required — uses heuristic decomposition)
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        from core.orchestrator import TaskDecomposer, ResultMerger, SubTask
        self.decomposer = TaskDecomposer()
        self.merger = ResultMerger()
        self.SubTask = SubTask

    def test_simple_task_not_decomposed(self):
        plan = self.decomposer.decompose_heuristic("list all python files")
        self.assertEqual(plan.strategy, "single")
        self.assertEqual(len(plan.subtasks), 1)
        self.assertFalse(plan.is_complex)

    def test_fix_test_task_decomposed(self):
        plan = self.decomposer.decompose_heuristic("fix the bug and run the tests")
        self.assertGreater(len(plan.subtasks), 1)
        self.assertTrue(plan.is_complex)
        roles = [st.role for st in plan.subtasks]
        self.assertIn("tester", roles)

    def test_implement_verify_decomposed(self):
        plan = self.decomposer.decompose_heuristic(
            "implement the login function and verify it works correctly")
        self.assertGreater(len(plan.subtasks), 1)
        roles = [st.role for st in plan.subtasks]
        self.assertIn("coder", roles)

    def test_long_task_decomposed(self):
        long_task = ("Analyze the authentication module, find all security vulnerabilities, "
                     "fix each one, write tests for the fixes, and confirm all tests pass.")
        plan = self.decomposer.decompose_heuristic(long_task)
        self.assertGreater(len(plan.subtasks), 1)

    def test_merge_single_result(self):
        import uuid
        from core.agent import TaskResult
        subtask = self.SubTask(uuid.uuid4().hex[:8], "general", "do something")
        subtask.result = TaskResult(
            task_id="t1", success=True, output="File list: a.py b.py",
            steps_taken=2, tokens_used=100, cost_usd=0.001,
            duration_ms=500, trace_id="tr1", errors=[],
            hallucination_flags=[], loop_breaks=0,
        )
        output, success = self.merger.merge("do something", [subtask])
        self.assertTrue(success)
        self.assertIn("a.py", output)

    def test_merge_with_failed_tester(self):
        import uuid
        from core.agent import TaskResult
        coder_task = self.SubTask(uuid.uuid4().hex[:8], "coder", "fix bug")
        coder_task.result = TaskResult(
            "t1", True, "Fixed the bug.", 3, 200, 0.002,
            600, "tr1", [], [], 0,
        )
        tester_task = self.SubTask(uuid.uuid4().hex[:8], "tester", "run tests")
        tester_task.result = TaskResult(
            "t2", False, "FAILED: 2 tests failed", 4, 300, 0.003,
            700, "tr1", ["test failed"], [], 0,
        )
        output, success = self.merger.merge("fix bug and test", [coder_task, tester_task])
        self.assertFalse(success)

    def test_conflict_detection(self):
        import uuid
        from core.agent import TaskResult
        st1 = self.SubTask(uuid.uuid4().hex[:8], "coder", "implement")
        st1.result = TaskResult("t1", True, "All tests passed.", 2, 100, 0.001, 500, "tr1", [], [], 0)
        st2 = self.SubTask(uuid.uuid4().hex[:8], "tester", "test")
        st2.result = TaskResult("t2", False, "2 tests failed.", 3, 200, 0.002, 600, "tr1", ["error"], [], 0)
        conflicts = self.merger.detect_conflicts([st1, st2])
        self.assertGreater(len(conflicts), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Budget Enforcement Tests
# ─────────────────────────────────────────────────────────────────────────────

class BudgetTests(unittest.TestCase):
    def test_agent_config_has_budget_field(self):
        from core.agent import AgentConfig
        cfg = AgentConfig(token_budget_usd=0.05)
        self.assertEqual(cfg.token_budget_usd, 0.05)

    def test_agent_config_default_no_budget(self):
        from core.agent import AgentConfig
        cfg = AgentConfig()
        self.assertIsNone(cfg.token_budget_usd)

    def test_orchestrator_config_budget(self):
        from core.orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig(token_budget_usd=0.10)
        self.assertEqual(cfg.token_budget_usd, 0.10)


# ─────────────────────────────────────────────────────────────────────────────
# Observability Event Bus Tests
# ─────────────────────────────────────────────────────────────────────────────

class EventBusTests(unittest.TestCase):
    def setUp(self):
        from observability.observability import EventBus, EventType
        self._tmpdir = tempfile.TemporaryDirectory()
        self.bus = EventBus(self._tmpdir.name)
        self.EventType = EventType

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_emit_and_retrieve(self):
        ev = self.bus.emit_simple(self.EventType.TASK_CREATED,
                                   trace_id="tr1", agent_id="agent_1",
                                   task_id="task_1")
        recent = self.bus.get_recent(10)
        self.assertGreater(len(recent), 0)

    def test_metrics_summary(self):
        self.bus.emit_simple(self.EventType.LLM_CALL_END,
                              tokens_in=100, tokens_out=50,
                              cost_usd=0.001, duration_ms=250, success=True)
        metrics = self.bus.get_metrics_summary()
        self.assertIn("total_events", metrics)
        self.assertIn("total_tokens", metrics)

    def test_trace_retrieval(self):
        self.bus.emit_simple(self.EventType.TASK_CREATED, trace_id="my_trace_123")
        self.bus.emit_simple(self.EventType.TASK_COMPLETED, trace_id="my_trace_123")
        trace_events = self.bus.get_trace("my_trace_123")
        self.assertEqual(len(trace_events), 2)

    def test_audit_log_written(self):
        self.bus.emit_simple(self.EventType.TASK_CREATED, trace_id="audit_test")
        import datetime
        date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        log_path = Path(self._tmpdir.name) / f"audit-{date}.jsonl"
        self.assertTrue(log_path.exists())
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        self.assertGreater(len(lines), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Live API Tests (skipped if no key/service)
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_groq_available(), "GROQ_API_KEY not set")
class GroqLiveTests(unittest.TestCase):
    def test_simple_completion(self):
        from providers.providers import GroqProvider
        p = GroqProvider()
        resp = p.complete(
            messages=[
                {"role": "system", "content": "Answer with a single digit only."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=10, temperature=0.0, stream=False,
        )
        self.assertIn("4", resp.content)
        self.assertGreater(resp.tokens_in, 0)

    def test_tool_call_parsed(self):
        from providers.providers import GroqProvider, parse_tool_calls
        from core.context_manager import ContextManager
        p = GroqProvider()
        ctx = ContextManager("llama-3.1-8b-instant")
        msgs, _ = ctx.build(
            system_prompt=(
                "When asked to find files, respond with EXACTLY:\n"
                "<tool>find_files</tool><args>{\"pattern\":\"*.py\"}</args>"
            ),
            current_input="Find all python files.",
        )
        resp = p.complete(msgs, "llama-3.1-8b-instant",
                          max_tokens=100, temperature=0.0, stream=False)
        tc = resp.tool_calls or parse_tool_calls(resp.content)
        self.assertTrue(len(tc) > 0, f"No tool call parsed from: {resp.content[:100]}")


@unittest.skipUnless(_ollama_running(), "Ollama not running")
class OllamaLiveTests(unittest.TestCase):
    def test_simple_answer(self):
        from providers.providers import OllamaProvider
        p = OllamaProvider()
        resp = p.complete(
            messages=[
                {"role": "system", "content": "Answer with just a number."},
                {"role": "user", "content": "What is 3+3?"},
            ],
            model="llama3.2:3b",
            max_tokens=16, temperature=0.0,
        )
        self.assertIn("6", resp.content)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kartavya Agent OS — Test Suite")
    parser.add_argument("filter", nargs="?", help="Filter by test class name")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    verbosity = 2 if args.verbose else 1

    if args.filter:
        suite = unittest.TestLoader().loadTestsFromName(args.filter, sys.modules[__name__])
    else:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
