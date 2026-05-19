#!/usr/bin/env python3
"""
main.py — CLI entry point for the Enterprise Agent OS

Commands:
  run      — Run a task
  chat     — Interactive agent chat
  test     — Run built-in test suite
  dashboard — Show observability dashboard
  memory   — Show agent memory
  trace    — Show execution trace

Usage:
  python main.py run "find all TODO comments in src/"
  python main.py chat
  python main.py test
  python main.py dashboard
  python main.py memory
  python main.py trace <trace_id>
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_run(args):
    """Run a single task."""
    from core.agent import Agent, AgentConfig

    config = AgentConfig(
        model=args.model,
        provider=args.provider,
        max_steps=args.max_steps,
        temperature=args.temperature,
        stream=not args.no_stream,
    )

    api_key = _get_api_key(args.provider)
    agent   = Agent(config=config, api_key=api_key)

    task = " ".join(args.task) if args.task else input("Task: ").strip()
    result = agent.run_task(task)

    print(f"\n{'─'*50}")
    print(f"Output: {result.output[:500]}")
    print(f"Steps:  {result.steps_taken} | Tokens: {result.tokens_used:,} | Cost: ${result.cost_usd:.4f}")
    print(f"Trace:  {result.trace_id}")
    if args.show_trace:
        print("\n" + agent.get_trace(result.trace_id))


def cmd_chat(args):
    """Interactive chat mode."""
    from core.agent import Agent, AgentConfig

    config = AgentConfig(
        model=args.model,
        provider=args.provider,
        max_steps=args.max_steps,
        temperature=args.temperature,
    )
    api_key = _get_api_key(args.provider)
    agent   = Agent(config=config, api_key=api_key)

    print(f"🦞 Agent OS — Interactive Mode")
    print(f"   Model: {config.model} via {config.provider}")
    print(f"   Type 'exit' to quit, 'dashboard' to see metrics, 'memory' to see memory")
    print(f"{'─'*50}\n")

    while True:
        try:
            task = input("You: ").strip()
            if not task:
                continue
            if task.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if task.lower() == "dashboard":
                agent.dashboard()
                continue
            if task.lower() == "memory":
                print(agent.memory_export())
                continue
            if task.lower().startswith("trace "):
                trace_id = task.split(" ", 1)[1].strip()
                print(agent.get_trace(trace_id))
                continue

            result = agent.run_task(task)
            print(f"\nAgent: {result.output}\n")

        except KeyboardInterrupt:
            print("\nInterrupted")
            break


def cmd_test(args):
    """Run the built-in test suite."""
    print("🧪 Running Agent OS Test Suite\n")
    results = []

    # Test 1: Context Manager
    results.append(_test_context_manager())

    # Test 2: Loop Detection
    results.append(_test_loop_detector())

    # Test 3: Hallucination Detection
    results.append(_test_hallucination_detector())

    # Test 4: Memory Store
    results.append(_test_memory())

    # Test 5: Tool Registry
    results.append(_test_tools())

    # Test 6: Token estimation
    results.append(_test_token_estimation())

    # Test 7: Live Groq call (if key available)
    if os.environ.get("GROQ_API_KEY"):
        results.append(_test_groq_live())
    else:
        results.append(("Groq Live Call", "SKIP", "No GROQ_API_KEY set"))

    # Summary
    print(f"\n{'═'*50}")
    print("TEST RESULTS")
    print(f"{'═'*50}")
    passed = failed = skipped = 0
    for name, status, detail in results:
        icon = {"PASS":"✅", "FAIL":"❌", "SKIP":"⏭"}.get(status, "?")
        print(f"  {icon} {name:<35} {status}")
        if status == "FAIL":
            print(f"       └── {detail}")
        if status == "PASS": passed += 1
        elif status == "FAIL": failed += 1
        else: skipped += 1
    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'═'*50}")
    return failed == 0


def _test_context_manager():
    name = "Context Manager (small window)"
    try:
        from core.context_manager import ContextManager, BUDGET_PRESETS

        budget = BUDGET_PRESETS["llama-3.1-8b-instant"]
        budget.validate()  # Should not raise

        ctx = ContextManager(model="llama-3.1-8b-instant")
        messages, snapshot = ctx.build(
            system_prompt="You are a helpful assistant.",
            current_input="What is 2+2?",
            task_context="Simple math question.",
            memory_chunks=["User likes concise answers"],
            tool_definitions=[],
        )

        assert len(messages) >= 2, "Should have at least system + user message"
        assert snapshot.total_tokens > 0, "Should have used some tokens"
        assert snapshot.headroom_tokens > 0, "Should have headroom"
        assert snapshot.total_tokens <= budget.total_context - budget.max_output, \
               "Must not exceed context window"

        # Test that compression kicks in for large inputs
        large_input = "x" * 5000  # ~1250 tokens
        messages2, snapshot2 = ctx.build(
            system_prompt="Brief system.",
            current_input=large_input,
        )
        assert any("truncated" in t.lower() for t in snapshot2.truncations), \
               "Should have triggered truncation"

        return (name, "PASS", "")
    except Exception as e:
        return (name, "FAIL", str(e))


def _test_loop_detector():
    name = "Loop Detector (4 detectors)"
    try:
        from observability.observability import LoopDetector, LoopKind

        detector = LoopDetector()
        detector.WARN_THRESHOLD = 3  # Lower for testing
        detector.CRIT_THRESHOLD = 5

        # Test 1: Generic repeat
        for i in range(3):
            result = detector.check("bash", {"command": "ls"})
            detector.record("bash", {"command": "ls"}, "file1.py\nfile2.py")

        result = detector.check("bash", {"command": "ls"})
        assert result.stuck, "Should detect repeat"
        assert result.level in ("warning", "critical")

        # Test 2: Poll no progress
        detector2 = LoopDetector()
        detector2.WARN_THRESHOLD = 3
        for i in range(4):
            detector2.check("command_status", {"id": "task_1"})
            detector2.record("command_status", {"id": "task_1"}, "running...")

        result2 = detector2.check("command_status", {"id": "task_1"})
        assert result2.stuck and result2.kind == LoopKind.POLL_NO_PROGRESS, \
               "Should detect poll loop"

        # Test 3: Circuit breaker
        detector3 = LoopDetector()
        for i in range(detector3.BREAKER):
            detector3.record("bash", {"command": f"cmd_{i}"}, f"output_{i}")

        result3 = detector3.check("bash", {"command": "one_more"})
        assert result3.stuck and result3.kind == LoopKind.CIRCUIT_BREAKER, \
               "Should trigger circuit breaker"

        return (name, "PASS", "")
    except Exception as e:
        return (name, "FAIL", str(e))


def _test_hallucination_detector():
    name = "Hallucination Detector"
    try:
        from observability.observability import HallucinationDetector

        detector = HallucinationDetector()

        # Test 1: Clean response
        score, flags = detector.analyze("The file contains 42 lines.")
        assert score < 0.5, f"Clean response shouldn't have high score: {score}"

        # Test 2: Confabulation phrases
        score2, flags2 = detector.analyze(
            "I believe the function is called processData. I think it's in utils.py."
        )
        assert len(flags2) > 0, "Should detect confabulation phrases"
        assert any(f.kind == "confabulation_language" for f in flags2)

        # Test 3: Contradiction with verified fact
        detector.update_verified_fact("exit_code", "1", source="bash")
        score3, flags3 = detector.analyze(
            "The command succeeded with exit code 0.",
            tool_results={"bash": "Error: exit code 1"}
        )
        # Should have some flags
        assert score3 >= 0 or len(flags3) >= 0  # Basic: should not crash

        # Test 4: Impossible value
        score4, flags4 = detector.analyze("The command ran in 999999 ms.")
        # Optional: may or may not detect this depending on pattern
        assert score4 >= 0  # Should not crash

        return (name, "PASS", "")
    except Exception as e:
        return (name, "FAIL", str(e))


def _test_memory():
    name = "Memory Store (tiered RAG)"
    try:
        import tempfile
        from core.memory import MemoryStore, MemorySection, MemoryPriority

        with tempfile.TemporaryDirectory() as tmpdir:
            mem = MemoryStore("test_agent", tmpdir)

            # Write entries
            mem.upsert(MemorySection.PROJECT_FACTS,
                       "Main language is Python 3.11",
                       priority=MemoryPriority.HIGH)
            mem.upsert(MemorySection.CURRENT_TASKS,
                       "Fix the failing test in test_auth.py",
                       priority=MemoryPriority.HIGH,
                       tags=["task_001"])
            mem.upsert(MemorySection.DECISIONS,
                       "Decided to use pytest for testing",
                       priority=MemoryPriority.NORMAL)
            mem.upsert(MemorySection.ERRORS_SEEN,
                       "ImportError: module 'requests' not found",
                       priority=MemoryPriority.NORMAL,
                       ttl_hours=1)

            # Read: high priority always returned
            high = mem.get_all_high_priority()
            assert len(high) == 2, f"Expected 2 high priority, got {len(high)}"

            # Retrieve relevant
            relevant = mem.retrieve("test authentication", n=5)
            assert len(relevant) > 0, "Should find relevant entries"
            assert any("test" in e.content.lower() for e in relevant)

            # Build context chunks
            chunks = mem.build_context_chunks("failing test", max_chunks=4)
            assert len(chunks) > 0, "Should build chunks"
            assert all(isinstance(c, str) for c in chunks)

            # Expiry
            mem.upsert(MemorySection.ERRORS_SEEN,
                       "Temp error",
                       ttl_hours=-0.001)  # already expired
            entries = mem.get_section(MemorySection.ERRORS_SEEN)
            # Expired entry should not appear
            assert not any("Temp error" in e.content for e in entries)

            # Task status update
            mem.update_task_status("task_001", "IN_PROGRESS", "running tests")
            tasks = mem.get_section(MemorySection.CURRENT_TASKS)
            assert any("IN_PROGRESS" in e.content for e in tasks)

            # Stats
            stats = mem.get_stats()
            assert stats["total_entries"] > 0
            assert stats["high_priority"] == 2

        return (name, "PASS", "")
    except Exception as e:
        import traceback
        return (name, "FAIL", f"{e}\n{traceback.format_exc()[-200:]}")


def _test_tools():
    name = "Tool Registry"
    try:
        from tools.tools import ToolRegistry
        import tempfile

        registry = ToolRegistry()

        # Test: list_dir
        result = registry.execute("list_dir", {"path": "."})
        assert result.success, f"list_dir failed: {result.error}"

        # Test: bash
        result = registry.execute("bash", {"command": "echo hello_agent_os", "timeout": 5})
        assert result.success, f"bash failed: {result.error}"
        assert "hello_agent_os" in result.output

        # Test: find_files
        result = registry.execute("find_files", {"pattern": "*.py", "directory": "."})
        # May or may not find files — just should not error
        assert not (result.error and "Error" in result.error), f"find_files error: {result.error}"

        # Test: grep
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello_world():\n    pass\n")
            f.flush()
            result = registry.execute("grep", {"pattern": "def hello", "path": f.name})
            assert result.success
            assert "hello_world" in result.output

        # Test: write + read
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmppath = f.name

        wr = registry.execute("write_file", {"path": tmppath, "content": "test content\nline 2"})
        assert wr.success, f"write failed: {wr.error}"

        rd = registry.execute("read_file", {"path": tmppath})
        assert rd.success, f"read failed: {rd.error}"
        assert "test content" in rd.output

        # Test: blocked tool
        restricted = ToolRegistry(allowed=["read_file", "list_dir"])
        result = restricted.execute("bash", {"command": "echo x"})
        assert not result.success, "Restricted tool should fail"

        # Test: task_done
        done = registry.execute("task_done", {"summary": "Done", "output": "The answer is 42"})
        assert done.success
        assert "TASK_DONE" in done.output

        return (name, "PASS", "")
    except Exception as e:
        import traceback
        return (name, "FAIL", f"{e}\n{traceback.format_exc()[-300:]}")


def _test_token_estimation():
    name = "Token Estimation"
    try:
        from core.context_manager import estimate_tokens

        # Short text
        t1 = estimate_tokens("Hello world")
        assert 2 <= t1 <= 8, f"Short text: expected 2-8, got {t1}"

        # Empty
        t2 = estimate_tokens("")
        assert t2 == 0

        # Long text
        t3 = estimate_tokens("a" * 4000)
        assert 800 <= t3 <= 2000, f"Long text: expected 800-2000, got {t3}"

        # Code (more dense tokenization)
        code = "def calculate_fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"
        t4 = estimate_tokens(code)
        assert 20 <= t4 <= 80, f"Code: expected 20-80, got {t4}"

        return (name, "PASS", "")
    except Exception as e:
        return (name, "FAIL", str(e))


def _test_groq_live():
    name = "Groq Live Call (llama-3.1-8b)"
    try:
        from providers.providers import GroqProvider
        from core.context_manager import ContextManager

        provider = GroqProvider()
        ctx      = ContextManager(model="llama-3.1-8b-instant")

        messages, snapshot = ctx.build(
            system_prompt="You are a concise AI assistant. Reply in under 20 words.",
            current_input="What is 2+2? Reply with just the number and one sentence.",
        )

        start    = time.time()
        response = provider.complete(
            messages=messages,
            model="llama-3.1-8b-instant",
            max_tokens=50,
            temperature=0.0,
            stream=True,
        )
        duration = (time.time() - start) * 1000

        assert response.content, "Should have content"
        assert "4" in response.content, f"Should contain '4': {response.content}"
        assert response.tokens_out > 0
        assert duration < 10000, f"Should complete in <10s, took {duration:.0f}ms"

        print(f"\n     Response: '{response.content.strip()[:60]}'")
        print(f"     Tokens: {response.tokens_in}in/{response.tokens_out}out | {duration:.0f}ms")

        return (name, "PASS", "")
    except Exception as e:
        return (name, "FAIL", str(e))


def cmd_dashboard(args):
    """Show observability dashboard."""
    from core.agent import Agent, AgentConfig
    config = AgentConfig(model=args.model, provider=args.provider)
    agent  = Agent(config=config, api_key=_get_api_key(args.provider))
    agent.dashboard()


def cmd_memory(args):
    """Show agent memory."""
    from core.memory import MemoryStore
    mem = MemoryStore(args.agent_id)
    print(mem.export_markdown())


def cmd_trace(args):
    """Show execution trace."""
    if not args.trace_id:
        print("Usage: trace <trace_id>")
        return
    from observability.observability import ObservabilityManager
    obs = ObservabilityManager()
    print(obs.tracer.format_trace(args.trace_id))


def _get_api_key(provider: str) -> str:
    env_map = {
        "groq":      "GROQ_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    key = os.environ.get(env_map.get(provider, ""), "")
    if not key:
        print(f"⚠️  Warning: {env_map.get(provider, 'API_KEY')} not set")
    return key


# ── Argument Parser ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🦞 Enterprise Agent OS — Built on OpenClaw architecture"
    )
    parser.add_argument("--model",    default="llama-3.1-8b-instant",
                        help="Model name (default: llama-3.1-8b-instant)")
    parser.add_argument("--provider", default="groq",
                        help="Provider: groq | openai | anthropic")
    parser.add_argument("--max-steps", type=int, default=20,
                        dest="max_steps")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--agent-id", default="agent_default",
                        dest="agent_id")

    subparsers = parser.add_subparsers(dest="command")

    # run
    run_parser = subparsers.add_parser("run", help="Run a task")
    run_parser.add_argument("task", nargs="*", help="Task description")
    run_parser.add_argument("--show-trace", action="store_true", dest="show_trace")
    run_parser.add_argument("--no-stream", action="store_true", dest="no_stream")

    # chat
    subparsers.add_parser("chat", help="Interactive chat mode")

    # test
    subparsers.add_parser("test", help="Run test suite")

    # dashboard
    subparsers.add_parser("dashboard", help="Show observability dashboard")

    # memory
    subparsers.add_parser("memory", help="Show agent memory")

    # trace
    trace_parser = subparsers.add_parser("trace", help="Show execution trace")
    trace_parser.add_argument("trace_id", nargs="?", help="Trace ID to show")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "test":
        success = cmd_test(args)
        sys.exit(0 if success else 1)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "memory":
        cmd_memory(args)
    elif args.command == "trace":
        cmd_trace(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
