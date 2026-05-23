#!/usr/bin/env python3
"""
main.py — CLI for Kartavya Agent OS v4

Usage:
  python cli/main.py run "fix the failing tests"
  python cli/main.py run "..." --model llama3.2:3b --provider ollama
  python cli/main.py run "..." --provider groq --model llama-3.3-70b-versatile
  python cli/main.py run "..." --provider anthropic --model claude-sonnet-4-20250514
  python cli/main.py chat
  python cli/main.py skills
  python cli/main.py memory
  python cli/main.py dashboard
  python cli/main.py models --provider ollama
"""

import sys, os, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _api_key(provider: str) -> str | None:
    key_map = {
        "groq":      "GROQ_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return os.environ.get(key_map.get(provider, ""), None)


def _make_agent(args) -> "AgentOS":
    from core.agent_os import AgentOS, AgentOSConfig
    cfg = AgentOSConfig(
        model=args.model,
        provider=args.provider,
        max_steps=getattr(args, "max_steps", 20),
        temperature=getattr(args, "temperature", 0.2),
        stream=not getattr(args, "no_stream", False),
        verify_outputs=not getattr(args, "no_verify", False),
        create_skills=not getattr(args, "no_skill_learn", False),
        allowed_tools=getattr(args, "tools", "").split(",")
                      if getattr(args, "tools", "") else [],
    )
    return AgentOS(
        config=cfg,
        api_key=_api_key(args.provider),
        ollama_host=getattr(args, "ollama_host", None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(args):
    agent = _make_agent(args)
    task  = " ".join(args.task) if args.task else input("Task: ").strip()
    if not task:
        print("No task provided.")
        return

    result = agent.run_task(task)

    print("\n" + "─" * 62)
    print("OUTPUT")
    print("─" * 62)
    print(result.output or "(no output captured)")
    print("─" * 62)
    print(
        f"steps={result.steps_taken}  tokens={result.tokens_used:,}  "
        f"cost=${result.cost_usd:.5f}  status={result.verify_status}"
    )
    if result.skill_used:
        print(f"skill_used={result.skill_used}")
    if result.errors:
        print(f"errors: {'; '.join(result.errors[:3])}")


def cmd_chat(args):
    agent = _make_agent(args)
    print(f"\n🚀 Agent OS — interactive chat")
    print(f"   {args.provider} / {args.model}")
    print(f"   Type 'exit', 'dashboard', 'memory', 'skills' for commands")
    print(f"{'─'*50}\n")

    while True:
        try:
            task = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not task:
            continue
        if task.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break
        if task.lower() == "dashboard":
            agent.dashboard()
            continue
        if task.lower() == "memory":
            print(agent.memory_export())
            continue
        if task.lower() == "skills":
            for s in agent.list_skills():
                print(
                    f"  {s['name']} v{s.get('version','?')} "
                    f"[{s['task_type']}] "
                    f"uses={s['usage_count']} ok={s['success_rate']:.0%}"
                )
            continue

        result = agent.run_task(task)
        print("\n── Output ──────────────────────────────────────────")
        print(result.output or "(no output)")
        print(f"── {result.steps_taken} steps | {result.tokens_used:,} tokens ──\n")


def cmd_skills(args):
    """List all loaded skills."""
    agent = _make_agent(args)
    skills = agent.list_skills()
    if not skills:
        print("No skills loaded.")
        return
    print(f"\n{'─'*62}")
    print(f"  SKILLS ({len(skills)} loaded)")
    print(f"{'─'*62}")
    for s in skills:
        health = "✅" if s.get("healthy", True) else "⚠️"
        print(
            f"  {health} {s['name']:<28} v{s.get('version','?'):<8} "
            f"[{s.get('task_type','?'):<12}]  "
            f"uses={s['usage_count']:>3}  "
            f"ok={s['success_rate']:.0%}"
        )
        print(f"     tags: {', '.join(s.get('tags', [])[:6])}")


def cmd_memory(args):
    """Show memory bank summaries."""
    agent = _make_agent(args)
    print(agent.memory_export())


def cmd_dashboard(args):
    agent = _make_agent(args)
    agent.dashboard()


def cmd_models(args):
    from providers.providers import get_provider
    key = _api_key(args.provider)
    try:
        provider = get_provider(
            args.provider,
            api_key=key,
            host=getattr(args, "ollama_host", None),
        )
        models = provider.list_models()
        print(f"\nModels available via {args.provider}:")
        for m in sorted(models):
            print(f"  {m}")
    except Exception as e:
        print(f"Error listing models: {e}")


def cmd_intake(args):
    """Show how the task intake processor classifies a task (debug)."""
    from core.task_intake import process_task
    task = " ".join(args.task) if args.task else input("Task: ").strip()
    intake = process_task(task)
    print(json.dumps({
        "task":       intake.task,
        "type":       intake.task_type.value,
        "complexity": intake.complexity.value,
        "risk":       intake.risk.value,
        "scope":      intake.scope,
        "action_verbs":   intake.action_verbs,
        "file_refs":       intake.file_refs,
        "tech_stack":      intake.tech_stack,
        "entities":        intake.entities,
        "needs_bash":      intake.needs_bash,
        "needs_external":  intake.needs_external_knowledge,
        "is_multi_step":   intake.is_multi_step,
        "skill_query":     intake.skill_query,
    }, indent=2))


def cmd_route(args):
    """Show how the model router configures a run (debug)."""
    from core.task_intake import process_task
    from core.model_router import route, ModelRouter
    task   = " ".join(args.task) if args.task else input("Task: ").strip()
    intake = process_task(task)
    config = route(args.model, args.provider, intake)
    router = ModelRouter()
    print(router.describe(config))


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _add_model_args(p):
    p.add_argument("--model",    default=os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant"))
    p.add_argument("--provider", default=os.getenv("DEFAULT_PROVIDER", "groq"))
    p.add_argument("--ollama-host", default=None)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--no-verify", action="store_true", help="Skip output verification")
    p.add_argument("--no-skill-learn", action="store_true", help="Skip skill creation")
    p.add_argument("--tools", default="", help="Comma-separated allowed tools")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="agent_os",
        description="Kartavya Agent OS — grounded AI task executor",
    )
    sub = ap.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Execute a single task")
    p_run.add_argument("task", nargs="*")
    _add_model_args(p_run)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat mode")
    _add_model_args(p_chat)

    # skills
    p_skills = sub.add_parser("skills", help="List loaded skills")
    _add_model_args(p_skills)

    # memory
    p_mem = sub.add_parser("memory", help="Show memory bank summaries")
    _add_model_args(p_mem)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Show observability dashboard")
    _add_model_args(p_dash)

    # models
    p_models = sub.add_parser("models", help="List available models")
    _add_model_args(p_models)

    # debug commands
    p_intake = sub.add_parser("intake", help="Debug: show task intake analysis")
    p_intake.add_argument("task", nargs="*")
    _add_model_args(p_intake)

    p_route = sub.add_parser("route", help="Debug: show model routing decision")
    p_route.add_argument("task", nargs="*")
    _add_model_args(p_route)

    return ap


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "run":       cmd_run,
        "chat":      cmd_chat,
        "skills":    cmd_skills,
        "memory":    cmd_memory,
        "dashboard": cmd_dashboard,
        "models":    cmd_models,
        "intake":    cmd_intake,
        "route":     cmd_route,
    }

    if not args.command:
        parser.print_help()
        return

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
