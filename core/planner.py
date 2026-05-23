"""
planner.py — Model-aware planning for the Agent OS.

Generates a short, explicit execution plan adapted to:
  - The model tier (SLM = compact plan; LLM = richer plan)
  - The task complexity and type
  - The matched skill (if any)
  - The retrieved memory

The plan is a structured prompt fragment injected into the system prompt.
It sets the agent's intention before the first LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from core.task_intake import TaskIntake, TaskType, Complexity
from core.model_router import ModelRuntimeConfig, ModelTier
from core.skill_engine import SkillMatch


# ─────────────────────────────────────────────────────────────────────────────
# Plan data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    index:        int
    description:  str
    tools:        list[str]
    hint:         str  = ""
    is_verify:    bool = False

@dataclass
class Plan:
    steps:            list[PlanStep]
    success_criteria: list[str]
    max_steps:        int
    depth:            str          # "minimal" | "standard" | "deep"
    skill_name:       str = ""     # skill that influenced this plan
    fallback_msg:     str = ""     # what to do if plan fails

    def to_prompt_fragment(self) -> str:
        """Convert plan to the text injected into the system prompt."""
        lines = ["## Execution Plan"]

        for step in self.steps:
            tools_str = (
                f"  [tools: {', '.join(step.tools)}]"
                if step.tools else ""
            )
            verify_mark = " ✓" if step.is_verify else ""
            lines.append(f"{step.index}. {step.description}{verify_mark}{tools_str}")
            if step.hint and self.depth != "minimal":
                lines.append(f"   → {step.hint}")

        if self.success_criteria:
            lines.append("")
            lines.append("Done when: " + "; ".join(self.success_criteria[:3]))

        if self.fallback_msg:
            lines.append("")
            lines.append(f"If stuck: {self.fallback_msg}")

        return "\n".join(lines)

    @property
    def estimated_steps(self) -> int:
        return len(self.steps)


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────

class Planner:
    """
    Generates a model-tier-appropriate plan from task intake + skill match.

    No LLM calls. Pure heuristic. The plan is a compact structured block
    injected into the system prompt before the first reasoning step.
    """

    def generate(
        self,
        intake:         TaskIntake,
        config:         ModelRuntimeConfig,
        skill_match:    Optional[SkillMatch] = None,
        prior_run_summary: str = "",
    ) -> Plan:

        depth = config.planning_depth

        # ── If we have a strong skill match, adapt its steps ─────────────────
        if skill_match and skill_match.score > 0.15:
            return self._plan_from_skill(intake, config, skill_match, depth)

        # ── Otherwise: heuristic plan by task type ────────────────────────────
        return self._plan_heuristic(intake, config, depth, prior_run_summary)

    # ── Skill-guided planning ─────────────────────────────────────────────────

    def _plan_from_skill(
        self,
        intake: TaskIntake,
        config: ModelRuntimeConfig,
        match:  SkillMatch,
        depth:  str,
    ) -> Plan:
        skill = match.skill
        max_s = min(config.max_plan_steps, config.max_skill_steps, len(skill.steps))
        steps = []
        for i, ss in enumerate(skill.steps[:max_s]):
            # Filter tools to allowed list
            tools = [t for t in ss.tools if t in config.allowed_tools]
            hint  = ss.prompt_hint if depth != "minimal" else ""
            steps.append(PlanStep(
                index=i + 1,
                description=ss.description,
                tools=tools,
                hint=hint,
                is_verify=("verify" in ss.step_id.lower() or "test" in ss.step_id.lower()),
            ))

        # Always add a verification step if there isn't one
        has_verify = any(s.is_verify for s in steps)
        if not has_verify and config.max_plan_steps > len(steps):
            steps.append(PlanStep(
                index=len(steps) + 1,
                description="Verify the output is correct and complete",
                tools=self._verify_tools(intake, config),
                is_verify=True,
            ))

        return Plan(
            steps=steps,
            success_criteria=skill.success_criteria or ["task completed"],
            max_steps=config.max_steps,
            depth=depth,
            skill_name=skill.name,
            fallback_msg="If a step fails, try a different tool or approach — don't retry identically.",
        )

    # ── Heuristic planning ────────────────────────────────────────────────────

    def _plan_heuristic(
        self,
        intake: TaskIntake,
        config: ModelRuntimeConfig,
        depth:  str,
        prior_run_summary: str,
    ) -> Plan:

        task_type = intake.task_type

        if config.tier == ModelTier.SLM:
            return self._slm_plan(intake, config)

        # LLM plans by task type
        if task_type == TaskType.CODING:
            return self._coding_plan(intake, config, depth)
        if task_type == TaskType.RESEARCH:
            return self._research_plan(intake, config, depth)
        if task_type == TaskType.AUTOMATION:
            return self._automation_plan(intake, config, depth)
        if task_type == TaskType.OPS:
            return self._ops_plan(intake, config, depth)
        if task_type == TaskType.DATA:
            return self._data_plan(intake, config, depth)
        return self._generic_plan(intake, config, depth)

    # ── SLM plan (minimal) ────────────────────────────────────────────────────

    def _slm_plan(self, intake: TaskIntake, config: ModelRuntimeConfig) -> Plan:
        """
        SLMs get a 3-step plan: Gather → Act → Verify.
        Keep it tight — they lose track of long plans.
        """
        steps = [
            PlanStep(1, "Gather needed information",
                     [t for t in ["bash", "grep", "find_files", "read_file"]
                      if t in config.allowed_tools][:2]),
            PlanStep(2, "Execute the main action",
                     [t for t in config.allowed_tools
                      if t not in ("task_done", "memory_read")][:2]),
            PlanStep(3, "Verify the result",
                     self._verify_tools(intake, config),
                     is_verify=True),
        ]
        return Plan(
            steps=steps,
            success_criteria=["output matches requirements"],
            max_steps=config.max_steps,
            depth="minimal",
            fallback_msg="If stuck, use bash to check the current state.",
        )

    # ── Type-specific LLM plans ───────────────────────────────────────────────

    def _coding_plan(self, intake: TaskIntake, config: ModelRuntimeConfig,
                      depth: str) -> Plan:
        has_files = bool(intake.file_refs)
        steps = []

        if has_files or depth == "deep":
            steps.append(PlanStep(
                1, f"Read relevant files and understand current state",
                [t for t in ["read_file", "grep", "find_files"]
                 if t in config.allowed_tools],
                hint="Read the specific files mentioned in the task first.",
            ))

        steps.append(PlanStep(
            len(steps) + 1,
            "Identify what exactly needs to change",
            [],
            hint="State the specific change needed before writing any code.",
        ))

        steps.append(PlanStep(
            len(steps) + 1,
            "Implement the change",
            [t for t in ["write_file", "bash"]
             if t in config.allowed_tools],
            hint="Make the minimum change needed. Don't refactor unrelated code.",
        ))

        steps.append(PlanStep(
            len(steps) + 1,
            "Verify the implementation is correct",
            self._verify_tools(intake, config),
            hint="Run tests or syntax check. Confirm the change works.",
            is_verify=True,
        ))

        return Plan(
            steps=steps,
            success_criteria=["code correct", "tests pass"],
            max_steps=config.max_steps,
            depth=depth,
            fallback_msg="If tests fail, read the error carefully before retrying.",
        )

    def _research_plan(self, intake: TaskIntake, config: ModelRuntimeConfig,
                        depth: str) -> Plan:
        steps = [
            PlanStep(1, "Map the relevant files and structure",
                     [t for t in ["find_files", "list_dir"] if t in config.allowed_tools],
                     hint="Use find_files to locate relevant code/docs."),
            PlanStep(2, "Read the key files",
                     [t for t in ["read_file", "grep"] if t in config.allowed_tools],
                     hint="Read at most 3-5 files most relevant to the question."),
            PlanStep(3, "Synthesize a specific, grounded answer",
                     [],
                     hint="Answer should cite file names and line numbers.",
                     is_verify=True),
        ]
        return Plan(
            steps=steps,
            success_criteria=["specific answer provided", "grounded in files"],
            max_steps=config.max_steps,
            depth=depth,
        )

    def _automation_plan(self, intake: TaskIntake, config: ModelRuntimeConfig,
                          depth: str) -> Plan:
        steps = [
            PlanStep(1, "Check environment and prerequisites",
                     [t for t in ["bash", "list_dir", "read_file"]
                      if t in config.allowed_tools],
                     hint="Check existing state before running commands."),
            PlanStep(2, "Execute the automation",
                     [t for t in ["bash"] if t in config.allowed_tools],
                     hint="Run one command at a time. Check exit code after each."),
            PlanStep(3, "Verify execution succeeded",
                     [t for t in ["bash"] if t in config.allowed_tools],
                     hint="Check output and exit code. Confirm side effects are correct.",
                     is_verify=True),
        ]
        return Plan(
            steps=steps,
            success_criteria=["exit code 0", "expected output"],
            max_steps=config.max_steps,
            depth=depth,
            fallback_msg="If a command fails, read its error output carefully before retrying.",
        )

    def _ops_plan(self, intake: TaskIntake, config: ModelRuntimeConfig,
                   depth: str) -> Plan:
        steps = [
            PlanStep(1, "Assess current state",
                     [t for t in ["bash", "read_file"] if t in config.allowed_tools]),
            PlanStep(2, "Apply change",
                     [t for t in ["bash", "write_file"] if t in config.allowed_tools]),
            PlanStep(3, "Verify change and check for side effects",
                     [t for t in ["bash"] if t in config.allowed_tools],
                     is_verify=True),
        ]
        return Plan(
            steps=steps,
            success_criteria=["service running", "config applied"],
            max_steps=config.max_steps,
            depth=depth,
        )

    def _data_plan(self, intake: TaskIntake, config: ModelRuntimeConfig,
                    depth: str) -> Plan:
        steps = [
            PlanStep(1, "Inspect data source and schema",
                     [t for t in ["bash", "read_file"] if t in config.allowed_tools]),
            PlanStep(2, "Process and transform the data",
                     [t for t in ["bash", "write_file"] if t in config.allowed_tools]),
            PlanStep(3, "Validate output",
                     [t for t in ["bash"] if t in config.allowed_tools],
                     is_verify=True),
        ]
        return Plan(
            steps=steps,
            success_criteria=["output matches expected format"],
            max_steps=config.max_steps,
            depth=depth,
        )

    def _generic_plan(self, intake: TaskIntake, config: ModelRuntimeConfig,
                       depth: str) -> Plan:
        steps = [
            PlanStep(1, "Understand the current state",
                     [t for t in ["bash", "read_file", "list_dir"]
                      if t in config.allowed_tools][:2]),
            PlanStep(2, "Execute the required action",
                     [t for t in config.allowed_tools
                      if t not in ("task_done", "memory_read")][:3]),
            PlanStep(3, "Verify and confirm the result",
                     self._verify_tools(intake, config),
                     is_verify=True),
        ]
        return Plan(
            steps=steps,
            success_criteria=["task completed", "output correct"],
            max_steps=config.max_steps,
            depth=depth,
        )

    def _verify_tools(self, intake: TaskIntake,
                       config: ModelRuntimeConfig) -> list[str]:
        if intake.needs_bash:
            return [t for t in ["bash"] if t in config.allowed_tools]
        return [t for t in ["read_file", "grep"]
                if t in config.allowed_tools][:1]
