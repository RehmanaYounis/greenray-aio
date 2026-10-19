"""Interactive hands-on lab: a simulated shell plus grading meta-commands."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..commands import HELP
from ..commands import names as command_names
from ..content import solutions
from ..content.scenarios import Scenario
from ..core.errors import CommandError
from ..core.shell import Shell
from ..core.state import COMPUTE_NODES, HEAD, make_cluster
from . import progress
from .ui import UI, enable_line_editing

LAB_HELP = """\
Lab commands (not part of the simulated cluster):
  task       show the scenario briefing again
  check      grade your work against the objectives
  hint       reveal the next hint (counts against a clean pass)
  solution   give up and show the reference solution, explanation and edge cases
  commands   list simulated commands available
  reset      restart this scenario from the broken state
  quit       leave the lab
Everything else is run in the simulated cluster. You start on the BCM head node;
use `ssh dgx-01` to reach a compute node and `exit` to come back.
"""

EXAM_LAB_HELP = """\
Exam lab commands:
  task       show the task again
  commands   list simulated commands available
  back       return to the exam menu (your work is kept)
"""


@dataclass
class LabSession:
    """One attempt at a scenario; keeps the shell so an exam can resume it."""

    scenario: Scenario | None
    shell: Shell = field(init=False)
    hints_used: int = 0
    started: float = field(default_factory=time.time)

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        state = self.scenario.build_state() if self.scenario else make_cluster()
        self.shell = Shell(state)

    def grade(self) -> list[tuple[str, bool, str]]:
        return self.scenario.grade(self.shell) if self.scenario else []

    def score(self) -> float:
        results = self.grade()
        return sum(ok for _, ok, _ in results) / len(results) if results else 0.0


def show_briefing(ui: UI, s: Scenario | None, exam: bool = False) -> None:
    if not s:
        ui.out(ui.bold("\nSandbox: a healthy 4-node DGX H100 cluster managed by BCM. Nothing is graded.\n"))
        return
    meta = f"   [{s.domain}]" if exam else f"   [{s.domain} | {s.difficulty} | ~{s.time_limit_min} min]"
    ui.out(ui.bold(f"\n=== {s.title} ===") + ui.dim(meta))
    ui.out(s.briefing.rstrip())
    if s.objectives:
        ui.out(ui.bold("\nObjectives:"))
        for o in s.objectives:
            ui.out(f"  • {o}")
    ui.out(ui.dim("\nType 'help' for lab commands.\n"))


def show_results(ui: UI, results: list[tuple[str, bool, str]]) -> None:
    for desc, ok, err in results:
        ui.out(f"  {ui.green('PASS') if ok else ui.red('FAIL')}  {desc}" + (ui.dim(f"  ({err})") if err else ""))


def show_solution(ui: UI, s: Scenario, title: str = "Reference solution") -> None:
    if not s.has_solution:
        ui.out(ui.yellow("\n" + solutions.missing_message()))
        return
    ui.out(ui.bold(f"\n--- {title} ---"))
    for step in s.solution:
        ui.out(ui.cyan("  $ " + step.rstrip("\n").replace("\n", "\n    ")))
    if s.explanation:
        ui.out(ui.bold("\nWhy:"))
        ui.out(s.explanation.rstrip())
    if s.edge_cases:
        ui.out(ui.bold("\nEdge cases / exam traps:"))
        for e in s.edge_cases:
            ui.out(f"  • {e}")
    if s.references:
        ui.out(ui.dim("\nReferences: " + "; ".join(s.references)))


def run_shell_line(ui: UI, session: LabSession, line: str) -> None:
    try:
        out = session.shell.run(line, heredoc_reader=ui.ask)
    except CommandError as e:
        out = str(e)
    if out:
        ui.out(out)


def _completion_words() -> list[str]:
    meta = ["task", "check", "hint", "solution", "commands", "reset", "quit", "back"]
    return command_names() + meta + [HEAD, *COMPUTE_NODES]


def run_lab(ui: UI, s: Scenario | None) -> None:
    """Practice mode: hints, solution and debrief available; result saved to progress."""
    enable_line_editing(_completion_words())
    session = LabSession(s)
    gave_up = passed = False
    show_briefing(ui, s)
    while True:
        line = ui.ask(ui.green(session.shell.prompt))
        cmd = line.strip()
        if cmd in ("quit", "q"):
            break
        if cmd in ("help", "?"):
            ui.out(LAB_HELP)
        elif cmd == "commands":
            ui.out("\n".join(f"  {n:<16} {HELP.get(n, '')}" for n in command_names()))
        elif s and cmd == "task":
            show_briefing(ui, s)
        elif s and cmd == "reset":
            session.reset()
            ui.out(ui.yellow("Scenario reset."))
        elif s and cmd == "hint":
            if session.hints_used < len(s.hints):
                ui.out(ui.yellow(f"Hint {session.hints_used + 1}/{len(s.hints)}: {s.hints[session.hints_used]}"))
                session.hints_used += 1
            else:
                ui.out(ui.dim("No more hints. Type 'solution' to see the answer."))
        elif s and cmd in ("solution", "giveup"):
            if ui.ask("Reveal the solution? This ends the attempt as not passed. [y/N] ").strip().lower() == "y":
                gave_up = True
                show_solution(ui, s)
                break
        elif s and cmd == "check":
            results = session.grade()
            show_results(ui, results)
            mins = (time.time() - session.started) / 60
            if all(ok for _, ok, _ in results):
                passed = True
                ui.out(
                    ui.bold(
                        ui.green(
                            f"\nAll objectives met in {mins:.1f} min "
                            f"(target {s.time_limit_min} min, hints used: {session.hints_used})."
                        )
                    )
                )
                if mins > s.time_limit_min:
                    ui.out(ui.yellow("Correct, but slower than the target. Repeat it later for speed."))
                show_solution(ui, s, title="Debrief: reference solution and edge cases")
                break
            met = sum(ok for _, ok, _ in results)
            ui.out(ui.dim(f"\n{met}/{len(results)} objectives met, {mins:.1f} min elapsed."))
        else:
            run_shell_line(ui, session, line)
    if s:
        progress.record_scenario(s.id, passed, time.time() - session.started, session.hints_used, gave_up)


def run_exam_lab(ui: UI, session: LabSession, deadline: float) -> None:
    """Exam mode: no hints or solutions; returns to the exam menu on `back` or timeout."""
    enable_line_editing(_completion_words())
    show_briefing(ui, session.scenario, exam=True)
    while time.time() < deadline:
        left = int((deadline - time.time()) // 60)
        line = ui.ask(ui.dim(f"[{left}m] ") + ui.green(session.shell.prompt))
        cmd = line.strip()
        if cmd in ("back", "quit", "q"):
            return
        if cmd in ("help", "?"):
            ui.out(EXAM_LAB_HELP)
        elif cmd == "commands":
            ui.out("\n".join(f"  {n:<16} {HELP.get(n, '')}" for n in command_names()))
        elif cmd == "task":
            show_briefing(ui, session.scenario, exam=True)
        elif cmd in ("check", "hint", "solution", "giveup", "reset"):
            ui.out(ui.dim("Not available during the exam. Your work is graded when you finish."))
        else:
            run_shell_line(ui, session, line)
    ui.out(ui.red("\nTime is up."))
