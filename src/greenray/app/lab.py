"""Interactive hands-on lab: a simulated shell plus grading meta-commands.

Two ways to run a scenario:

* guided (``greenray lab <id>``): title, objectives, hints and solution are available.
* incident (``greenray incident``): only the incident report is shown. The id, title and
  objectives stay hidden (they point at the root cause) until the debrief. Revealing
  objectives or hints marks the current attempt as guided. ``retry`` starts a fresh
  attempt that can still earn an independent pass.
"""

from __future__ import annotations

import random
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field

from .. import config
from ..commands import HELP
from ..commands import names as command_names
from ..content import guides, solutions
from ..content import scenarios as sc
from ..content.scenarios import Scenario
from ..core.errors import CommandError
from ..core.shell import Shell
from ..core.state import COMPUTE_NODES, HEAD, make_cluster
from . import progress
from .ui import UI

LAB_HELP = """\
Lab commands (not part of the simulated cluster):
  task             show the scenario briefing again
  check            grade your work against the objectives
  explain <cmd>    what a command is for, its key options and what to look for in its output
  hint             reveal the next hint (the pass then counts as guided)
  solution         give up and show the reference solution, explanation and edge cases
  commands         list the simulated commands
  reset            put the cluster back to the broken state (same attempt: hints and timer carry over)
  quit             leave the lab
Everything else runs in the simulated cluster. You start on the BCM head node;
use `ssh dgx-01` to reach a compute node and `exit` to come back. Ctrl-C cancels the current line.
"""

INCIDENT_HELP = """\
Incident commands (not part of the simulated cluster):
  task             show the incident report again
  check            see whether the service is healthy again (unmet criteria stay hidden)
  explain <cmd>    what a command is for, its key options and what to look for in its output
  objectives       reveal the success criteria (this attempt then counts as guided)
  hint             reveal the next hint (this attempt then counts as guided)
  solution         give up and show the debrief
  commands         list the simulated commands
  reset            put the cluster back to the broken state (same attempt)
  retry            start a fresh attempt: cluster, timer, hints and objectives all reset
  quit             leave the incident
Ctrl-C cancels the current line.
"""

EXAM_LAB_HELP = """\
Exam lab commands:
  task       show the task again
  commands   list simulated commands available
  back       return to the exam menu (your work is kept)
"""

INCIDENT_GOAL = (
    "Your goal: find out what is wrong, fix the cause within the limits stated above, and make sure "
    "the service is really healthy again. Type `check` whenever you think you are done."
)


@dataclass
class LabSession:
    """One attempt at a scenario; keeps the shell so an exam can resume it."""

    scenario: Scenario | None
    color: bool = False
    ui: UI | None = None
    shell: Shell = field(init=False)
    hints_used: int = 0
    objectives_revealed: bool = False
    attempt: int = 1
    started: float = field(default_factory=time.time)

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        """Rebuild the broken cluster. Same attempt: hints, objectives and timer carry over."""
        state = self.scenario.build_state() if self.scenario else make_cluster()
        self.shell = Shell(state, color=self.color)
        if self.ui is not None and self.ui.interactive:  # stream watch/dmon samples on a real terminal
            self.shell.ctx.emit = self.ui.out
            self.shell.ctx.live_delay = 1.0

    def new_attempt(self) -> None:
        """Fresh attempt: cluster, timer and guidance counters all start over."""
        self.reset()
        self.hints_used = 0
        self.objectives_revealed = False
        self.attempt += 1
        self.started = time.time()

    @property
    def scenario_required(self) -> Scenario:
        """The scenario of a graded attempt (only the sandbox has none)."""
        if self.scenario is None:
            raise ValueError("this lab session has no scenario (sandbox)")
        return self.scenario

    @property
    def guided(self) -> bool:
        return self.hints_used > 0 or self.objectives_revealed

    def grade(self) -> list[tuple[str, bool, str]]:
        return self.scenario.grade(self.shell) if self.scenario else []

    def score(self) -> float:
        results = self.grade()
        return sum(ok for _, ok, _ in results) / len(results) if results else 0.0


def read_line(ui: UI, prompt: str) -> str:
    """Read one line; Ctrl-C cancels the line (as in bash) instead of ending the program."""
    try:
        return ui.ask(prompt)
    except KeyboardInterrupt:
        ui.out("^C")
        return ""


SEVERITY = {"easy": "Sev 4", "medium": "Sev 3", "hard": "Sev 2"}


def ticket_number(s: Scenario) -> str:
    """A stable, realistic-looking ticket number that does not reveal the scenario id."""
    return f"INC{zlib.crc32(s.id.encode()) % 900000 + 100000}"


def show_briefing(
    ui: UI, s: Scenario | None, exam: bool = False, incident: bool = False, objectives_revealed: bool = False
) -> None:
    if not s:
        ui.out(ui.panel("A healthy 4-node DGX H100 cluster managed by BCM. Nothing is graded.", title="Sandbox"))
        return
    body = [s.briefing.rstrip()]
    if s.objectives and (not incident or objectives_revealed):
        body += ["", ui.bold("Objectives:"), *(f"  • {o}" for o in s.objectives)]
    if incident:
        body += ["", INCIDENT_GOAL]
        if not objectives_revealed:
            body.append(
                ui.dim(
                    "The success criteria are hidden because they would point at the cause. "
                    "`objectives` reveals them (this attempt then counts as guided)."
                )
            )
        title = f"Incident report {ticket_number(s)} · {SEVERITY.get(s.difficulty, 'Sev 3')}"
        subtitle = f"{s.domain} · reported via monitoring and user tickets"
    else:
        title = s.title
        subtitle = s.domain if exam else f"{s.domain} · {s.difficulty} · ~{s.time_limit_min} min"
    ui.out("")
    ui.out(ui.panel("\n".join(body), title=title, subtitle=subtitle, style="yellow" if incident else "green"))
    ui.out(ui.dim("Type 'help' for lab commands, 'explain <command>' to learn a command.\n"))


def show_results(ui: UI, results: list[tuple[str, bool, str]], hide_unmet: bool = False) -> None:
    """Print check results. ``hide_unmet`` keeps unmet criteria secret (incident mode)."""
    debug = config.debug()
    rows, styles, unmet = [], [], 0
    for desc, ok, err in results:
        if hide_unmet and not ok:
            unmet += 1
            continue
        detail = f"  ({err})" if err and debug else ""
        if ui.color:
            rows.append(["✓" if ok else "✗", desc + detail])
            styles.append("green" if ok else "red")
        else:
            ui.out(f"  {'PASS' if ok else 'FAIL'}  {desc}{detail}")
    if unmet:
        noun = "criterion" if unmet == 1 else "criteria"
        line = f"{unmet} more success {noun} not met yet"
        if ui.color:
            rows.append(["…", line])
            styles.append("dim")
        else:
            ui.out(f"  ....  {line}")
    if rows:
        ui.out(ui.table(["", "Success criteria"], rows, styles))


def show_solution(ui: UI, s: Scenario, title: str = "Reference solution") -> None:
    if not s.has_solution:
        ui.out(ui.yellow("\n" + solutions.missing_message()))
        return
    steps = "\n".join("$ " + step.rstrip("\n").replace("\n", "\n  ") for step in s.solution)
    ui.out("")
    ui.out(ui.panel(steps, title=title, style="cyan"))
    if s.explanation:
        ui.out(ui.panel(s.explanation.rstrip(), title="Why", style="green"))
    if s.edge_cases:
        ui.out(ui.panel("\n".join(f"• {e}" for e in s.edge_cases), title="Edge cases / exam traps", style="yellow"))
    if s.references:
        ui.out(ui.dim("References: " + "; ".join(s.references)))


def your_commands(session: LabSession, limit: int = 15) -> list[str]:
    """The learner's own shell commands this attempt, without consecutive repeats."""
    out: list[str] = []
    for line in session.shell.ctx.history:
        if not out or out[-1] != line:
            out.append(line)
    return out if len(out) <= limit else out[:limit] + [f"... and {len(out) - limit} more"]


def show_debrief(ui: UI, session: LabSession, passed: bool, identity_hidden: bool) -> None:
    s = session.scenario_required
    if identity_hidden:
        ui.out(ui.bold(f"\nThis incident was: {s.title}") + ui.dim(f"  ({s.id}, ticket {ticket_number(s)})"))
    show_solution(ui, s, title="Debrief: reference solution and edge cases" if passed else "Reference solution")
    mine = your_commands(session)
    if mine:
        ui.out(
            ui.panel(
                "\n".join(f"$ {line}" for line in mine),
                title="Your commands (compare them with the reference)",
                style="blue",
            )
        )
    ui.out(
        ui.dim(
            "Note: the grader checks that your diagnostic commands were run, "
            "not the order you ran them in. Diagnose before you change things."
        )
    )
    done = {sid for sid, rec in progress.load()["scenarios"].items() if rec.get("passed")}
    nxt = sc.similar(s, exclude=done | {s.id})
    if nxt:
        ui.out(
            ui.yellow(
                f"\nPractise next: greenray incident --id {nxt.id}   (a related problem: {', '.join(sorted(set(nxt.tags) & set(s.tags))) or nxt.domain})"
            )
        )


def prompt(ui: UI, session: LabSession) -> str:
    return session.shell.styled_prompt() if ui.color else session.shell.prompt


def run_shell_line(ui: UI, session: LabSession, line: str) -> None:
    try:
        out = session.shell.run(line, heredoc_reader=lambda p: read_line(ui, p))
    except CommandError as e:
        out = str(e)
    if out:
        ui.out(out)


def explain(ui: UI, args: str) -> None:
    name = args.strip().split()[0] if args.strip() else ""
    if not name:
        ui.out("Usage: explain <command>. Commands with a full guide: " + ", ".join(guides.names()))
        return
    text = guides.guide(name)
    if text:
        ui.out(text.rstrip())
    elif name in HELP:
        ui.out(f"{name}: {HELP[name]}\n(No detailed guide yet. Try `{name} --help` or `{name} -h`.)")
    elif name in command_names():
        ui.out(f"{name} is available in this simulator. Try `{name} --help`.")
    else:
        ui.out(f"`{name}` is not a simulated command. `commands` lists what is available.")


def _completion_words() -> list[str]:
    meta = ["task", "check", "hint", "solution", "commands", "reset", "retry", "objectives", "explain", "quit", "back"]
    return command_names() + meta + [HEAD, *COMPUTE_NODES]


def _record(session: LabSession, passed: bool, gave_up: bool, incident: bool) -> None:
    progress.record_scenario(
        session.scenario_required.id,
        passed,
        time.time() - session.started,
        session.hints_used,
        gave_up,
        objectives_revealed=session.objectives_revealed,
        mode="incident" if incident else "guided",
    )


def run_lab(ui: UI, s: Scenario | None, incident: bool = False, hide_identity: bool = False) -> None:
    """Practice a scenario (guided by default, or as an incident); the result is saved to progress."""
    session = LabSession(s, color=ui.color, ui=ui)
    ui.set_completer(_completion_words(), lambda: session.shell.ctx)
    incident = incident and s is not None
    gave_up = passed = False
    show_briefing(ui, s, incident=incident)
    while True:
        line = read_line(ui, prompt(ui, session))
        cmd = line.strip()
        verb, _, rest = cmd.partition(" ")
        if cmd in ("quit", "q"):
            break
        if cmd in ("help", "?"):
            ui.out(INCIDENT_HELP if incident else LAB_HELP)
        elif cmd == "commands":
            ui.out("\n".join(f"  {n:<16} {HELP.get(n, '')}" for n in command_names()))
        elif verb == "explain":
            explain(ui, rest)
        elif s and cmd == "task":
            show_briefing(ui, s, incident=incident, objectives_revealed=session.objectives_revealed)
        elif s and cmd == "reset":
            session.reset()
            ui.out(ui.yellow("Scenario reset to the broken state. Same attempt: hints used and the timer carry over."))
        elif s and incident and cmd == "retry":
            _record(session, False, False, incident)
            session.new_attempt()
            ui.out(ui.yellow(f"Fresh attempt #{session.attempt}: cluster, timer, hints and objectives reset."))
            show_briefing(ui, s, incident=True)
        elif s and incident and cmd == "objectives":
            if (
                not session.objectives_revealed
                and ui.ask(
                    "Reveal the success criteria? They point at the cause, so this attempt counts as guided. [y/N] "
                )
                .strip()
                .lower()
                != "y"
            ):
                continue
            session.objectives_revealed = True
            for o in s.objectives:
                ui.out(f"  • {o}")
        elif s and cmd == "hint":
            n = session.hints_used
            if n >= len(s.hints):
                ui.out(ui.dim("No more hints. Type 'solution' to see the answer."))
                continue
            if (
                n == len(s.hints) - 1
                and len(s.hints) > 1
                and ui.ask("The last hint is close to the answer. Show it? [y/N] ").strip().lower() != "y"
            ):
                continue
            ui.out(ui.yellow(f"Hint {n + 1}/{len(s.hints)}: {s.hints[n]}"))
            session.hints_used += 1
        elif s and cmd in ("solution", "giveup"):
            if ui.ask("Reveal the solution? This ends the attempt as not passed. [y/N] ").strip().lower() == "y":
                gave_up = True
                show_debrief(ui, session, passed=False, identity_hidden=hide_identity)
                break
        elif s and cmd == "check":
            results = session.grade()
            show_results(ui, results, hide_unmet=incident and not session.objectives_revealed)
            mins = (time.time() - session.started) / 60
            if all(ok for _, ok, _ in results):
                passed = True
                how = "with guidance" if session.guided else ("independently" if incident else "without hints")
                ui.out(
                    ui.bold(
                        ui.green(
                            f"\nAll objectives met in {mins:.1f} min, {how} "
                            f"(target {s.time_limit_min} min, hints used: {session.hints_used})."
                        )
                    )
                )
                if mins > s.time_limit_min:
                    ui.out(ui.yellow("Correct, but slower than the target. Repeat it later for speed."))
                show_debrief(ui, session, passed=True, identity_hidden=hide_identity)
                break
            met = sum(ok for _, ok, _ in results)
            ui.out(ui.dim(f"\n{met}/{len(results)} criteria met, {mins:.1f} min elapsed."))
        else:
            run_shell_line(ui, session, line)
    if s:
        _record(session, passed, gave_up, incident)


def pick_incident(domain: str | None = None, tag: str | None = None, seed: int | None = None) -> Scenario | None:
    """A random scenario for incident mode, preferring ones not yet passed independently."""
    pool = [s for s in sc.load_all() if (not domain or s.domain == domain) and (not tag or tag in s.tags)]
    if not pool:
        return None
    done = {sid for sid, rec in progress.load()["scenarios"].items() if rec.get("independent_passes")}
    fresh = [s for s in pool if s.id not in done]
    return random.Random(seed).choice(fresh or pool)


def run_exam_lab(ui: UI, session: LabSession, deadline: float, status: Callable[[], str] | None = None) -> None:
    """Exam mode: no hints or solutions; returns to the exam menu on `back` or timeout.

    ``status`` supplies the rest of the exam status line (questions answered, labs touched).
    """
    ui.set_completer(_completion_words(), lambda: session.shell.ctx)
    show_briefing(ui, session.scenario, exam=True)
    while time.time() < deadline:
        left = int((deadline - time.time()) // 60)
        extra = f" · {status()}" if status else ""
        line = read_line(ui, ui.dim(f"[{left}m left{extra}] ") + prompt(ui, session))
        cmd = line.strip()
        if cmd in ("back", "quit", "q"):
            return
        if cmd in ("help", "?"):
            ui.out(EXAM_LAB_HELP)
        elif cmd == "commands":
            ui.out("\n".join(f"  {n:<16} {HELP.get(n, '')}" for n in command_names()))
        elif cmd == "task":
            show_briefing(ui, session.scenario, exam=True)
        elif cmd.split(" ")[0] in ("check", "hint", "solution", "giveup", "reset", "explain", "objectives", "retry"):
            ui.out(ui.dim("Not available during the exam. Your work is graded when you finish."))
        else:
            run_shell_line(ui, session, line)
    ui.out(ui.red("\nTime is up."))
