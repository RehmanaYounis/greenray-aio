"""greenray: NCP-AIO hands-on practice lab.

greenray                     interactive menu
greenray list                scenarios by domain, with your status
greenray lab <id>            hands-on scenario (practice mode)
greenray mock [--labs-only]  full mock exam: 30 questions + 3 labs, 120 minutes
greenray exam [-n] [-m] [-d] question drill
greenray sandbox             healthy cluster, free practice
greenray progress            scores and weakest domain
greenray next                what to practise next
greenray plan                four-week study schedule
greenray validate            content self-test
"""

from __future__ import annotations

import argparse
import sys

from . import __version__, config
from .app.exam import run_exam
from .app.lab import pick_incident, run_lab
from .app.mock import run_mock
from .app.reports import list_scenarios, show_progress
from .app.study import show_next, show_plan
from .app.ui import UI
from .content import scenarios as sc
from .content import solutions
from .content.domains import DOMAINS
from .content.validate import check_scenario, structural_problems

MENU = """\
0  Incident        investigate a random problem; root cause hidden
1  Labs            pick a hands-on scenario (guided)
2  Mock exam       30 questions + 3 labs in 120 minutes
3  Question drill  60 questions, weighted like the blueprint
4  Domain drill    20 questions from one domain
5  Sandbox         a healthy cluster to explore
6  Progress        what you have passed and your weakest domain
7  Next            what to practise next
8  Plan            a four-week study schedule
q  Quit"""


def menu(ui: UI) -> None:
    ui.out(ui.banner(__version__, "Hands-on labs for AI infrastructure operations · NCP-AIO exam prep"))
    while True:
        ui.out("")
        ui.out(ui.panel(MENU, title="Main menu"))
        choice = ui.ask("> ").strip().lower()
        if choice == "0":
            for i, d in enumerate(DOMAINS, 1):
                ui.out(f"  {i}) {d}")
            k = ui.ask("Domain (blank = any): ").strip()
            domain = DOMAINS[int(k) - 1] if k.isdigit() and 1 <= int(k) <= len(DOMAINS) else None
            start_incident(ui, domain=domain)
        elif choice == "1":
            list_scenarios(ui)
            sid = ui.ask("\nScenario id (blank to go back): ").strip()
            if sid:
                try:
                    run_lab(ui, sc.find(sid))
                except KeyError:
                    ui.out(ui.red(f"No scenario '{sid}'."))
        elif choice == "2":
            run_mock(ui)
        elif choice == "3":
            run_exam(ui)
        elif choice == "4":
            for i, d in enumerate(DOMAINS, 1):
                ui.out(f"  {i}) {d}")
            k = ui.ask("> ").strip()
            if k.isdigit() and 1 <= int(k) <= len(DOMAINS):
                run_exam(ui, n=20, minutes=40, domain=DOMAINS[int(k) - 1])
        elif choice == "5":
            run_lab(ui, None)
        elif choice == "6":
            show_progress(ui)
        elif choice == "7":
            show_next(ui)
        elif choice == "8":
            show_plan(ui)
        elif choice in ("q", "quit"):
            return


def start_incident(ui: UI, domain: str | None = None, tag: str | None = None, sid: str | None = None) -> int:
    """Incident mode. A random pick hides the id and title; ``--id`` is for a deliberate retry."""
    if sid:
        run_lab(ui, sc.find(sid), incident=True, hide_identity=False)
        return 0
    s = pick_incident(domain=domain, tag=tag)
    if s is None:
        ui.out(
            ui.red("No scenario matches that domain/topic. Topics are scenario tags, e.g. slurm, kubernetes, bcm, mig.")
        )
        return 1
    run_lab(ui, s, incident=True, hide_identity=True)
    return 0


def doctor(ui: UI) -> int:
    """Environment report to paste into bug reports."""
    import platform
    import shutil
    import sys

    from .commands.editor import _external_editor
    from .content import scenarios as sc

    editor = _external_editor("nano")
    rows = [
        ("GreenRay AIO", __version__),
        ("Python", f"{platform.python_version()} ({sys.executable})"),
        ("OS", platform.platform()),
        (
            "Terminal",
            f"{'interactive' if ui.interactive else 'not a tty'}, colour {'on' if ui.color else 'off'}, "
            f"{shutil.get_terminal_size((80, 24)).columns} columns",
        ),
        ("Editor", " ".join(editor) if editor else "none found (inline editor will be used)"),
        ("Solutions", solutions.source()),
        ("Progress", str(config.home() / "progress.json")),
        ("Scenarios", str(len(sc.load_all()))),
    ]
    for k, v in rows:
        ui.out(f"{k:<13} {v}")
    return 0


def validate(ui: UI) -> int:
    problems = structural_problems()
    for p in problems:
        ui.out(ui.red(f"BAD {p}"))
    if not solutions.available():
        ui.out(
            ui.yellow("Solutions not available: skipped replaying reference solutions.\n") + solutions.missing_message()
        )
    else:
        for s in sc.load_all():
            err = check_scenario(s) if s.has_solution else "no reference solution in the solutions repo"
            ui.out(f"{ui.red('BAD') if err else ui.green('ok ')} {s.id}" + (f": {err}" if err else ""))
            problems += [err] if err else []
    ui.out(f"\n{len(sc.load_all())} scenarios, {'no' if not problems else len(problems)} problems")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="greenray", description="NCP-AIO hands-on practice lab")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--no-color", action="store_true", help="plain output (also honours NO_COLOR)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list", help="list scenarios by domain")
    lab = sub.add_parser("lab", help="start a hands-on scenario")
    lab.add_argument("id")
    inc = sub.add_parser("incident", help="investigate a random incident; root cause, id and title stay hidden")
    inc.add_argument("-d", "--domain", choices=DOMAINS)
    inc.add_argument("-t", "--topic", help="scenario tag, e.g. slurm, kubernetes, bcm, mig, infiniband")
    inc.add_argument("--id", help="retry a specific scenario as an incident (its id is known, objectives stay hidden)")
    mock = sub.add_parser("mock", help="full mock exam: 30 questions + 3 labs in 120 minutes")
    mock.add_argument("--labs-only", action="store_true", help="the 3 labs alone in 40 minutes (exam-day pacing)")
    ex = sub.add_parser("exam", help="timed question drill")
    ex.add_argument("-n", type=int, default=60, help="number of questions (default 60)")
    ex.add_argument("-m", type=int, default=120, help="minutes (default 120)")
    ex.add_argument("-d", "--domain", choices=DOMAINS)
    sub.add_parser("sandbox", help="healthy cluster, nothing graded")
    sub.add_parser("progress", help="your scores and weakest domain")
    sub.add_parser("next", help="what to practise next")
    sub.add_parser("plan", help="a four-week study schedule covering every scenario")
    sub.add_parser("validate", help="check scenarios and questions")
    sub.add_parser("doctor", help="show version, environment and where solutions come from")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ui = UI(color=not args.no_color, history_file=config.home() / "history")
    try:
        if args.cmd is None:
            menu(ui)
        elif args.cmd == "list":
            list_scenarios(ui)
        elif args.cmd == "incident":
            return start_incident(ui, domain=args.domain, tag=args.topic, sid=args.id)
        elif args.cmd == "lab":
            run_lab(ui, sc.find(args.id))
        elif args.cmd == "mock":
            run_mock(ui, labs_only=args.labs_only)
        elif args.cmd == "exam":
            run_exam(ui, n=args.n, minutes=args.m, domain=args.domain)
        elif args.cmd == "sandbox":
            run_lab(ui, None)
        elif args.cmd == "progress":
            show_progress(ui)
        elif args.cmd == "next":
            show_next(ui)
        elif args.cmd == "plan":
            show_plan(ui)
        elif args.cmd == "validate":
            return validate(ui)
        elif args.cmd == "doctor":
            return doctor(ui)
    except KeyError as e:
        ui.out(ui.red(f"Unknown scenario {e}. Run: greenray list"))
        return 1
    except KeyboardInterrupt:
        ui.out("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
