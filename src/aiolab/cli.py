"""aiolab: NCP-AIO hands-on practice lab.

aiolab                     interactive menu
aiolab list                scenarios by domain, with your status
aiolab lab <id>            hands-on scenario (practice mode)
aiolab mock                full mock exam: 30 questions + 3 labs, 120 minutes
aiolab exam [-n] [-m] [-d] question drill
aiolab sandbox             healthy cluster, free practice
aiolab progress            scores and weakest domain
aiolab validate            content self-test
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .app.exam import run_exam
from .app.lab import run_lab
from .app.mock import run_mock
from .app.reports import list_scenarios, show_progress
from .app.ui import UI
from .content import scenarios as sc
from .content import solutions
from .content.domains import DOMAINS
from .content.validate import check_scenario, structural_problems


def menu(ui: UI) -> None:
    while True:
        ui.out(ui.bold("\nNCP-AIO Practice Lab"))
        ui.out(
            "  1) Hands-on scenarios\n  2) Full mock exam (30 questions + 3 labs, 120 min)\n"
            "  3) Question drill (60 questions, blueprint-weighted)\n  4) Domain drill (20 questions)\n"
            "  5) Sandbox (healthy cluster)\n  6) Progress\n  q) Quit"
        )
        choice = ui.ask("> ").strip().lower()
        if choice == "1":
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
        elif choice in ("q", "quit"):
            return


def validate(ui: UI) -> int:
    problems = structural_problems()
    for p in problems:
        ui.out(ui.red(f"BAD {p}"))
    if not solutions.available():
        ui.out(
            ui.yellow("Solutions not installed: skipped replaying reference solutions.\n") + solutions.missing_message()
        )
    else:
        for s in sc.load_all():
            err = check_scenario(s) if s.has_solution else "no reference solution in the solutions repo"
            ui.out(f"{ui.red('BAD') if err else ui.green('ok ')} {s.id}" + (f": {err}" if err else ""))
            problems += [err] if err else []
    ui.out(f"\n{len(sc.load_all())} scenarios, {'no' if not problems else len(problems)} problems")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aiolab", description="NCP-AIO hands-on practice lab")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list", help="list scenarios by domain")
    lab = sub.add_parser("lab", help="start a hands-on scenario")
    lab.add_argument("id")
    sub.add_parser("mock", help="full mock exam: 30 questions + 3 labs in 120 minutes")
    ex = sub.add_parser("exam", help="timed question drill")
    ex.add_argument("-n", type=int, default=60, help="number of questions (default 60)")
    ex.add_argument("-m", type=int, default=120, help="minutes (default 120)")
    ex.add_argument("-d", "--domain", choices=DOMAINS)
    sub.add_parser("sandbox", help="healthy cluster, nothing graded")
    sub.add_parser("progress", help="your scores and weakest domain")
    sub.add_parser("validate", help="check scenarios and questions")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ui = UI()
    try:
        if args.cmd is None:
            menu(ui)
        elif args.cmd == "list":
            list_scenarios(ui)
        elif args.cmd == "lab":
            run_lab(ui, sc.find(args.id))
        elif args.cmd == "mock":
            run_mock(ui)
        elif args.cmd == "exam":
            run_exam(ui, n=args.n, minutes=args.m, domain=args.domain)
        elif args.cmd == "sandbox":
            run_lab(ui, None)
        elif args.cmd == "progress":
            show_progress(ui)
        elif args.cmd == "validate":
            return validate(ui)
    except KeyError as e:
        ui.out(ui.red(f"Unknown scenario {e}. Run: aiolab list"))
        return 1
    except KeyboardInterrupt:
        ui.out("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
