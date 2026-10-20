"""Scenario list and progress report."""

from __future__ import annotations

from ..content import scenarios as sc
from ..content.domains import DOMAIN_WEIGHTS, DOMAINS
from . import progress
from .ui import UI


def list_scenarios(ui: UI) -> None:
    prog = progress.load()["scenarios"]
    scenarios = sc.load_all()
    for d in DOMAINS:
        ui.out(ui.bold(f"\n{d}  ({int(DOMAIN_WEIGHTS[d] * 100)}% of exam)"))
        for s in (s for s in scenarios if s.domain == d):
            p = prog.get(s.id)
            mark = ui.green("✔") if p and p["passed"] else (ui.yellow("…") if p else " ")
            ui.out(f"  {mark} {s.id:<38} {s.title}  {ui.dim(s.difficulty)}")


def show_progress(ui: UI) -> None:
    data = progress.load()
    scenarios = sc.load_all()
    ui.out(ui.bold("\nHands-on scenarios by domain"))
    for d in DOMAINS:
        ids = [s.id for s in scenarios if s.domain == d]
        done = sum(bool(data["scenarios"].get(i, {}).get("passed")) for i in ids)
        clean = sum(bool(data["scenarios"].get(i, {}).get("clean_passes")) for i in ids)
        alone = sum(bool(data["scenarios"].get(i, {}).get("independent_passes")) for i in ids)
        ui.out(
            f"  {d:<36} passed {done}/{len(ids)}   without hints {clean}/{len(ids)}   "
            f"as an incident, unguided {alone}/{len(ids)}"
        )
    todo = [s.id for s in scenarios if not data["scenarios"].get(s.id, {}).get("passed")]
    if todo:
        ui.out(ui.dim(f"  next up: {', '.join(todo[:5])}"))

    ui.out(ui.bold("\nExams"))
    if not data["exams"]:
        ui.out("  none yet. Try: aiolab mock")
    for e in data["exams"][-10:]:
        domains = "  ".join(f"{d.split()[0]}:{v}" for d, v in e["per_domain"].items())
        ui.out(
            f"  {e['date']}  {e.get('kind', 'questions'):<9} {e['score']:5.1f}%  "
            f"({e['correct']}/{e['total']} Q, {e['minutes']} min)  {domains}"
        )

    totals: dict[str, list[int]] = {}
    for e in data["exams"][-5:]:
        for d, v in e["per_domain"].items():
            c, t = map(int, v.split("/"))
            totals.setdefault(d, [0, 0])
            totals[d][0] += c
            totals[d][1] += t
    if totals:
        worst, (c, t) = min(totals.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1]))
        ui.out(
            ui.yellow(
                f"\nWeakest domain (last 5 exams): {worst} ({100 * c / max(1, t):.0f}%). "
                f'Drill it: aiolab exam -d "{worst}"'
            )
        )
