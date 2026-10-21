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
        items = [s for s in scenarios if s.domain == d]
        rows, styles = [], []
        for s in items:
            p = prog.get(s.id)
            mark = "✔" if p and p["passed"] else ("…" if p else " ")
            rows.append([mark, s.id, s.title, s.difficulty])
            styles.append("green" if mark == "✔" else ("yellow" if mark == "…" else ""))
        ui.out("")
        if ui.color:
            ui.out(ui.bold(f"{d}  ({int(DOMAIN_WEIGHTS[d] * 100)}% of exam)"))
            ui.out(ui.table(["", "Scenario", "Title", "Level"], rows, styles))
        else:
            ui.out(f"{d}  ({int(DOMAIN_WEIGHTS[d] * 100)}% of exam)")
            for mark, sid, title, level in rows:
                ui.out(f"  {mark} {sid:<38} {title}  {level}")


def show_progress(ui: UI) -> None:
    data = progress.load()
    scenarios = sc.load_all()
    rows = []
    for d in DOMAINS:
        ids = [s.id for s in scenarios if s.domain == d]
        done = sum(bool(data["scenarios"].get(i, {}).get("passed")) for i in ids)
        clean = sum(bool(data["scenarios"].get(i, {}).get("clean_passes")) for i in ids)
        alone = sum(bool(data["scenarios"].get(i, {}).get("independent_passes")) for i in ids)
        rows.append(
            [d, f"{ui.bar(done, len(ids), 12)} {done}/{len(ids)}", f"{clean}/{len(ids)}", f"{alone}/{len(ids)}"]
        )
    ui.out("")
    if ui.color:
        ui.out(ui.table(["Domain", "Passed", "Without hints", "Unguided incidents"], rows))
    else:
        ui.out("Hands-on scenarios by domain")
        for domain, passed_cell, clean_cell, alone_cell in rows:
            ui.out(
                f"  {domain:<36} passed {passed_cell.split()[-1]}   without hints {clean_cell}   "
                f"as an incident, unguided {alone_cell}"
            )
    todo = [s.id for s in scenarios if not data["scenarios"].get(s.id, {}).get("passed")]
    if todo:
        ui.out(ui.dim(f"  next up: {', '.join(todo[:5])}"))

    ui.out(ui.bold("\nExams"))
    if not data["exams"]:
        ui.out("  none yet. Try: greenray mock")
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
                f'Drill it: greenray exam -d "{worst}"'
            )
        )
