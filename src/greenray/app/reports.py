"""Scenario list and progress report."""

from __future__ import annotations

from ..content import scenarios as sc
from ..content.domains import DOMAIN_WEIGHTS, DOMAINS
from . import progress, reflect, study
from .ui import UI


def list_scenarios(ui: UI, at_prompt: bool = False) -> None:
    """Every lab by exam domain, easy ones first. ``at_prompt``: a prompt for the code follows.

    Codes are fixed names, so a lab added later has a high number whatever its level;
    grouping by level keeps the list readable without ever renumbering.
    """
    prog = progress.load()["scenarios"]
    scenarios = sc.load_all()
    for d in DOMAINS:
        in_domain = sorted((s for s in scenarios if s.domain == d), key=sc.code_order)
        rows, styles = [], []
        for level in sc.DIFFICULTIES:
            group = [s for s in in_domain if s.difficulty == level]
            if not group:
                continue
            rows.append(["", ui.level(level, level.capitalize()), "", ""])
            styles.append("")
            for s in group:
                p = prog.get(s.id)
                mark = "✔" if p and p["passed"] else ("…" if p else " ")
                rows.append([mark, s.code, s.id, s.title])
                styles.append("green" if mark == "✔" else ("yellow" if mark == "…" else ""))
        ui.out("")
        if ui.color:
            ui.out(ui.bold(f"{d}  ({int(DOMAIN_WEIGHTS[d] * 100)}% of exam)"))
            ui.out(ui.table(["", "Code", "Scenario", "Title"], rows, styles))
        else:
            ui.out(f"{d}  ({int(DOMAIN_WEIGHTS[d] * 100)}% of exam)")
            for mark, code, sid, title in rows:
                ui.out(f"    {mark} {code:<7} {sid:<42} {title}" if sid else f"  {code}")
    # the hint has to be what works where it is read: a shell command, or just the code at the menu's prompt
    ui.out(ui.dim(f"\nStart one with its code or id: {'tsh-1' if at_prompt else 'greenray lab tsh-1'}"))


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
    nxt = study.recommend(data, limit=1)
    if nxt:
        ui.out(ui.dim(f"  next up: {nxt[0].command}"))
    if reflect.count():
        ui.out(ui.dim(f"  your notes: {reflect.count()} scenario(s) in {reflect.notes_dir()}"))

    ui.out(ui.bold("\nExams"))
    if not data["exams"]:
        ui.out("  none yet. Try: greenray mock")
    for e in data["exams"][-10:]:
        domains = "  ".join(f"{d.split()[0]}:{v}" for d, v in e["per_domain"].items())
        ui.out(
            f"  {e['date']}  {e.get('kind', 'questions'):<9} {e['score']:5.1f}%  "
            f"({e['correct']}/{e['total']} Q, {e['minutes']} min)  {domains}"
        )

    weak = study.weakest_domain(data)
    if weak:
        worst, pct = weak
        ui.out(
            ui.yellow(f'\nWeakest domain (last 5 exams): {worst} ({pct:.0f}%). Drill it: greenray exam -d "{worst}"')
        )
