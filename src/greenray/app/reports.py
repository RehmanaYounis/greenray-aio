"""Scenario list and progress report."""

from __future__ import annotations

from ..content import scenarios as sc
from ..content.domains import DOMAIN_WEIGHTS, DOMAINS
from . import progress, reflect, study
from .ui import OPEN, PASSED, STARTED, UI


def list_scenarios(ui: UI, at_prompt: bool = False) -> None:
    """Every lab by exam domain, easy ones first. ``at_prompt``: a prompt for the code follows.

    Codes are fixed names, so a lab added later has a high number whatever its level;
    ordering by level keeps the list readable without ever renumbering.
    """
    data = progress.load()
    prog = data["scenarios"]
    scenarios = sc.load_all()
    passed = sum(bool(prog.get(s.id, {}).get("passed")) for s in scenarios)
    share = round(100 * passed / len(scenarios)) if scenarios else 0
    ui.out("")
    ui.out(
        ui.tiles(
            [
                ("Labs", str(len(scenarios)), ""),
                ("Passed", str(passed), "green"),
                ("Progress", f"{share}%  {ui.bar(passed, len(scenarios), 14)}", ""),
            ]
        )
    )
    for d in DOMAINS:
        labs = sorted(
            (s for s in scenarios if s.domain == d),
            key=lambda s: (sc.DIFFICULTIES.index(s.difficulty), sc.code_order(s)),
        )
        rows = []
        for s in labs:
            p = prog.get(s.id)
            mark = PASSED if p and p["passed"] else (STARTED if p else OPEN)
            rows.append((mark, s.code, s.name or s.title, s.difficulty))
        done = sum(mark == PASSED for mark, *_ in rows)
        ui.out("")
        ui.out(ui.lab_panel(d, f"{int(DOMAIN_WEIGHTS[d] * 100)}% of exam · {done}/{len(rows)} passed", rows))
    nxt = study.recommend(data, limit=1)
    if nxt:
        s = nxt[0].scenario
        body = f"{ui.bold(s.name or s.title)}\n{s.title}\n{ui.dim(' · '.join(s.tags))}"
        ui.out("")
        ui.out(ui.panel(body, title=f"Next up · {s.code}", subtitle=f"{s.difficulty} · ~{s.time_limit_min} min"))
    # the hint has to be what works where it is read: a shell command, or just the code at the menu's prompt
    ui.out(ui.dim(f"\nStart one with its code: {'tsh-1' if at_prompt else 'greenray lab tsh-1'}"))


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
