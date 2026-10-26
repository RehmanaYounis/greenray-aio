"""Full mock exam in the current NCP-AIO shape: 30 questions + 3 hands-on labs, one clock.

The format (30 multiple-choice questions and 3 lab exercises in 120 minutes, labs on
Slurm, Kubernetes and Base Command Manager) follows the NVIDIA certification page.
NVIDIA does not publish how the two sections are weighted or the cut score; this
mock weights them 50/50 and uses a 70% practice target. Treat the number as a guide.
``labs_only`` keeps the three labs and a 40-minute clock for exam-day pacing.
"""

from __future__ import annotations

import random
import time

from ..content import questions as qb
from ..content import scenarios as sc
from ..content.domains import DOMAINS
from . import progress
from .exam import PRACTICE_PASS_PCT, ask_questions, load_scored_bank, offer_review
from .lab import LabSession, run_exam_lab, show_results, show_solution
from .study import ON_RAMP
from .ui import UI

N_QUESTIONS = 30
MINUTES = 120
LABS_ONLY_MINUTES = 40
LAB_AREAS = ("slurm", "kubernetes", "bcm")  # one lab per platform the exam labs run on
QUESTION_WEIGHT = 0.5


def pick_labs(seed: int | None = None) -> list[sc.Scenario]:
    """One scenario per area (by tag), never the same scenario twice. On-ramp labs are too easy for an exam."""
    rng = random.Random(seed)
    chosen: list[sc.Scenario] = []
    for area in LAB_AREAS:
        pool = [s for s in sc.load_all() if area in s.tags and ON_RAMP not in s.tags and s not in chosen]
        if pool:
            chosen.append(rng.choice(pool))
    return chosen


def combined_score(question_pct: float, lab_fractions: list[float]) -> float:
    lab_pct = 100.0 * sum(lab_fractions) / len(lab_fractions) if lab_fractions else 0.0
    return QUESTION_WEIGHT * question_pct + (1 - QUESTION_WEIGHT) * lab_pct


def report_card(ui: UI, qs: list[dict], res: dict, labs: list[LabSession], lab_scores: dict[str, float]) -> None:
    """Per-domain bars over questions and labs together, then what to practise."""
    cards: list[tuple[str, str, str, float | None]] = []
    for d in DOMAINS:
        parts: list[float] = []
        q_cell = res["per_domain"].get(d, "")
        if q_cell:
            c, t = map(int, q_cell.split("/"))
            parts.append(c / max(1, t))
        in_domain = [lab_scores[lab.scenario_required.id] for lab in labs if lab.scenario_required.domain == d]
        lab_cell = ""
        if in_domain:
            parts.append(sum(in_domain) / len(in_domain))
            lab_cell = f"{parts[-1]:.0%}"
        cards.append((d, q_cell, lab_cell, 100 * sum(parts) / len(parts) if parts else None))

    rows = [
        [d, q, lab, f"{ui.bar(round(pct), 100, 12)} {pct:3.0f}%" if pct is not None else "no data"]
        for d, q, lab, pct in cards
    ]
    ui.out("")
    if ui.color:
        ui.out(ui.table(["Domain", "Questions", "Labs", "Combined"], rows))
    else:
        for d, q_cell, lab_cell, bar in rows:
            ui.out(f"  {d:<36} questions {q_cell or '-':>5}  labs {lab_cell or '-':>4}  {bar}")

    todo = [f"greenray lab {lab.scenario_required.code}" for lab in labs if lab_scores[lab.scenario_required.id] < 1]
    linked = {sc.find(q["lab"]).code for q in qs if q["id"] in res["missed"] and q.get("lab")}
    todo += [f"greenray lab {code}" for code in sorted(linked) if f"greenray lab {code}" not in todo]
    todo = todo[:5]
    scored = [(d, pct) for d, _, _, pct in cards if pct is not None]
    if scored:
        worst = min(scored, key=lambda dp: dp[1])[0]
        todo.append(f'greenray exam -n 20 -m 40 -d "{worst}"')
    if todo:
        ui.out(ui.yellow("\nPractise next:"))
        for item in todo:
            ui.out(f"  {item}")


def run_mock(ui: UI, seed: int | None = None, labs_only: bool = False) -> None:
    qs: list[dict] = []
    if not labs_only:
        bank = load_scored_bank(ui)
        if bank is None:
            return
        qs = qb.pick(bank, N_QUESTIONS, seed=seed)
    labs = [LabSession(s, color=ui.color, ui=ui) for s in pick_labs(seed)]
    minutes = LABS_ONLY_MINUTES if labs_only else MINUTES
    start = time.time()
    deadline = start + minutes * 60
    answers: dict = {}
    flagged: set = set()
    touched: set[int] = set()

    if labs_only:
        ui.out(ui.bold(f"\nLabs-only mock: {len(labs)} labs, {minutes} minutes total (exam-day pacing)"))
    else:
        ui.out(ui.bold(f"\nMock NCP-AIO exam: {len(qs)} questions + {len(labs)} labs, {minutes} minutes total"))
    ui.out("Do the sections in any order. Labs keep your work when you go back to the menu.")
    ui.out(
        ui.dim(
            "Format per NVIDIA's exam page (re-check before booking). The 50/50 question/lab weighting "
            f"and the {PRACTICE_PASS_PCT:.0f}% target are this simulator's assumptions, not NVIDIA's rules."
        )
    )
    ui.out(ui.dim("No hints, no solutions, no grading until you finish.\n"))

    with ui.private_history():
        while time.time() < deadline:
            left = int((deadline - time.time()) // 60)
            ui.out(ui.bold(f"\n--- Exam menu ({left} min left) ---"))
            if qs:
                ui.out(f"  q) Questions ({len(answers)}/{len(qs)} answered)")
            for i, lab in enumerate(labs, 1):
                ui.out(f"  {i}) Lab {i}: {lab.scenario_required.title}")
            ui.out("  f) Finish and grade")
            ui.out("  x) Leave without grading")
            try:
                choice = ui.ask("> ").strip().lower()
            except KeyboardInterrupt:  # Ctrl-C at the menu is a request to leave, not a cancelled line
                ui.out("")
                choice = "x"
            if choice == "q" and qs:
                ask_questions(ui, qs, deadline, answers, flagged, leave="return to the exam menu")
            elif choice.isdigit() and 1 <= int(choice) <= len(labs):
                lab_no = int(choice)
                touched.add(lab_no)

                def status(n: int = lab_no) -> str:
                    q_part = f"Q {len(answers)}/{len(qs)} · " if qs else ""
                    return f"{q_part}lab {n}/{len(labs)}"

                run_exam_lab(ui, labs[lab_no - 1], deadline, status=status)
            elif choice in ("f", "finish"):
                if ui.ask("Finish the exam now? [y/N] ").strip().lower() == "y":
                    break
            elif choice in ("x", "exit", "quit", "abandon"):
                if ui.ask("Leave the exam? Nothing is graded or recorded. [y/N] ").strip().lower() == "y":
                    ui.out(ui.dim("Exam abandoned."))
                    return
            elif choice:
                ui.out(ui.red(f"Unknown choice '{choice}'. Enter q, a lab number, f to finish, or x to leave."))
        else:
            ui.out(ui.red("\nTime is up. Grading."))

    elapsed = (time.time() - start) / 60
    res = qb.score(qs, answers)
    # a lab that was never opened scores nothing: its untouched invariants are not the learner's work
    lab_scores = {lab.scenario_required.id: lab.score() if i in touched else 0.0 for i, lab in enumerate(labs, 1)}
    lab_pct = 100.0 * sum(lab_scores.values()) / max(1, len(lab_scores))
    total = lab_pct if labs_only else combined_score(res["pct"], list(lab_scores.values()))
    verdict = ui.green("PASS") if total >= PRACTICE_PASS_PCT else ui.red("NOT YET")

    ui.out(ui.bold(f"\nResult: {total:.1f}%  {verdict}   ({elapsed:.0f} of {minutes} min used)"))
    if qs:
        ui.out(f"  Questions: {res['correct']}/{res['total']} = {res['pct']:.0f}%")
    for i, lab in enumerate(labs, 1):
        if i not in touched:
            ui.out(f"  Lab {lab.scenario_required.id}: not attempted")
            continue
        ui.out(f"  Lab {lab.scenario_required.id}: {lab_scores[lab.scenario_required.id] * 100:.0f}% of objectives")
        show_results(ui, lab.grade(), kinds=lab.scenario_required.kinds)
    if not labs_only:
        ui.out(
            ui.dim(
                f"(Questions and labs weighted {QUESTION_WEIGHT:.0%}/{1 - QUESTION_WEIGHT:.0%}; "
                "NVIDIA does not publish the real weighting or cut score.)"
            )
        )
    report_card(ui, qs, res, labs, lab_scores)
    progress.record_exam(
        "labs" if labs_only else "mock",
        total,
        res["total"],
        res["correct"],
        res["per_domain"],
        elapsed,
        res["missed"],
        labs={k: round(v, 2) for k, v in lab_scores.items()},
    )

    if qs:
        offer_review(ui, qs, answers, res["missed"], flagged)
    for lab in labs:
        if lab_scores[lab.scenario_required.id] < 1 and ui.ask(
            f"\nShow the solution for lab '{lab.scenario_required.title}'? [Y/n] "
        ).strip().lower() in ("", "y", "yes"):
            show_solution(ui, lab.scenario_required)
