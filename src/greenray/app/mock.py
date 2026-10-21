"""Full mock exam in the current NCP-AIO shape: 30 questions + 3 hands-on labs, one clock.

The format (30 multiple-choice questions and 3 lab exercises in 120 minutes, labs on
Slurm, Kubernetes and Base Command Manager) follows the NVIDIA certification page.
NVIDIA does not publish how the two sections are weighted or the cut score; this
mock weights them 50/50 and uses a 70% practice target. Treat the number as a guide.
"""

from __future__ import annotations

import random
import time

from ..content import questions as qb
from ..content import scenarios as sc
from . import progress
from .exam import PRACTICE_PASS_PCT, ask_questions, load_scored_bank, offer_review
from .lab import LabSession, read_line, run_exam_lab, show_results, show_solution
from .ui import UI

N_QUESTIONS = 30
MINUTES = 120
LAB_AREAS = ("slurm", "kubernetes", "bcm")  # one lab per platform the exam labs run on
QUESTION_WEIGHT = 0.5


def pick_labs(seed: int | None = None) -> list[sc.Scenario]:
    """One scenario per area (by tag), never the same scenario twice."""
    rng = random.Random(seed)
    chosen: list[sc.Scenario] = []
    for area in LAB_AREAS:
        pool = [s for s in sc.load_all() if area in s.tags and s not in chosen]
        if pool:
            chosen.append(rng.choice(pool))
    return chosen


def combined_score(question_pct: float, lab_fractions: list[float]) -> float:
    lab_pct = 100.0 * sum(lab_fractions) / len(lab_fractions) if lab_fractions else 0.0
    return QUESTION_WEIGHT * question_pct + (1 - QUESTION_WEIGHT) * lab_pct


def run_mock(ui: UI, seed: int | None = None) -> None:
    bank = load_scored_bank(ui)
    if bank is None:
        return
    qs = qb.pick(bank, N_QUESTIONS, seed=seed)
    labs = [LabSession(s, color=ui.color) for s in pick_labs(seed)]
    start = time.time()
    deadline = start + MINUTES * 60
    answers: dict = {}
    flagged: set = set()

    ui.out(ui.bold(f"\nMock NCP-AIO exam: {len(qs)} questions + {len(labs)} labs, {MINUTES} minutes total"))
    ui.out("Do the sections in any order. Labs keep your work when you go back to the menu.")
    ui.out(
        ui.dim(
            "Format per NVIDIA's exam page (re-check before booking). The 50/50 question/lab weighting "
            f"and the {PRACTICE_PASS_PCT:.0f}% target are this simulator's assumptions, not NVIDIA's rules."
        )
    )
    ui.out(ui.dim("No hints, no solutions, no grading until you finish.\n"))

    while time.time() < deadline:
        left = int((deadline - time.time()) // 60)
        ui.out(ui.bold(f"\n--- Exam menu ({left} min left) ---"))
        ui.out(f"  q) Questions ({len(answers)}/{len(qs)} answered)")
        for i, lab in enumerate(labs, 1):
            ui.out(f"  {i}) Lab {i}: {lab.scenario_required.title}")
        ui.out("  f) Finish and grade")
        choice = read_line(ui, "> ").strip().lower()
        if choice == "q":
            ask_questions(ui, qs, deadline, answers, flagged)
        elif choice.isdigit() and 1 <= int(choice) <= len(labs):
            run_exam_lab(ui, labs[int(choice) - 1], deadline)
        elif choice in ("f", "finish", "quit"):
            if ui.ask("Finish the exam now? [y/N] ").strip().lower() == "y":
                break
    else:
        ui.out(ui.red("\nTime is up. Grading."))

    elapsed = (time.time() - start) / 60
    res = qb.score(qs, answers)
    lab_scores = {lab.scenario_required.id: lab.score() for lab in labs}
    total = combined_score(res["pct"], list(lab_scores.values()))
    verdict = ui.green("PASS") if total >= PRACTICE_PASS_PCT else ui.red("NOT YET")

    ui.out(ui.bold(f"\nResult: {total:.1f}%  {verdict}   ({elapsed:.0f} of {MINUTES} min used)"))
    ui.out(f"  Questions: {res['correct']}/{res['total']} = {res['pct']:.0f}%")
    for d, s in res["per_domain"].items():
        ui.out(f"    {d:<36} {s}")
    for lab in labs:
        ui.out(f"  Lab {lab.scenario_required.id}: {lab_scores[lab.scenario_required.id] * 100:.0f}% of objectives")
        show_results(ui, lab.grade())
    ui.out(
        ui.dim(
            f"(Questions and labs weighted {QUESTION_WEIGHT:.0%}/{1 - QUESTION_WEIGHT:.0%}; "
            "NVIDIA does not publish the real weighting or cut score.)"
        )
    )
    progress.record_exam(
        "mock",
        total,
        res["total"],
        res["correct"],
        res["per_domain"],
        elapsed,
        res["missed"],
        labs={k: round(v, 2) for k, v in lab_scores.items()},
    )

    offer_review(ui, qs, answers, res["missed"], flagged)
    for lab in labs:
        if lab_scores[lab.scenario_required.id] < 1 and ui.ask(
            f"\nShow the solution for lab '{lab.scenario_required.title}'? [Y/n] "
        ).strip().lower() in ("", "y", "yes"):
            show_solution(ui, lab.scenario_required)
