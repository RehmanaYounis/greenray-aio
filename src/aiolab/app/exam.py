"""Timed multiple-choice exam (full mock or single-domain drill)."""

from __future__ import annotations

import time

from ..content import questions as qb
from ..content import solutions
from . import progress
from .ui import UI

PRACTICE_PASS_PCT = 70.0  # NVIDIA does not publish a cut score; this is a practice target


def ask_questions(ui: UI, qs: list[dict], deadline: float, answers: dict, flagged: set) -> None:
    """Ask each unanswered question once, then revisit skipped ones. Mutates ``answers``/``flagged``."""

    def ask(i: int) -> str:
        q = qs[i]
        left = deadline - time.time()
        if left <= 0:
            return "quit"
        ui.out(ui.bold(f"\nQ{i + 1}/{len(qs)}  [{q['domain']}]  ({int(left // 60)}m left)"))
        ui.out(q["question"].rstrip())
        for j, c in enumerate(q["choices"]):
            ui.out(f"  {qb.LETTERS[j]}. {c}")
        while True:
            raw = ui.ask("> ").strip().lower()
            if raw in ("q", "quit"):
                return "quit"
            if raw == "s":
                return "skip"
            if raw == "f":
                flagged.add(q["id"])
                ui.out(ui.dim("flagged for review"))
                continue
            ans = qb.parse_answer(raw, len(q["choices"]))
            if ans is None:
                ui.out(ui.dim("Enter letters like B, or A C when the question asks you to choose two."))
                continue
            answers[q["id"]] = ans
            return "answered"

    pending = [i for i, q in enumerate(qs) if q["id"] not in answers]
    for attempt in range(2):
        skipped = []
        for i in pending:
            result = ask(i)
            if result == "quit":
                return
            if result == "skip":
                skipped.append(i)
        if not skipped:
            return
        if attempt == 0:
            ui.out(ui.yellow(f"\nGoing back to {len(skipped)} skipped question(s)."))
        pending = skipped


def review(ui: UI, q: dict, given: set | None) -> None:
    right = ", ".join(qb.LETTERS[a] for a in q["answer"])
    yours = ", ".join(qb.LETTERS[a] for a in sorted(given)) if given else "-"
    ui.out(ui.bold(f"\n[{q['id']}] {q['domain']}"))
    ui.out(q["question"].rstrip())
    for j, c in enumerate(q["choices"]):
        mark = ui.green("✔") if j in q["answer"] else (ui.red("✘") if given and j in given else " ")
        ui.out(f"  {mark} {qb.LETTERS[j]}. {c}")
    ui.out(f"Your answer: {yours}   Correct: {right}")
    if q.get("explanation"):
        ui.out(ui.dim(q["explanation"].rstrip()))
    if q.get("lab"):
        ui.out(ui.yellow(f"Practise hands-on: aiolab lab {q['lab']}"))


def offer_review(ui: UI, qs: list[dict], answers: dict, missed: list[str], flagged: set) -> None:
    to_review = [q for q in qs if q["id"] in missed or q["id"] in flagged]
    if to_review and ui.ask("\nReview explanations for missed/flagged questions? [Y/n] ").strip().lower() in (
        "",
        "y",
        "yes",
    ):
        for q in to_review:
            review(ui, q, answers.get(q["id"]))


def load_scored_bank(ui: UI) -> list[dict] | None:
    bank = qb.load_bank()
    if not bank:
        ui.out(ui.red("The question bank is empty."))
        return None
    if not qb.has_answers(bank):
        ui.out(ui.red("The exam needs the answer key to score you.\n") + solutions.missing_message())
        return None
    return bank


def run_exam(ui: UI, n: int = 60, minutes: int = 120, domain: str | None = None) -> None:
    bank = load_scored_bank(ui)
    if bank is None:
        return
    qs = qb.pick(bank, n, domain)
    scope = f"domain: {domain}" if domain else "weighted by the official blueprint"
    ui.out(ui.bold(f"\nQuestion drill: {len(qs)} questions, {minutes} minutes, {scope}"))
    ui.out("Answer with letters (B, or A C for 'choose two'). 's' skip, 'f' flag, 'q' finish early.\n")
    start = time.time()
    answers: dict = {}
    flagged: set = set()
    ask_questions(ui, qs, start + minutes * 60, answers, flagged)
    elapsed = (time.time() - start) / 60
    res = qb.score(qs, answers)
    verdict = ui.green("PASS") if res["pct"] >= PRACTICE_PASS_PCT else ui.red("NOT YET")
    ui.out(
        ui.bold(
            f"\nScore: {res['correct']}/{res['total']} = {res['pct']:.1f}%  {verdict}  "
            f"(practice target {PRACTICE_PASS_PCT:.0f}%, {elapsed:.0f} min)"
        )
    )
    for d, s in res["per_domain"].items():
        ui.out(f"  {d:<36} {s}")
    progress.record_exam(
        "drill" if domain else "questions",
        res["pct"],
        res["total"],
        res["correct"],
        res["per_domain"],
        elapsed,
        res["missed"],
    )
    offer_review(ui, qs, answers, res["missed"], flagged)
