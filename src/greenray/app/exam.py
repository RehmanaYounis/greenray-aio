"""Timed multiple-choice exam (full mock or single-domain drill)."""

from __future__ import annotations

import time

from ..content import questions as qb
from ..content import scenarios as sc
from ..content import solutions
from . import progress
from .ui import UI, TimeUp

PRACTICE_PASS_PCT = 70.0  # NVIDIA does not publish a cut score; this is a practice target


def required_answers(q: dict) -> int:
    """How many choices the question takes: one, or the number its text asks for ("Choose two")."""
    return len(q["answer"])


def read_answer(ui: UI, q: dict, deadline: float, flagged: set, keep: str | None = None) -> str | set[int]:
    """Read one answer to ``q``. Returns the chosen set, or "quit", "skip", "keep" or "timeout".

    An answer with the wrong number of letters is not recorded: the learner stays on the question
    and is told how many it takes, as an exam form would refuse a second radio button.
    """
    need = required_answers(q)
    while True:
        try:
            raw = ui.ask_until("> ", deadline, typed=True).strip().lower()
        except KeyboardInterrupt:
            ui.out("")
            return "quit"
        except TimeUp:
            return "timeout"
        if raw in ("q", "quit"):
            return "quit"
        if raw == "s" and keep is None:
            return "skip"
        if raw == "" and keep is not None:
            return "keep"
        if raw == "f":
            flagged.add(q["id"])
            ui.out(ui.dim("flagged for review"))
            continue
        if raw == "u":
            flagged.discard(q["id"])
            ui.out(ui.dim("flag removed"))
            continue
        ans = qb.parse_answer(raw, len(q["choices"]))
        if ans is None:
            ui.out(ui.dim("Enter letters like B, or A C when the question asks you to choose two."))
            continue
        if len(ans) != need:
            wanted = "exactly one answer" if need == 1 else f"exactly {need} answers"
            ui.out(ui.yellow(f"This question takes {wanted}; you gave {len(ans)}. Nothing was recorded."))
            continue
        return ans


def show_question(ui: UI, qs: list[dict], i: int, deadline: float) -> None:
    q = qs[i]
    left = max(0.0, deadline - time.time())
    ui.out(ui.bold(f"\nQ{i + 1}/{len(qs)}") + ui.dim(f"  [{q['domain']}]  ({int(left // 60)}m left)"))
    ui.out(ui.bold(q["question"].rstrip()))
    for j, c in enumerate(q["choices"]):
        ui.out(f"  {qb.LETTERS[j]}. {c}")


def ask_questions(
    ui: UI,
    qs: list[dict],
    deadline: float,
    answers: dict,
    flagged: set,
    leave: str = "finish early",
    flag_note: str = "f flags for the review after the exam",
) -> None:
    """Ask each unanswered question once, then revisit skipped ones. Mutates ``answers``/``flagged``.

    ``leave`` says where ``q`` takes the learner, since a mock returns to its menu and a drill ends.
    An answer typed after ``deadline`` is not recorded.
    """
    ui.out(ui.dim(f"Answer with letters (B, or A C for 'choose two'). s skips, {flag_note}, q to {leave}."))

    def ask(i: int) -> str:
        if time.time() >= deadline:
            return "quit"
        show_question(ui, qs, i, deadline)
        result = read_answer(ui, qs[i], deadline, flagged)
        if result == "timeout":
            ui.out(ui.red("\nTime is up. That answer was not recorded."))
            return "quit"
        if isinstance(result, set):
            answers[qs[i]["id"]] = result
            return "answered"
        return result

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


def review_before_submit(ui: UI, qs: list[dict], deadline: float, answers: dict, flagged: set) -> None:
    """Let the learner go back to any question before the section is submitted; answers stay changeable."""

    def letters(q: dict) -> str:
        given = answers.get(q["id"])
        return " ".join(qb.LETTERS[a] for a in sorted(given)) if given else "-"

    while time.time() < deadline:
        marked = [i for i, q in enumerate(qs) if q["id"] in flagged]
        open_ = [i for i, q in enumerate(qs) if q["id"] not in answers]
        ui.out(ui.bold("\n--- Review before submitting ---"))
        ui.out(f"  Flagged: {', '.join(f'Q{i + 1}' for i in marked) or 'none'}")
        ui.out(f"  Unanswered: {', '.join(f'Q{i + 1}' for i in open_) or 'none'}")
        ui.out(ui.dim("  Enter a question number to revisit it, f for the flagged ones in turn, or q to go back."))
        try:
            choice = ui.ask_until("review> ", deadline, typed=True).strip().lower()
        except KeyboardInterrupt:
            ui.out("")
            return
        except TimeUp:
            return
        if choice in ("", "q", "quit", "b", "back"):
            return
        if choice == "f":
            todo = marked
            if not todo:
                ui.out(ui.dim("No question is flagged."))
        elif choice.lstrip("q").isdigit() and 1 <= int(choice.lstrip("q")) <= len(qs):
            todo = [int(choice.lstrip("q")) - 1]
        else:
            ui.out(ui.red(f"Unknown choice '{choice}'. Enter a number from 1 to {len(qs)}, f, or q."))
            continue
        for i in todo:
            q = qs[i]
            show_question(ui, qs, i, deadline)
            state = "flagged" if q["id"] in flagged else "not flagged"
            ui.out(
                ui.dim(
                    f"Your answer: {letters(q)} ({state}). New letters change it, Enter keeps it, "
                    "f flags, u removes the flag, q stops."
                )
            )
            result = read_answer(ui, q, deadline, flagged, keep=letters(q))
            if result in ("timeout", "quit"):
                if result == "timeout":
                    ui.out(ui.red("\nTime is up. That answer was not recorded."))
                break
            if isinstance(result, set):
                answers[q["id"]] = result
                ui.out(ui.dim(f"Answer changed to {letters(q)}."))


def review(ui: UI, q: dict, given: set | None) -> None:
    right = ", ".join(qb.LETTERS[a] for a in q["answer"])
    yours = ", ".join(qb.LETTERS[a] for a in sorted(given)) if given else "-"
    ui.out(ui.bold(f"\n[{q['id']}] {q['domain']}"))
    ui.out(q["question"].rstrip())
    for j, c in enumerate(q["choices"]):
        line = f"{qb.LETTERS[j]}. {c}"
        if j in q["answer"]:
            ui.out("  " + ui.green(f"✔ {line}"))
        elif given and j in given:
            ui.out("  " + ui.red(f"✘ {line}"))
        else:
            ui.out(f"    {line}")
    ui.out(f"Your answer: {yours}   Correct: {right}")
    if q.get("explanation"):
        ui.out(ui.blue(q["explanation"].rstrip()))
    if q.get("lab"):
        ui.out(ui.yellow(f"Practise hands-on: greenray lab {sc.find(q['lab']).code}"))


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
    start = time.time()
    answers: dict = {}
    flagged: set = set()
    ask_questions(ui, qs, start + minutes * 60, answers, flagged)
    elapsed = (time.time() - start) / 60
    if not answers:
        ui.out(ui.yellow("\nNo question was answered; nothing is recorded."))
        return
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
