"""Three questions after a passed lab, answered in the learner's own words and kept as notes.

Nothing here is graded. Explaining the evidence, the choice of fix and the verification is
what turns a solved lab into something that transfers to the next incident.
"""

from __future__ import annotations

import time
from pathlib import Path

from .. import config
from ..content.scenarios import Scenario
from .ui import UI

QUESTIONS = (
    "Which piece of evidence told you the root cause?",
    "Why was your fix safer than the alternative you did not choose?",
    "What would you verify before declaring recovery complete?",
)


def notes_dir() -> Path:
    return config.home() / "notes"


def note_path(scenario_id: str) -> Path:
    return notes_dir() / f"{scenario_id}.md"


def save(scenario: Scenario, answers: list[str]) -> Path | None:
    """Append the answered questions to the scenario's note file; nothing is written when all are blank."""
    if not any(a.strip() for a in answers):
        return None
    path = note_path(scenario.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    head = f"## {stamp}" + (f" ({scenario.variant_label})" if scenario.variant_label else "")
    body = "".join(f"**{q}**\n\n{a.strip() or '-'}\n\n" for q, a in zip(QUESTIONS, answers))
    with path.open("a", encoding="utf-8") as f:
        if path.stat().st_size == 0:
            f.write(f"# {scenario.title}\n\n")
        f.write(f"{head}\n\n{body}")
    return path


def count() -> int:
    d = notes_dir()
    return len(list(d.glob("*.md"))) if d.is_dir() else 0


def run(ui: UI, scenario: Scenario) -> None:
    """Ask the three questions; Enter skips one, and all blank skips the note."""
    ui.out("")
    ui.out(ui.bold("Reflect (not graded; Enter to skip a question):"))
    answers = []
    for q in QUESTIONS:
        try:
            answers.append(ui.ask(f"  {q}\n  > "))
        except KeyboardInterrupt:
            ui.out("")
            return
    path = save(scenario, answers)
    if path:
        ui.out(ui.dim(f"Saved to {path}"))
