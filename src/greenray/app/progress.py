"""Local progress store: ``~/.greenray/progress.json`` (override with ``GREENRAY_HOME``)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import config

EMPTY = {"scenarios": {}, "exams": []}


def _path() -> Path:
    return config.home() / "progress.json"


def load() -> dict:
    path = _path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {**EMPTY, **data}
        except json.JSONDecodeError:
            path.rename(path.with_suffix(".corrupt.json"))  # keep it for inspection, start fresh
    return {"scenarios": {}, "exams": []}


def save(data: dict) -> None:
    path = _path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic on POSIX and Windows


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def record_scenario(
    sid: str,
    passed: bool,
    seconds: float,
    hints_used: int,
    gave_up: bool,
    objectives_revealed: bool = False,
    mode: str = "guided",
) -> None:
    """Record one attempt.

    ``clean_passes``: passed without hints (any mode, as before).
    ``independent_passes``: passed in incident mode with no hints and no revealed objectives.
    """
    data = load()
    rec = data["scenarios"].setdefault(
        sid, {"attempts": 0, "passed": False, "best_seconds": None, "clean_passes": 0, "last": None}
    )
    rec.setdefault("independent_passes", 0)
    rec["attempts"] += 1
    rec["last"] = _now()
    if passed and not gave_up:
        rec["passed"] = True
        if hints_used == 0 and not objectives_revealed:
            rec["clean_passes"] += 1
            if mode == "incident":
                rec["independent_passes"] += 1
        if rec["best_seconds"] is None or seconds < rec["best_seconds"]:
            rec["best_seconds"] = round(seconds)
    save(data)


def record_exam(
    kind: str,
    score: float,
    total: int,
    correct: int,
    per_domain: dict,
    minutes: float,
    missed: list,
    labs: dict | None = None,
) -> None:
    """``kind`` is ``questions``, ``drill`` or ``mock``; ``labs`` maps scenario id -> fraction passed."""
    data = load()
    entry = {
        "date": _now(),
        "kind": kind,
        "score": round(score, 1),
        "total": total,
        "correct": correct,
        "per_domain": per_domain,
        "minutes": round(minutes, 1),
        "missed": missed,
    }
    if labs is not None:
        entry["labs"] = labs
    data["exams"].append(entry)
    save(data)
