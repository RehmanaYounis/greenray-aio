"""Locate the private solutions repository.

Reference solutions, explanations and exam answer keys are kept out of the
public repository. They live in a separate (private) checkout with this layout::

    scenarios/<scenario-id>.yaml   solution, explanation, edge_cases
    questions/<bank-file>.yaml     {question-id: {answer: [...], explanation: ...}}
    labs/...                       SOLUTION.md files for the real-hardware labs

The first existing directory wins:

1. ``$AIOLAB_SOLUTIONS`` (if set, nothing else is searched)
2. ``<repo>/solutions/``              (git-ignored local copy)
3. ``<repo>/../NVIDIA-NCP-AIO-Prep-solutions/``  (sibling clone)
4. ``~/NVIDIA-NCP-AIO-Prep-solutions/``
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml

REPO_NAME = "NVIDIA-NCP-AIO-Prep-solutions"
# src/aiolab/content/solutions.py -> repository root
_REPO_ROOT = Path(__file__).resolve().parents[4]


def candidates() -> list[Path]:
    """Search order. An explicit ``$AIOLAB_SOLUTIONS`` is the only place searched."""
    if env := os.environ.get("AIOLAB_SOLUTIONS"):
        return [Path(env).expanduser()]
    return [
        _REPO_ROOT / "solutions",
        _REPO_ROOT.parent / REPO_NAME,
        _REPO_ROOT.parent / REPO_NAME.lower(),
        Path.home() / REPO_NAME,
    ]


def solutions_dir() -> Path | None:
    for p in candidates():
        if (p / "scenarios").is_dir() or (p / "questions").is_dir():
            return p
    return None


def available() -> bool:
    return solutions_dir() is not None


def missing_message() -> str:
    return (
        "Solutions are not installed. Clone your private solutions repository next to this one:\n"
        f"  git clone https://github.com/<you>/{REPO_NAME}.git\n"
        "or point AIOLAB_SOLUTIONS at it. Searched:\n" + "\n".join(f"  {p}" for p in candidates())
    )


@cache
def _load(rel: str) -> dict:
    base = solutions_dir()
    if base is None:
        return {}
    path = base / rel
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def scenario_solution(scenario_id: str) -> dict:
    """``{solution, explanation, edge_cases}`` for a scenario, or ``{}``."""
    return _load(f"scenarios/{scenario_id}.yaml")


def answer_key(bank_file: str) -> dict:
    """``{question_id: {answer, explanation}}`` for one question-bank file, or ``{}``."""
    return _load(f"questions/{bank_file}")


def clear_cache() -> None:
    _load.cache_clear()
