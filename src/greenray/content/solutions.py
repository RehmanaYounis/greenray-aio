"""Where reference solutions and answer keys come from.

Layout, both bundled and external::

    scenarios/<scenario-id>.yaml   solution, explanation, edge_cases
    questions/<bank-file>.yaml     {question-id: {answer: [...], explanation: ...}}

The package ships a complete set under ``data/solutions``. An external directory
takes precedence, file by file, so authors can work on unreleased content:
``$GREENRAY_SOLUTIONS`` if set, else ``<repo>/solutions``, else a sibling or
home-directory clone named ``greenray-aio-solutions``.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

from .. import config

REPO_NAME = "greenray-aio-solutions"
BUNDLED = Path(__file__).resolve().parent.parent / "data" / "solutions"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def candidates() -> list[Path]:
    """External locations in precedence order; the bundled copy is the fallback."""
    if explicit := config.env("SOLUTIONS"):
        return [Path(explicit).expanduser()]
    return [_REPO_ROOT / "solutions", _REPO_ROOT.parent / REPO_NAME, Path.home() / REPO_NAME]


def _is_solutions_dir(path: Path) -> bool:
    return (path / "scenarios").is_dir() or (path / "questions").is_dir()


def solutions_dir() -> Path | None:
    for p in candidates():
        if _is_solutions_dir(p):
            return p
    return BUNDLED if _is_solutions_dir(BUNDLED) else None


def available() -> bool:
    return solutions_dir() is not None


def source() -> str:
    """Human-readable origin for `greenray doctor`."""
    found = solutions_dir()
    if found is None:
        return "none"
    return "bundled" if found == BUNDLED else str(found)


def missing_message() -> str:
    return (
        "Solutions are not available. Reinstall the package, or point GREENRAY_SOLUTIONS at a solutions "
        "directory. Searched:\n" + "\n".join(f"  {p}" for p in [*candidates(), BUNDLED])
    )


@cache
def _load(rel: str) -> dict:
    """An external directory overrides file by file, so it can carry only unreleased content."""
    base = solutions_dir()
    if base is None:
        return {}
    for path in (base / rel, BUNDLED / rel):
        if path.is_file():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def scenario_solution(scenario_id: str) -> dict:
    """``{solution, explanation, edge_cases}`` for a scenario, or ``{}``."""
    return _load(f"scenarios/{scenario_id}.yaml")


def answer_key(bank_file: str) -> dict:
    """``{question_id: {answer, explanation}}`` for one question-bank file, or ``{}``."""
    return _load(f"questions/{bank_file}")


def clear_cache() -> None:
    _load.cache_clear()
