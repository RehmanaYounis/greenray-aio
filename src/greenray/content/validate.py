"""Content self-checks shared by ``greenray validate`` and the test-suite."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

from ..core.shell import Shell
from . import questions as qb
from . import scenarios as sc
from .domains import DOMAINS

COVERAGE_DOC = Path(__file__).resolve().parents[3] / "docs" / "coverage.md"


def check_scenario(s: sc.Scenario) -> str | None:
    """``None`` if the scenario starts unsolved and its reference solution solves it.

    Any exception (bad setup op, broken check, replay crash) is reported, not raised,
    so one broken scenario cannot hide the results for the others.
    """
    try:
        return _check_scenario(s)
    except Exception as e:
        return f"raised {type(e).__name__}: {e}"


def _check_scenario(s: sc.Scenario) -> str | None:
    shell = Shell(s.build_state())
    if all(ok for _, ok, _ in s.grade(shell)):
        return "already solved before any action"
    sc.replay_solution(s, shell)
    failed = [desc + (f" ({err})" if err else "") for desc, ok, err in s.grade(shell) if not ok]
    return f"solution leaves checks failing: {failed}" if failed else None


def fix_steps(s: sc.Scenario) -> list[str]:
    """Solution steps that make a non-evidence check pass: the commands that change the cluster."""
    shell = Shell(s.build_state())
    evidence = {c.get("desc", c["expr"]) for c in s.checks if "ran(" in c["expr"]}
    passing = {d for d, ok, _ in s.grade(shell) if ok}
    steps = []
    for step in s.solution:
        lines = step.split("\n")
        feed = iter(lines[1:])
        with contextlib.suppress(Exception):  # replay failures are check_scenario's job, not this one's
            shell.run(lines[0], heredoc_reader=lambda _p, feed=feed: next(feed, None))
        now = {d for d, ok, _ in s.grade(shell) if ok}
        if (now - passing) - evidence:
            steps.append(lines[0].strip())
        passing |= now
    return steps


def hint_leaks(scenarios: list[sc.Scenario] | None = None) -> list[str]:
    """Hints that quote a fix command verbatim. Naming a tool is fine; handing over the command is not."""
    problems = []
    for s in scenarios or sc.load_all():
        if not s.has_solution:
            continue
        fixes = [f for f in fix_steps(s) if " " in f]  # a bare command name is a pointer, not a recipe
        for i, hint in enumerate(s.hints, 1):
            for f in fixes:
                if f in hint:
                    problems.append(f"{s.id}: hint {i} quotes the fix step {f!r}")
    return problems


def coverage_ids() -> set[str]:
    """Scenario ids referenced in docs/coverage.md."""
    if not COVERAGE_DOC.is_file():
        return set()
    return set(re.findall(r"`((?:ins|adm|wkl|tsh)-[a-z0-9-]+)`", COVERAGE_DOC.read_text(encoding="utf-8")))


def structural_problems() -> list[str]:
    """Problems that don't need the private solutions (safe to run in public CI)."""
    problems = []
    scenarios = sc.load_all()
    ids = {s.id for s in scenarios}
    missing_domains = set(DOMAINS) - {s.domain for s in scenarios}
    if missing_domains:
        problems.append(f"domains without scenarios: {sorted(missing_domains)}")
    for s in scenarios:
        if not s.hints:
            problems.append(f"{s.id}: no hints")
        if not s.objectives:
            problems.append(f"{s.id}: no objectives")
    bank = qb.load_bank()
    if set(DOMAINS) - {q["domain"] for q in bank}:
        problems.append("some domains have no questions")
    for q in bank:
        if q.get("lab") and q["lab"] not in ids:
            problems.append(f"question {q['id']} links to unknown lab {q['lab']}")
    for sid in sorted(coverage_ids() - ids):
        problems.append(f"coverage.md lists missing scenario {sid}")
    return problems
