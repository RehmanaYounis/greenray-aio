"""Content structure checks that run without the private solutions (public CI)."""

from pathlib import Path

import pytest
import yaml

from conftest import requires_solutions
from greenray.content import questions as qb
from greenray.content import scenarios as sc
from greenray.content.validate import coverage_ids, hint_leaks, structural_problems


def test_no_structural_problems():
    assert structural_problems() == []


def test_coverage_map_lists_every_scenario():
    assert {s.id for s in sc.load_all()} <= coverage_ids()


@pytest.mark.parametrize("path", sorted(sc.SCENARIO_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_public_scenario_has_no_solution(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert not set(sc.PRIVATE_KEYS) & set(data)


@pytest.mark.parametrize("path", sorted(qb.BANK_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_public_questions_have_no_answers(path: Path):
    for q in yaml.safe_load(path.read_text(encoding="utf-8")):
        assert not qb.PRIVATE_KEYS & set(q), q["id"]


@requires_solutions
def test_hints_never_quote_a_fix_command():
    assert hint_leaks() == []
