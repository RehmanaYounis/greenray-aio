"""Every scenario must start broken and be solved by its own reference solution."""

import pytest

from aiolab.content import scenarios as sc
from aiolab.content.validate import check_scenario
from conftest import requires_solutions

SCENARIOS = sc.load_all()
IDS = [s.id for s in SCENARIOS]


@requires_solutions
@pytest.mark.parametrize("s", SCENARIOS, ids=IDS)
def test_reference_solution_solves_scenario(s):
    assert s.has_solution, f"{s.id} has no solution in the solutions repo"
    assert check_scenario(s) is None


@pytest.mark.parametrize("s", SCENARIOS, ids=IDS)
def test_public_teaching_content(s):
    assert s.hints, "add at least one hint"
    assert s.objectives, "add objectives"


@requires_solutions
@pytest.mark.parametrize("s", SCENARIOS, ids=IDS)
def test_private_teaching_content(s):
    assert s.explanation.strip(), "add an explanation"
    assert s.edge_cases, "add edge cases / exam traps"
