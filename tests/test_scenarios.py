"""Every scenario must start broken and be solved by its own reference solution."""

import pytest

from conftest import requires_solutions
from greenray.content import scenarios as sc
from greenray.content.validate import check_scenario

SCENARIOS = sc.load_all()
IDS = [s.id for s in SCENARIOS]
VARIANTS = [v for s in SCENARIOS for v in s.all_variants()]
VARIANT_IDS = [f"{v.id}[{v.variant_index + 1}]" if v.variant_index is not None else v.id for v in VARIANTS]


@requires_solutions
@pytest.mark.parametrize("s", VARIANTS, ids=VARIANT_IDS)
def test_reference_solution_solves_scenario(s):
    assert s.has_solution, f"{s.id} has no reference solution"
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
