"""Every scenario, every variant, driven through the lab interface exactly as a learner types it.

The reference solution goes in line by line (heredoc bodies included), then `check`. The lab
must pass, and nothing the learner sees may contain an unfilled placeholder.
"""

import re

import pytest

from conftest import ScriptedUI, requires_solutions
from greenray.app import lab, reports, study
from greenray.content import scenarios as sc

PLACEHOLDER = re.compile(r"\{\w+\}")
VARIANTS = [v for s in sc.load_all() for v in s.all_variants()]
IDS = [f"{v.id}[{v.variant_index + 1}]" if v.variant_index is not None else v.id for v in VARIANTS]


def typed(scenario: sc.Scenario) -> list[str]:
    """The solution as keystrokes: multi-line steps become the command plus its heredoc lines."""
    return [line for step in scenario.solution for line in step.rstrip("\n").split("\n")]


@requires_solutions
@pytest.mark.parametrize("s", VARIANTS, ids=IDS)
def test_guided_lab_passes_when_typed(s):
    ui = ScriptedUI([*typed(s), "check", "", "", ""])
    lab.run_lab(ui, s)
    assert "All objectives met" in ui.text, ui.text[-1500:]
    assert not PLACEHOLDER.search(ui.text), PLACEHOLDER.search(ui.text).group(0)


@requires_solutions
@pytest.mark.parametrize("s", VARIANTS, ids=IDS)
def test_incident_passes_when_typed(s):
    ui = ScriptedUI([*typed(s), "check", "", "", ""])
    lab.run_lab(ui, s, incident=True, hide_identity=True)
    assert "All objectives met" in ui.text and "independently" in ui.text
    assert not PLACEHOLDER.search(ui.text)


@requires_solutions
@pytest.mark.parametrize("raw", [s for s in sc.load_all() if s.variants], ids=lambda s: s.id)
def test_lab_started_without_a_variant_shows_the_variant_it_grades(raw):
    ui = ScriptedUI(["task", "hint", "check", "quit"])
    lab.run_lab(ui, raw)
    assert not PLACEHOLDER.search(ui.text), PLACEHOLDER.search(ui.text).group(0)
    nodes = {v.get("node") or v.get("slow") or v.get("mig") or v.get("c1") for v in raw.variants} - {None}
    if nodes:
        assert any(n in ui.text for n in nodes)


def test_retry_shows_the_next_variants_briefing():
    raw = sc.find("tsh-power-cap-persistence")
    ui = ScriptedUI(["retry", "quit"])
    lab.run_lab(ui, raw.variant(0), incident=True, hide_identity=True)
    assert not PLACEHOLDER.search(ui.text)
    assert raw.variants[1]["node"] in ui.text.split("Fresh attempt #2")[1]


def test_lists_and_recommendations_never_show_placeholders():
    ui = ScriptedUI([])
    reports.list_scenarios(ui)
    study.show_next(ui)
    study.show_plan(ui)
    assert not PLACEHOLDER.search(ui.text)
    assert not any(PLACEHOLDER.search(s.title) for s in sc.load_all())
