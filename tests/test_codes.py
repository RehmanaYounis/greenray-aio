"""Short lab codes: stable, validated, accepted wherever an id is, and shown where ids are shown."""

import pytest

from conftest import ScriptedUI
from greenray import cli
from greenray.app import reports, study
from greenray.content import scenarios as sc
from greenray.content import validate


def test_every_scenario_has_a_well_formed_unique_code():
    assert validate.code_problems(sc.load_all()) == []
    codes = [s.code for s in sc.load_all()]
    assert len(set(codes)) == len(codes)


def test_code_problems_catch_gaps_duplicates_and_wrong_prefixes():
    base = sc.find("tsh-first-look-at-logs")
    wrong = sc.replace(base, id="adm-x", code="tsh-1")
    gap = [sc.replace(base, code="tsh-1"), sc.replace(base, code="tsh-3")]
    dup = [sc.replace(base, code="tsh-1"), sc.replace(base, code="tsh-1")]
    assert any("adm-N" in p for p in validate.code_problems([wrong]))
    assert any("gaps" in p for p in validate.code_problems(gap))
    assert any("gaps" in p for p in validate.code_problems(dup))
    assert validate.code_problems([sc.replace(base, code="")])


def test_find_accepts_code_or_id_and_ignores_case():
    s = sc.find("tsh-first-look-at-logs")
    assert sc.find("tsh-1") is s and sc.find("TSH-1") is s
    with pytest.raises(KeyError):
        sc.find("tsh-999")


def test_on_ramp_labs_lead_each_domain():
    assert sc.find("tsh-1").id == study.ON_RAMP_ORDER[0]
    assert sc.find("adm-1").id == "adm-bcm-cmsh-inventory"
    assert sc.find("wkl-1").id == "wkl-slurm-first-gpu-job"
    assert sc.find("ins-1").difficulty == "easy"


def test_list_shows_codes_in_order():
    ui = ScriptedUI([])
    reports.list_scenarios(ui)
    lines = [line for line in ui.output if line.strip().startswith(("tsh-", "adm-", "wkl-", "ins-"))]
    tsh = [line.split()[0] for line in lines if line.split()[0].startswith("tsh-")]
    assert tsh == [f"tsh-{i}" for i in range(1, len(tsh) + 1)]
    assert "greenray lab tsh-1" in ui.text


def test_menu_takes_a_code():
    ui = ScriptedUI(["1", "tsh-3", "quit", "q"])
    cli.menu(ui)
    assert "Lab code or id" in ui.prompts[1]
    assert "find the clock limiter" in ui.text and "tsh-3 · " in ui.text
    ui = ScriptedUI(["lab TSH-3", "quit", "q"])
    cli.menu(ui)
    assert "find the clock limiter" in ui.text
    ui = ScriptedUI(["1", "3", "q"])
    cli.menu(ui)
    assert "No scenario '3'" in ui.text and "tsh-1" in ui.text


def test_recommendations_and_plan_use_codes():
    picks = study.recommend(limit=3)
    assert all(p.command.split()[-1] == p.scenario.code for p in picks)
    acts = [a for _, _, days in study.plan() for day in days for a in day if a.startswith("lab ")]
    assert len(acts) == len(sc.load_all()) and all(sc.find(a[4:]) for a in acts)
