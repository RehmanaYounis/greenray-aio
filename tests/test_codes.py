"""Short lab codes: stable, validated, accepted wherever an id is, and shown where ids are shown."""

import pytest

from conftest import ScriptedUI
from greenray import cli
from greenray.app import progress, reports, study
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


def listed(ui: ScriptedUI) -> list[tuple[str, str]]:
    """(level, code) for every lab line of a printed list, in the order shown."""
    found = []
    for line in "\n".join(ui.output).splitlines():
        words = [w for w in line.split() if w not in ("passed", "started")]
        if words and words[0].startswith(("tsh-", "adm-", "wkl-", "ins-")) and words[-1] in sc.DIFFICULTIES:
            found.append((words[-1], words[0]))
    return found


def test_list_groups_each_domain_by_level_with_codes_in_order():
    ui = ScriptedUI([])
    reports.list_scenarios(ui)
    shown = listed(ui)
    assert sorted(code for _, code in shown) == sorted(s.code for s in sc.load_all())
    assert all(sc.find(code).difficulty == level for level, code in shown)
    for prefix in ("ins", "adm", "wkl", "tsh"):
        mine = [(level, code) for level, code in shown if code.startswith(prefix)]
        levels = [level for level, _ in mine]
        assert levels == sorted(levels, key=sc.DIFFICULTIES.index)
        for level in sc.DIFFICULTIES:
            numbers = [int(code.split("-")[1]) for lvl, code in mine if lvl == level]
            assert numbers == sorted(numbers)
    assert "greenray lab tsh-1" in ui.text


def test_a_lab_added_later_is_listed_with_its_level_not_at_the_end():
    ui = ScriptedUI([])
    reports.list_scenarios(ui)
    adm = [code for _, code in listed(ui) if code.startswith("adm-")]
    assert sc.find("adm-14").difficulty == "easy" and sc.find("adm-3").difficulty == "medium"
    assert adm.index("adm-14") < adm.index("adm-3")


def test_the_list_shows_short_names_progress_and_what_to_do_next():
    progress.record_scenario("tsh-first-look-at-logs", True, 120, 0, False)
    progress.record_scenario("tsh-which-gpu-is-which", False, 120, 1, True)
    ui = ScriptedUI([])
    reports.list_scenarios(ui)
    text = ui.text
    assert "Labs: 53" in text and "Passed: 1" in text and "Progress: 2%" in text
    assert "Troubleshooting and Optimization  (23% of exam · 1/19 passed)" in text
    lines = {line.split()[0]: line.split() for line in text.splitlines() if line.strip().startswith("tsh-")}
    assert lines["tsh-1"][1:] == ["Find", "why", "a", "job", "ended", "easy", "passed"]
    assert lines["tsh-2"][-2:] == ["easy", "started"]
    assert lines["tsh-3"][-1] == "easy"  # a lab not opened yet has no status word
    assert "tsh-first-look-at-logs" not in text  # the long ids and symptom titles stay out of the rows
    assert "Next up" in text and "greenray lab tsh-1" in text


def test_every_lab_has_a_short_name_that_fits_the_list():
    assert validate.name_problems(sc.load_all()) == []
    base = sc.load_all()[0]
    assert validate.name_problems([sc.replace(base, name="")])
    assert validate.name_problems([sc.replace(base, name="x" * 35)])
    assert any("already used" in p for p in validate.name_problems([base, sc.replace(base, id="tsh-other")]))


def test_the_hint_under_the_list_is_what_works_where_it_is_shown():
    ui = ScriptedUI([])
    reports.list_scenarios(ui)
    assert "Start one with its code: greenray lab tsh-1" in ui.text
    ui = ScriptedUI(["1", "", "q"])
    cli.menu(ui)
    assert "Start one with its code: tsh-1" in ui.text and "greenray lab tsh-1" not in ui.text


@pytest.mark.parametrize("typed", ["greenray lab tsh-3", "lab tsh-3", "  TSH-3  "])
def test_the_lab_prompt_in_the_menu_takes_the_whole_command(typed):
    ui = ScriptedUI(["1", typed, "quit", "q"])
    cli.menu(ui)
    assert "find the clock limiter" in ui.text and "No scenario" not in ui.text


def test_clear_at_the_menu_is_not_an_unknown_choice():
    ui = ScriptedUI(["clear", "cls", "q"])
    cli.menu(ui)
    assert "Unknown choice" not in ui.text
    assert ui.text.count("Main menu") == 3  # the menu is drawn again after each


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
