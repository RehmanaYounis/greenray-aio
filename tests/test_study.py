"""`greenray next` recommends sensibly and `greenray plan` covers every scenario once."""

from conftest import ScriptedUI
from greenray.app import progress, study
from greenray.content import scenarios as sc


def test_fresh_profile_starts_with_the_on_ramp():
    picks = study.recommend(limit=6)
    assert [p.scenario.id for p in picks] == list(study.ON_RAMP_ORDER)
    assert picks[0].command == "greenray lab tsh-1"
    assert sc.find("tsh-1").id == study.ON_RAMP_ORDER[0]
    assert "on-ramp" in picks[0].why


def test_weak_domain_comes_first_once_the_on_ramp_is_done():
    for sid in study.ON_RAMP_ORDER:
        progress.record_scenario(sid, True, 60, 0, False)
    per_domain = {d: "9/10" for d in sc.DOMAINS}
    per_domain["Workload Management"] = "2/10"
    progress.record_exam("questions", 72.0, 40, 29, per_domain, 50.0, [])
    first = study.recommend(limit=1)[0]
    assert first.scenario.domain == "Workload Management" and first.scenario.difficulty == "easy"
    assert "weakest domain" in first.why


def test_guided_passes_are_offered_as_incidents_when_nothing_is_left():
    for s in sc.load_all():
        progress.record_scenario(s.id, True, 60, 1, False)  # passed, but with a hint
    picks = study.recommend(limit=2)
    assert all(p.command.startswith("greenray incident --id ") for p in picks)
    assert "unguided" in picks[0].why


def test_everything_done_points_at_the_mock():
    for s in sc.load_all():
        progress.record_scenario(s.id, True, 60, 0, False, mode="incident")
    ui = ScriptedUI([])
    study.show_next(ui)
    assert "greenray mock" in ui.text


def test_plan_covers_every_scenario_exactly_once():
    labs = [
        a.removeprefix("lab ") for _, _, days in study.plan() for acts in days for a in acts if a.startswith("lab ")
    ]
    assert sorted(labs) == sorted(s.code for s in sc.load_all())
    assert len(study.plan()) == 4 and all(len(days) == study.DAYS_PER_WEEK for _, _, days in study.plan())


def test_plan_marks_passed_labs():
    progress.record_scenario("tsh-first-look-at-logs", True, 60, 0, False)
    ui = ScriptedUI([])
    study.show_plan(ui)
    line = next(l for l in ui.output if "lab tsh-1 " in l or l.endswith("lab tsh-1"))
    assert "✔" in line
    assert "Week 4" in ui.text and "mock" in ui.text
