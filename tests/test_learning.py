"""P1 learning experience: incident mode, command help, hints, verification, debrief, progress.

Most tests drive the real lab loop with a scripted terminal, using tsh-power-cap-persistence
(fix: restore the power limit and enable persistence on dgx-03).
"""

import json
import re
import time

import pytest

from conftest import ScriptedUI, concrete, requires_solutions
from greenray.app import lab, progress
from greenray.commands import names as command_names
from greenray.content import guides, solutions
from greenray.content import scenarios as sc
from greenray.core.shell import Shell

SID = "tsh-power-cap-persistence"
DIAGNOSE = [
    "ssh dgx-03",
    "nvidia-smi --query-gpu=index,power.draw,power.limit --format=csv",
    "nvidia-smi -q -d PERFORMANCE",
]
FIX = ["nvidia-smi -pl 700", "nvidia-smi -pm 1"]


def record():
    return progress.load()["scenarios"][SID]


def play(lines, incident=False, hide_identity=False, scenario=SID):
    ui = ScriptedUI(lines)
    lab.run_lab(ui, concrete(scenario), incident=incident, hide_identity=hide_identity)
    return ui


def grade(cmds, scenario=SID):
    s = concrete(scenario)
    sh = Shell(s.build_state())
    for c in cmds:
        sh.run(c)
    return {d: ok for d, ok, _ in s.grade(sh)}


# 1. beginner with hints ------------------------------------------------------------------


def test_beginner_with_hints_passes_as_guided():
    ui = play(["hint", "hint", "hint", "y", *DIAGNOSE, *FIX, "check"])
    assert "Hint 1/3" in ui.text and "Hint 3/3" in ui.text
    assert "with guidance" in ui.text
    rec = record()
    assert rec["passed"] and rec["clean_passes"] == 0 and rec["independent_passes"] == 0


def test_last_hint_needs_confirmation():
    ui = play(["hint", "hint", "hint", "n", "quit"])
    assert any("close to the answer" in p for p in ui.prompts)
    assert "Hint 2/3" in ui.text and "Hint 3/3" not in ui.text
    assert concrete(SID).hints[2] not in ui.text


# 2. experienced learner, independently ----------------------------------------------------


def test_independent_incident_pass():
    ui = play([*DIAGNOSE, *FIX, "check"], incident=True, hide_identity=True)
    assert "independently" in ui.text
    rec = record()
    assert rec["independent_passes"] == 1 and rec["clean_passes"] == 1


# 3. irrelevant commands ------------------------------------------------------------------


def test_irrelevant_commands_change_nothing():
    base = grade([])
    assert grade(["kubectl get nodes", "sinfo", "ssh dgx-01", "nvidia-smi", "exit", "df -h"]) == base


# 4. incorrect configuration change -------------------------------------------------------


def test_wrong_power_limit_is_not_accepted():
    res = grade([*DIAGNOSE, "nvidia-smi -pl 600", "nvidia-smi -pm 1"])
    assert not res["power limit is 700 W on every dgx-03 GPU"]
    assert not all(res.values())


# 5. command help -------------------------------------------------------------------------


def test_explain_command_in_lab():
    ui = play(["explain nvidia-smi", "explain frobnicate", "explain", "quit"])
    assert "What it does" in ui.text and "Useful options" in ui.text
    assert "`frobnicate` is not a simulated command" in ui.text
    assert "Commands with a full guide" in ui.text


def test_every_guide_is_for_a_real_command_and_scenario_agnostic():
    available = set(command_names())
    ids = {s.id for s in sc.load_all()}
    for name in guides.names():
        assert name in available, name
        text = guides.guide(name)
        assert not re.search(r"dgx-0\d|bcm-head", text), name
        assert not any(i in text for i in ids), name


@requires_solutions
def test_guides_do_not_contain_reference_solutions():
    text = "\n".join(guides.guide(n) for n in guides.names())
    for s in sc.load_all():
        for step in s.solution:
            first = step.splitlines()[0].strip()
            if len(first) > 25:  # long, scenario-specific commands only
                assert first not in text, (s.id, first)


def test_explain_is_not_available_in_the_mock_exam():
    ui = ScriptedUI(["explain nvidia-smi", "back"])
    lab.run_exam_lab(ui, lab.LabSession(concrete(SID)), deadline=time.time() + 3600)
    assert "Not available during the exam" in ui.text
    assert "What it does" not in ui.text


# 6. symptom fixed, system not fully healthy ----------------------------------------------


def test_partial_fix_does_not_pass_and_hides_unmet_criteria():
    ui = play([*DIAGNOSE, "nvidia-smi -pl 700", "check", "quit"], incident=True, hide_identity=True)
    assert "1 more success criterion not met yet" in ui.text
    assert "Persistence mode" not in ui.text and "persistence mode enabled" not in ui.text
    assert not record()["passed"]


# 7. reset and retry ----------------------------------------------------------------------


def test_reset_restores_cluster_but_keeps_attempt_guidance():
    s = concrete(SID)
    session = lab.LabSession(s)
    initial = [ok for _, ok, _ in session.grade()]
    for c in [*DIAGNOSE, *FIX]:
        session.shell.run(c)
    session.hints_used = 1
    session.reset()
    assert [ok for _, ok, _ in session.grade()] == initial
    assert session.hints_used == 1  # same attempt: no laundering hints through reset


def test_retry_starts_a_fresh_independent_attempt_on_another_variant():
    other = [c.replace("dgx-03", "dgx-02") for c in DIAGNOSE]  # retry moves to the next variant
    ui = play(["objectives", "y", "retry", *other, *FIX, "check"], incident=True)
    assert "Fresh attempt #2" in ui.text and "variant 2 of 3" in ui.text
    rec = record()
    assert rec["attempts"] == 2 and rec["independent_passes"] == 1 and rec["variants_seen"] == [0, 1]


def test_revealing_objectives_makes_the_attempt_guided():
    ui = play(["objectives", "y", *DIAGNOSE, *FIX, "check"], incident=True)
    assert "with guidance" in ui.text
    assert record()["independent_passes"] == 0


# 8. multiple valid resolutions -----------------------------------------------------------


@pytest.mark.parametrize(
    "fix",
    [
        ["nvidia-smi -pl 700", "nvidia-smi -pm 1"],
        ["nvidia-smi -i 0,1,2,3,4,5,6,7 -pl 700", "nvidia-smi -pm ENABLED"],
        ["nvidia-smi -pl 700", "systemctl enable --now nvidia-persistenced"],
        ["nvidia-smi -pl 700", "systemctl start nvidia-persistenced"],
    ],
    ids=["pm-1", "explicit-ids", "daemon-enable-now", "daemon-start"],
)
def test_valid_alternatives_pass(fix):
    assert all(grade([*DIAGNOSE, *fix]).values())


def test_remote_commands_from_head_node_pass():
    cmds = [
        "ssh dgx-03 nvidia-smi -q -d PERFORMANCE",
        "ssh dgx-03 nvidia-smi -pl 700",
        "ssh dgx-03 nvidia-smi -pm 1",
    ]
    assert all(grade(cmds).values())


def test_persistence_daemon_state_is_consistent_evidence():
    sh = Shell(concrete(SID).build_state())
    sh.run("ssh dgx-03")
    assert "inactive" in sh.run("systemctl status nvidia-persistenced")
    sh.run("systemctl stop nvidia-persistenced")
    sh.run("nvidia-smi -pm 1")
    sh.run("reboot")  # -pm does not survive a reboot while the daemon is disabled
    assert not sh.ctx.state.nodes["dgx-03"].gpus[0].persistence_mode


def test_mig_alternative_syntax_passes():
    cmds = [
        "ssh dgx-02",
        "nvidia-smi",
        "kill 48211",
        "nvidia-smi -i 0 -mig 1",
        "nvidia-smi mig -i 0 -cgi 1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb,1g.10gb",
        "nvidia-smi mig -i 0 -cci",
    ]
    assert all(grade(cmds, scenario="adm-mig-h100-1g").values())


# 10. no premature leaks ------------------------------------------------------------------


def test_incident_mode_hides_identity_and_objectives_until_debrief():
    s = concrete(SID)
    ui = play(["task", "check", "quit"], incident=True, hide_identity=True)
    text = ui.text
    assert s.title not in text and s.id not in text
    for objective in s.objectives:
        assert objective not in text
    assert "Your goal:" in text


def test_guided_lab_still_shows_title_and_objectives():
    s = concrete(SID)
    ui = play(["quit"])
    assert s.title in ui.text and all(o in ui.text for o in s.objectives)


def test_debrief_reveals_identity_and_shows_learner_commands():
    ui = play([*DIAGNOSE, *FIX, "check"], incident=True, hide_identity=True)
    s = concrete(SID)
    assert f"This incident was: {s.title}" in ui.text
    assert "Your commands" in ui.text and "$ nvidia-smi -pl 700" in ui.text
    assert "Practise next: greenray incident --id" in ui.text


@requires_solutions
def test_solution_only_after_pass_or_explicit_request():
    s = concrete(SID)
    marker = s.explanation.strip().splitlines()[0]
    assert marker not in play(["hint", "check", "quit"]).text
    assert marker in play(["solution", "y"]).text


@requires_solutions
def test_progress_file_contains_no_solution_text():
    play([*DIAGNOSE, *FIX, "check"], incident=True)
    blob = json.dumps(progress.load())
    s = concrete(SID)
    assert s.explanation.strip().splitlines()[0] not in blob
    assert not any(step in blob for step in s.solution if len(step) > 15)


def test_bundled_solutions_back_the_debrief(monkeypatch, tmp_path):
    monkeypatch.setenv("GREENRAY_SOLUTIONS", str(tmp_path / "nowhere"))
    monkeypatch.setattr(solutions, "_REPO_ROOT", tmp_path)
    solutions.clear_cache()
    try:
        ui = play([*DIAGNOSE, *FIX, "check"], incident=True, hide_identity=True)
        assert "All objectives met" in ui.text
        assert "Why" in ui.text and "Solutions are not available" not in ui.text
    finally:
        solutions.clear_cache()
