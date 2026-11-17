"""App flows from the 1.5.1 audit: prompts that must not lose an attempt, and inputs that must be refused."""

from __future__ import annotations

import json

import pytest

from conftest import ScriptedUI, concrete, requires_solutions
from greenray.app import exam, lab, mock, progress
from greenray.cli import main

SID = "tsh-power-cap-persistence"


@requires_solutions
def test_ctrl_c_at_the_solution_prompt_keeps_the_lab_running():
    ui = ScriptedUI(["solution", KeyboardInterrupt(), "hint", "quit"])
    lab.run_lab(ui, concrete(SID))
    assert "Hint 1/" in ui.text
    rec = progress.load()["scenarios"][SID]
    assert rec["attempts"] == 1 and not rec["passed"]


@requires_solutions
def test_ctrl_c_at_the_objectives_prompt_in_an_incident():
    ui = ScriptedUI(["objectives", KeyboardInterrupt(), "check", "quit"])
    lab.run_lab(ui, concrete(SID), incident=True, hide_identity=True)
    assert "more success" in ui.text  # criteria still hidden: the reveal was declined


def test_ctrl_c_at_a_mock_confirmation_returns_to_the_menu():
    ui = ScriptedUI(["f", KeyboardInterrupt(), "x", "y"])
    mock.run_mock(ui, labs_only=True)
    assert "Exam abandoned" in ui.text and "Result:" not in ui.text


def test_mock_ends_cleanly_at_end_of_input():
    ui = ScriptedUI([])
    mock.run_mock(ui, labs_only=True)
    assert ui.text.count("Exam menu") <= 2


def test_labs_only_mock_explains_q():
    ui = ScriptedUI(["q", "x", "y"])
    mock.run_mock(ui, labs_only=True)
    assert "no questions" in ui.text and "Enter q" not in ui.text


def test_drill_records_nothing_when_nothing_was_answered():
    ui = ScriptedUI(["q"])
    exam.run_exam(ui, n=5, minutes=5)
    assert "nothing is recorded" in ui.text and progress.load()["exams"] == []


def test_drill_arguments_are_validated(capsys):
    for bad in ("0", "-3"):
        with pytest.raises(SystemExit):
            main(["exam", "-n", bad])
    assert "at least 1" in capsys.readouterr().err


def test_progress_file_shapes_are_tolerated(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text("[]")
    assert progress.load() == {"scenarios": {}, "exams": []}
    path.write_text(json.dumps({"scenarios": {SID: {"attempts": 1, "passed": True}}, "exams": [{"score": 50}]}))
    data = progress.load()
    assert data["scenarios"][SID]["clean_passes"] == 0 and data["exams"][0]["per_domain"] == {}
    progress.record_scenario(SID, passed=True, seconds=10, hints_used=0, gave_up=False)
    assert progress.load()["scenarios"][SID]["clean_passes"] == 1


@requires_solutions
def test_guided_debrief_does_not_offer_retry():
    s = concrete(SID)
    ui = ScriptedUI(
        [*[line for step in s.solution for line in step.rstrip("\n").split("\n")], "check", "", "", "", "", ""]
    )
    lab.run_lab(ui, s)
    assert "All objectives met" in ui.text
    assert "`retry` does too" not in ui.text
    assert "where an objective says so" in ui.text


def test_checks_true_from_the_start_are_marked():
    s = concrete("adm-slurm-partition-time-limit")
    ui = ScriptedUI(["check", "quit"])
    lab.run_lab(ui, s)
    assert "true from the start" in ui.text
