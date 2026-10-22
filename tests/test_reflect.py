"""Reflection after a passed lab: optional, saved as notes, never graded, never in the exam."""

import time

from conftest import ScriptedUI, concrete
from greenray.app import lab, reflect
from greenray.content import scenarios as sc

SID = "tsh-power-cap-persistence"
SOLVE = ["ssh dgx-03", "nvidia-smi -q -d PERFORMANCE", "nvidia-smi -pl 700", "nvidia-smi -pm 1", "check"]


def test_answers_are_saved_per_scenario():
    ui = ScriptedUI(
        [*SOLVE, "", "", "the SW Power Cap reason", "it restores the default instead of guessing", "clocks under load"]
    )
    lab.run_lab(ui, concrete(SID))
    path = reflect.note_path(SID)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# ") and "SW Power Cap reason" in text and "variant 1 of 3" in text
    assert reflect.count() == 1
    assert "Saved to" in ui.text


def test_blank_answers_write_nothing():
    ui = ScriptedUI([*SOLVE, "", "", "", "", ""])
    lab.run_lab(ui, concrete(SID))
    assert not reflect.note_path(SID).exists() and reflect.count() == 0


def test_no_reflection_in_the_exam_or_on_a_failed_lab():
    ui = ScriptedUI(SOLVE)
    lab.run_exam_lab(ui, lab.LabSession(concrete(SID)), deadline=time.time() + 600)
    assert not any("evidence told you" in p for p in ui.prompts)
    ui = ScriptedUI(["solution", "y"])
    lab.run_lab(ui, sc.find("adm-mig-h100-1g"))
    assert not any("evidence told you" in p for p in ui.prompts)


def test_debrief_and_reflection_are_offered_not_forced():
    ui = ScriptedUI([*SOLVE, "n", "n"])
    lab.run_lab(ui, concrete(SID))
    assert "All objectives met" in ui.text
    assert "Debrief:" not in ui.text and "Reflect (not graded" not in ui.text
    assert not reflect.note_path(SID).exists()
    ui = ScriptedUI([*SOLVE, "", "n"])
    lab.run_lab(ui, concrete(SID))
    assert "Debrief:" in ui.text and "Reflect (not graded" not in ui.text
