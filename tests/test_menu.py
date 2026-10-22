"""The interactive menu guides instead of ignoring: help, names, pasted commands, unknown input."""

from conftest import ScriptedUI
from greenray import cli


def run(lines):
    ui = ScriptedUI(lines)
    cli.menu(ui)
    return ui


def test_first_run_points_at_next_and_help_explains_the_commands():
    ui = run(["help", "q"])
    assert "First time here? Choose 7" in ui.text
    assert "greenray next" in ui.text and "greenray --help" in ui.text


def test_unknown_input_is_answered_and_the_menu_continues():
    ui = run(["frobnicate", "q"])
    assert "Unknown choice 'frobnicate'" in ui.text
    assert ui.text.count("Main menu") == 2


def test_entry_names_and_pasted_commands_work():
    ui = run(["greenray next", "plan", "quit"])
    assert "Practise next" in ui.text and "Week 1" in ui.text


def test_lab_with_an_id_starts_that_lab():
    ui = run(["lab tsh-power-cap-persistence", "quit", "q"])
    assert "find the clock limiter" in ui.text
    ui = run(["lab nope", "q"])
    assert "No scenario 'nope'" in ui.text
