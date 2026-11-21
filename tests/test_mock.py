from conftest import ScriptedUI, requires_solutions
from greenray.app import mock


def test_pick_labs_one_per_platform_and_never_on_ramp():
    for seed in range(20):
        labs = mock.pick_labs(seed=seed)
        assert len(labs) == len(mock.LAB_AREAS)
        assert len({s.id for s in labs}) == len(labs)
        for area, s in zip(mock.LAB_AREAS, labs):
            assert area in s.tags and "on-ramp" not in s.tags


def test_combined_score():
    assert mock.combined_score(80.0, [1.0, 0.5, 0.0]) == 0.5 * 80 + 0.5 * 50
    assert mock.combined_score(100.0, []) == 50.0


@requires_solutions
def test_mock_runs_end_to_end():
    ui = ScriptedUI(["q"] + ["a"] * 35 + ["y", "f", "y", "n", "n", "n", "n"])
    mock.run_mock(ui, seed=5)
    text = "\n".join(ui.output)
    assert "Result:" in text and "Questions:" in text
    assert "this simulator's assumptions" in text


@requires_solutions
def test_mock_ends_with_a_report_card():
    ui = ScriptedUI(["q"] + ["a"] * 35 + ["y", "f", "y", "n", "n", "n", "n"])
    mock.run_mock(ui, seed=5)
    text = "\n".join(ui.output)
    assert "Practise next:" in text and "greenray lab " in text
    assert "greenray exam -n 20 -m 40 -d" in text


@requires_solutions
def test_labs_only_mock_skips_questions():
    ui = ScriptedUI(["f", "y", "n", "n", "n"])
    mock.run_mock(ui, seed=7, labs_only=True)
    text = "\n".join(ui.output)
    assert "Labs-only mock" in text and "q) Questions" not in text and "Questions:" not in text
    assert "Result:" in text and "weighted" not in text
    from greenray.app import progress

    exam = progress.load()["exams"][-1]
    assert exam["kind"] == "labs" and exam["total"] == 0 and len(exam["labs"]) == 3


def test_exam_menu_can_be_left_without_grading_and_answers_unknown_input():
    ui = ScriptedUI(["q", "a", "q", "bogus", "x", "y"])
    mock.run_mock(ui, seed=1)
    assert "Unknown choice 'bogus'" in ui.text and "Exam abandoned." in ui.text
    assert "Result:" not in ui.text and mock.progress.load()["exams"] == []
    assert "q to return to the section menu" in ui.text


def test_ctrl_c_in_a_question_returns_to_the_menu_and_at_the_menu_offers_to_leave():
    ui = ScriptedUI(["q", KeyboardInterrupt(), KeyboardInterrupt(), "y"])
    mock.run_mock(ui, seed=1)
    assert "Question section" in ui.text and "Exam abandoned." in ui.text and mock.progress.load()["exams"] == []


def test_mock_draws_35_distinct_questions():
    from greenray.content import questions as qb

    bank = qb.load_bank()
    assert mock.N_QUESTIONS == 35
    for seed in range(5):
        qs = qb.pick(bank, mock.N_QUESTIONS, seed=seed)
        assert len(qs) == 35 and len({q["id"] for q in qs}) == 35


def test_questions_come_first_and_cannot_be_reopened_after_submission():
    # answer two, decline the submit once, submit with 33 unanswered, then try to get back to the questions
    ui = ScriptedUI(["q", "a", "a", "q", "s", "n", "s", "y", "q", "x", "y"])
    mock.run_mock(ui, seed=3)
    text = ui.text
    assert text.index("Section 1 of 2") < text.index("Section 2 of 2")
    assert any("33 unanswered count as wrong" in p for p in ui.prompts) and "Question section submitted." in text
    assert "q) Questions" not in text.split("Section 2 of 2")[1]
    assert "question section is closed" in text
    assert "Exam abandoned." in text and mock.progress.load()["exams"] == []


@requires_solutions
def test_answering_every_question_leads_to_the_submit_step_and_unanswered_count_as_wrong():
    ui = ScriptedUI(["q", "a", "q", "s", "y", "f", "y", "n", "n", "n", "n"])
    mock.run_mock(ui, seed=2)
    assert "Questions: " in ui.text
    exam = mock.progress.load()["exams"][-1]
    assert exam["kind"] == "mock" and exam["total"] == 35 and exam["correct"] <= 1
    ui = ScriptedUI(["q"] + ["a"] * 35 + ["n", "s", "y", "f", "y", "n", "n", "n", "n"])
    mock.run_mock(ui, seed=2)
    assert "All 35 questions answered." in ui.text and ui.text.count("Question section submitted.") == 1


def test_untouched_labs_score_nothing_and_say_so():
    ui = ScriptedUI(["q"] + ["a"] * 35 + ["y", "f", "y", "n", "n", "n", "n"])
    mock.run_mock(ui, seed=1)
    assert ui.text.count("not attempted") == 3 and "% of objectives" not in ui.text
    assert all(v == 0 for v in mock.progress.load()["exams"][-1]["labs"].values())
