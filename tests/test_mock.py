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
    ui = ScriptedUI(["q"] + ["a"] * 30 + ["f", "y", "n", "n", "n", "n"])
    mock.run_mock(ui, seed=5)
    text = "\n".join(ui.output)
    assert "Result:" in text and "Questions:" in text
    assert "assumptions, not NVIDIA's rules" in text


@requires_solutions
def test_mock_ends_with_a_report_card():
    ui = ScriptedUI(["q"] + ["a"] * 30 + ["f", "y", "n", "n", "n", "n"])
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
