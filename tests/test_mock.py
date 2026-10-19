from aiolab.app import mock
from conftest import requires_solutions


def test_pick_labs_one_per_platform():
    labs = mock.pick_labs(seed=3)
    assert len(labs) == len(mock.LAB_AREAS)
    assert len({s.id for s in labs}) == len(labs)
    for area, s in zip(mock.LAB_AREAS, labs):
        assert area in s.tags


def test_combined_score():
    assert mock.combined_score(80.0, [1.0, 0.5, 0.0]) == 0.5 * 80 + 0.5 * 50
    assert mock.combined_score(100.0, []) == 50.0


class ScriptedUI:
    """Feeds canned input lines; records output."""

    def __init__(self, lines):
        self.lines = iter(lines)
        self.output = []

    def out(self, s=""):
        self.output.append(s)

    def ask(self, prompt):
        return next(self.lines, "quit")

    def __getattr__(self, name):  # colour helpers
        return lambda s: s


@requires_solutions
def test_mock_runs_end_to_end():
    ui = ScriptedUI(["q"] + ["a"] * 30 + ["f", "y", "n", "n", "n", "n"])
    mock.run_mock(ui, seed=5)
    text = "\n".join(ui.output)
    assert "Result:" in text and "Questions:" in text
