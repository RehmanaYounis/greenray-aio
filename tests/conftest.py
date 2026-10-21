import pytest

from greenray.app.ui import UI
from greenray.content import solutions

requires_solutions = pytest.mark.skipif(not solutions.available(), reason="no solutions available")


@pytest.fixture(autouse=True)
def isolated_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("GREENRAY_HOME", str(tmp_path))


class ScriptedUI(UI):
    """The real UI with colour off, fed by canned input lines. A line that is an exception is raised."""

    def __init__(self, lines):
        super().__init__(color=False)
        self.lines = iter(lines)
        self.output = []
        self.prompts = []

    def out(self, s=""):
        self.output.append(s)

    def ask(self, prompt):
        self.prompts.append(prompt)
        line = next(self.lines, "quit")
        if isinstance(line, BaseException):
            raise line
        return line

    @property
    def text(self):
        return "\n".join(self.output)
