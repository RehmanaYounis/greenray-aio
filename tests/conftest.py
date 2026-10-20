import pytest

from aiolab.content import solutions

requires_solutions = pytest.mark.skipif(not solutions.available(), reason="private solutions repository not installed")


@pytest.fixture(autouse=True)
def isolated_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("AIOLAB_HOME", str(tmp_path))


class ScriptedUI:
    """Feeds canned input lines and records output. A line that is an exception instance is raised."""

    def __init__(self, lines):
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

    def __getattr__(self, name):  # colour helpers
        return lambda s: s
