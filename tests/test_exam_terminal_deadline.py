"""The mock exam's deadline on a real terminal: an open prompt closes by itself, on a basic terminal too."""

from __future__ import annotations

import contextlib
import json
import os
import re
import select
import sys
import time
from pathlib import Path

import pytest

from conftest import requires_solutions
from greenray.app import mock
from greenray.content import questions as qb

pytestmark = [requires_solutions, pytest.mark.skipif(sys.platform == "win32", reason="needs a pseudo-terminal")]

SEED = 5
SECONDS = 3.0
TERMS = ["dumb", "xterm-256color"]

# the mock with a clock of a few seconds; afterwards it reports every command the labs ran
CHILD = """
import sys
from greenray.app import mock
from greenray.app.lab import LabSession
from greenray.app.ui import UI

mock.MINUTES = mock.LABS_ONLY_MINUTES = float(sys.argv[2]) / 60
sessions = []
made = LabSession.__post_init__

def remember(self):
    made(self)
    sessions.append(self)

LabSession.__post_init__ = remember
mock.run_mock(UI(color=False), seed=int(sys.argv[3]), labs_only=sys.argv[1] == "labs")
print("RAN", [" ".join(argv) for s in sessions for _, argv in s.shell.ctx.executed])
print("EXAM OVER")
"""


class Sitting:
    """What one mock exam in a pseudo-terminal showed and saved."""

    def __init__(self, text: str, graded_after: float | None, home: Path):
        self.text = text
        self.graded_after = graded_after  # seconds from the first banner to "Time is up"
        saved = home / "progress.json"
        self.exams: list = json.loads(saved.read_text())["exams"] if saved.exists() else []

    @property
    def ran(self) -> str:
        found = re.search(r"^RAN (.*)$", self.text, re.M)
        return found.group(1).strip() if found else ""


def sit(term: str, mode: str, steps: list[tuple[str, str]], home: Path) -> Sitting:
    """Run the mock under ``TERM=term``. Each step waits for its text to appear, then types its keys.

    Enter is only ever sent where a step spells it out, and "n" to the yes/no prompts after
    grading: if the exam gets that far, the clock closed the open prompt without help.
    """
    import pty

    env = dict(os.environ, TERM=term, GREENRAY_HOME=str(home), COLUMNS="200", LINES="50")
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe(sys.executable, [sys.executable, "-c", CHILD, mode, str(SECONDS), str(SEED)], env)
    todo, raw, seen, answered = list(steps), b"", 0, 0
    started = graded = None
    give_up = time.time() + SECONDS + 15
    try:
        while time.time() < give_up:
            if not select.select([fd], [], [], 0.05)[0]:
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError:  # the child closed the terminal
                break
            if not chunk:
                break
            raw += chunk
            text = re.sub(rb"\x1b\[[0-9;?]*[a-zA-Z]", b"", raw).decode("utf-8", "replace")
            if started is None and "Section 1 of" in text:
                started = time.time()
            if graded is None and "Time is up" in text:
                graded = time.time()
            while todo and todo[0][0] in text[seen:]:
                seen = text.index(todo[0][0], seen) + len(todo[0][0])
                os.write(fd, todo.pop(0)[1].encode())
            while text.count("[Y/n]") > answered:
                answered += 1
                os.write(fd, b"n\n")
            if "EXAM OVER" in text:
                break
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, 9)
        os.waitpid(pid, 0)
        os.close(fd)
    text = re.sub(rb"\x1b\[[0-9;?]*[a-zA-Z]", b"", raw).decode("utf-8", "replace").replace("\r", "")
    return Sitting(text, graded - started if started and graded else None, home)


def right_answer(number: int) -> str:
    question = qb.pick(qb.load_bank(), mock.N_QUESTIONS, seed=SEED)[number - 1]
    return " ".join(qb.LETTERS[i] for i in sorted(question["answer"]))


def graded_once_on_time(sitting: Sitting) -> None:
    assert sitting.graded_after is not None, f"the prompt was still open after the deadline:\n{sitting.text[-600:]}"
    assert sitting.graded_after < SECONDS + 3
    assert "EXAM OVER" in sitting.text
    assert sitting.text.count("Result:") == 1
    assert len(sitting.exams) == 1


@pytest.mark.parametrize("term", TERMS)
def test_an_idle_question_prompt_closes_at_the_deadline(term, tmp_path):
    sitting = sit(term, "questions", [("Leave without grading", "q\n")], tmp_path)
    graded_once_on_time(sitting)
    assert "Questions: 0/35" in sitting.text


@pytest.mark.parametrize("term", TERMS)
def test_an_answer_left_on_the_prompt_line_earns_nothing(term, tmp_path):
    # question 1 is answered in time; the right answer to question 2 is typed and never sent
    steps = [("Leave without grading", "q\n"), ("Q1/35", right_answer(1) + "\n"), ("Q2/35", right_answer(2))]
    sitting = sit(term, "questions", steps, tmp_path)
    graded_once_on_time(sitting)
    assert "Questions: 1/35" in sitting.text


@pytest.mark.parametrize("term", TERMS)
def test_an_idle_lab_prompt_closes_at_the_deadline(term, tmp_path):
    sitting = sit(term, "labs", [("Leave without grading", "1\n")], tmp_path)
    graded_once_on_time(sitting)
    assert sitting.ran == "[]"


@pytest.mark.parametrize("term", TERMS)
def test_a_lab_command_left_on_the_prompt_line_does_not_run(term, tmp_path):
    steps = [
        ("Leave without grading", "1\n"),
        ("Type 'help'", "touch /root/in-time.txt\n"),
        ("in-time.txt", "touch /root/too-late.txt"),
    ]
    sitting = sit(term, "labs", steps, tmp_path)
    graded_once_on_time(sitting)
    assert sitting.ran == "['touch /root/in-time.txt']"
