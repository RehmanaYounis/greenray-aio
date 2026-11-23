"""The mock exam's rules: one deadline that is enforced, no credit for navigation, counted answers, review, banners."""

from __future__ import annotations

import time

import pytest

from conftest import ScriptedUI, requires_solutions
from greenray.app import exam, mock, progress
from greenray.app.console import LineReader, TimeUp
from greenray.app.lab import LabSession
from greenray.content import questions as qb

pytestmark = requires_solutions


class Clock:
    """Stands in for time.time so a test can let exam minutes pass in an instant."""

    def __init__(self) -> None:
        self.real = time.time
        self.offset = 0.0

    def __call__(self) -> float:
        return self.real() + self.offset


@pytest.fixture
def clock(monkeypatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(time, "time", fake)
    return fake


class TimedUI(ScriptedUI):
    """ScriptedUI where a line ``("after", minutes, text)`` arrives that many minutes later, the prompt idle meanwhile."""

    def __init__(self, lines, clock: Clock):
        super().__init__(lines)
        self.clock = clock

    def ask(self, prompt, typed=False):
        self.prompts.append(prompt)
        line = next(self.lines, "quit")
        if isinstance(line, tuple):
            self.clock.offset += line[1] * 60
            line = line[2]
        if isinstance(line, BaseException):
            raise line
        return line


def drawn(seed: int) -> list[dict]:
    return qb.pick(qb.load_bank(), mock.N_QUESTIONS, seed=seed)


def right(q: dict) -> str:
    return " ".join(qb.LETTERS[a] for a in sorted(q["answer"]))


def wrong(q: dict) -> str:
    """An answer with the right number of letters that is not the correct one."""
    others = [i for i in range(len(q["choices"])) if i not in q["answer"]]
    picked = (others + sorted(q["answer"]))[: len(q["answer"])]
    return " ".join(qb.LETTERS[a] for a in sorted(picked))


@pytest.fixture
def sessions(monkeypatch) -> list[LabSession]:
    """The lab sessions a mock creates, so a test can look at their cluster state afterwards."""
    made: list[LabSession] = []

    class Recording(LabSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(mock, "LabSession", Recording)
    return made


def last_exam() -> dict:
    return progress.load()["exams"][-1]


# ---- 1. the deadline


def test_an_answer_typed_after_the_deadline_does_not_count(clock):
    qs = drawn(5)
    ui = TimedUI(["q", ("after", mock.MINUTES + 1, right(qs[0])), "n", "n", "n", "n", "n"], clock)
    mock.run_mock(ui, seed=5)
    assert "Time is up. That answer was not recorded." in ui.text and "Time is up. Grading." in ui.text
    assert "Questions: 0/35" in ui.text and last_exam()["correct"] == 0


def test_an_answer_typed_just_before_the_deadline_counts(clock):
    qs = drawn(5)
    ui = TimedUI(
        ["q", ("after", mock.MINUTES - 1, right(qs[0])), ("after", 2, right(qs[1])), "n", "n", "n", "n", "n"], clock
    )
    mock.run_mock(ui, seed=5)
    assert "Questions: 1/35" in ui.text and last_exam()["correct"] == 1


def test_a_lab_command_typed_after_the_deadline_is_not_run(clock, sessions):
    ui = TimedUI(
        ["1", "echo BEFORE > /root/before.txt", ("after", 41, "echo AFTER_DEADLINE > /root/late.txt"), "n", "n", "n"],
        clock,
    )
    mock.run_mock(ui, seed=7, labs_only=True)
    files = sessions[0].shell.ctx.state.nodes["bcm-head"].files
    assert "/root/before.txt" in files and "/root/late.txt" not in files
    assert ["echo", "AFTER_DEADLINE"] not in [argv for _host, argv in sessions[0].shell.ctx.executed]
    assert "Time is up. Grading." in ui.text and "Result:" in ui.text


def test_expiry_at_a_menu_or_a_confirmation_grades_instead_of_acting(clock):
    ui = TimedUI(
        [("after", 41, "x"), "n", "n", "n"], clock
    )  # "leave" typed too late: the exam is graded, not abandoned
    mock.run_mock(ui, seed=7, labs_only=True)
    assert "Exam abandoned." not in ui.text and "Time is up. Grading." in ui.text and last_exam()["kind"] == "labs"

    ui = TimedUI(["f", ("after", 41, "n"), "n", "n", "n"], clock)
    clock.offset = 0
    mock.run_mock(ui, seed=7, labs_only=True)
    assert "Time is up. Grading." in ui.text


def test_questions_and_labs_share_one_deadline(clock):
    qs = drawn(5)
    # 100 minutes go by in the questions: the labs get what is left of the same 120, not a clock of their own
    ui = TimedUI(
        ["q", ("after", 100, right(qs[0])), "q", "s", "y", "1", "back", ("after", 21, "1"), "n", "n", "n", "n"], clock
    )
    mock.run_mock(ui, seed=5)
    assert any("[19m left" in p for p in ui.prompts) and "Lab section (19 min left)" in ui.text
    assert "Time is up. Grading." in ui.text and ui.text.count("=== Section") == 2
    assert "(121 of 120 min used)" in ui.text


def test_time_running_out_in_the_questions_skips_the_labs(clock):
    ui = TimedUI(["q", ("after", mock.MINUTES + 1, "a"), "n", "n", "n", "n", "n"], clock)
    mock.run_mock(ui, seed=5)
    assert "Section 2 of 2" not in ui.text and ui.text.count("not attempted") == 3


def test_the_drill_refuses_a_late_answer_too(clock):
    ui = TimedUI([("after", 3, "a")], clock)
    exam.run_exam(ui, n=3, minutes=2)
    assert "That answer was not recorded." in ui.text and "nothing is recorded" in ui.text


def test_the_terminal_prompt_closes_itself_at_the_deadline():
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        reader = LineReader(None, input=pipe, output=DummyOutput())
        pipe.send_text("half typed and never ent")  # no Enter: the learner is idle at the prompt
        started = time.time()
        with pytest.raises(TimeUp):
            reader.read("> ", timeout=0.3)
        assert 0.2 < time.time() - started < 5

        pipe.send_text("in time\n")  # the same reader still works for a prompt that is answered
        assert reader.read("> ", timeout=5) == "in time"


def test_ask_until_refuses_a_line_that_arrives_late(clock):
    ui = TimedUI([("after", 2, "late"), "on time"], clock)
    with pytest.raises(TimeUp):
        ui.ask_until("> ", time.time() + 60)
    assert ui.ask_until("> ", time.time() + 60) == "on time"
    with pytest.raises(TimeUp):
        ui.ask_until("> ", time.time() - 1)


# ---- 2. no credit for opening a lab


def _lab_with_a_criterion_met_at_the_start() -> int:
    for seed in range(40):
        for scenario in mock.pick_labs(seed)[:1]:
            if LabSession(scenario).initially_true:
                return seed
    raise AssertionError("no seed puts such a lab first")


def test_opening_a_lab_and_leaving_scores_nothing():
    seed = _lab_with_a_criterion_met_at_the_start()
    ui = ScriptedUI(["1", "check", "task", "ls", "back", "f", "y", "n", "n", "n"])
    mock.run_mock(ui, seed=seed, labs_only=True)
    assert "Not available during the exam" in ui.text
    assert "Result: 0.0%" in ui.text and "opened, nothing achieved (0%)" in ui.text
    assert all(v == 0 for v in last_exam()["labs"].values()) and last_exam()["score"] == 0


def test_safety_criteria_count_once_the_learner_achieves_something(sessions):
    seed = _lab_with_a_criterion_met_at_the_start()
    scenario = mock.pick_labs(seed)[0]
    ui = ScriptedUI(_script("1", _solve(sessions, 1), "back", "f", "y", "n", "n", "n"))
    mock.run_mock(ui, seed=seed, labs_only=True)
    lab = sessions[0]
    assert lab.initially_true and lab.exam_score() == lab.score() == 1.0
    assert last_exam()["labs"][scenario.id] == 1.0 and f"Lab {scenario.id}: 100% of objectives" in ui.text


def test_partial_work_earns_its_share_including_the_safety_criteria():
    seed = _lab_with_a_criterion_met_at_the_start()
    lab = LabSession(mock.pick_labs(seed)[0])
    assert lab.exam_score() == 0.0 and not lab.achieved_something()
    for step in lab.scenario_required.solution:
        lines = step.split("\n")
        feed = iter(lines[1:])
        lab.shell.run(lines[0], heredoc_reader=lambda _p, feed=feed: next(feed, None))
        if lab.achieved_something():
            break
    assert 0 < lab.exam_score() == lab.score()


# ---- 3. the number of answers


def _first(qs: list[dict], count: int) -> int:
    return next(i for i, q in enumerate(qs) if len(q["answer"]) == count)


@pytest.mark.parametrize(
    ("count", "too_many", "message"), [(1, "A B C D", "exactly one answer"), (2, "A B C", "exactly 2 answers")]
)
def test_an_answer_with_the_wrong_number_of_letters_keeps_the_learner_on_the_question(count, too_many, message):
    qs = drawn(5)
    i = _first(qs, count)
    too_few = "A" if count == 2 else None
    tries = [too_many] + ([too_few] if too_few else [])
    ui = ScriptedUI(["q", *["s"] * i, *tries, right(qs[i]), "q", "s", "y", "f", "y", "n", "n", "n", "n"])
    mock.run_mock(ui, seed=5)
    assert ui.text.count(f"This question takes {message}") == len(tries) and "Nothing was recorded." in ui.text
    assert ui.text.count(f"\nQ{i + 1}/35") == 1  # asked once: the refused tries did not move on
    assert last_exam()["correct"] == 1


def test_every_question_states_how_many_answers_it_takes():
    for q in qb.load_bank():
        assert len(q["answer"]) == 1 or f"choose {['', '', 'two', 'three'][len(q['answer'])]}" in " ".join(
            q["question"].lower().split()
        ), q["id"]


# ---- 4. review before submitting


def test_flagged_questions_can_be_reviewed_and_changed_before_submitting():
    qs = drawn(5)
    script = [
        "q",
        "f",
        wrong(qs[0]),
        *[wrong(q) for q in qs[1:]],
        "n",
    ]  # flag Q1, answer all wrongly, decline the submit
    script += ["q", "f", right(qs[0]), "3", right(qs[2]), "2", "", "q"]  # re-enter: flagged Q1 fixed, Q3 fixed, Q2 kept
    script += ["s", "y", "f", "y", "n", "n", "n", "n"]
    ui = ScriptedUI(script)
    mock.run_mock(ui, seed=5)
    assert "--- Review before submitting ---" in ui.text and "Flagged: Q1" in ui.text
    assert (
        ui.text.count("Submit the question section") == 0
        and sum("Submit the question section" in p for p in ui.prompts) == 2
    )
    assert "Questions: 2/35" in ui.text and last_exam()["correct"] == 2


def test_review_is_offered_in_the_menu_and_keeps_answer_counts():
    qs = drawn(5)
    i = _first(qs, 1)
    ui = ScriptedUI(
        [
            "q",
            *["s"] * i,
            wrong(qs[i]),
            "q",
            "r",
            str(i + 1),
            "A B",
            right(qs[i]),
            "q",
            "s",
            "y",
            "f",
            "y",
            "n",
            "n",
            "n",
            "n",
        ]
    )
    mock.run_mock(ui, seed=5)
    assert "r) Review or change answers before submitting" in ui.text
    assert "This question takes exactly one answer" in ui.text and last_exam()["correct"] == 1


def test_a_flag_can_be_removed_and_the_hint_says_what_flags_are_for():
    qs = drawn(5)
    ui = ScriptedUI(["q", "f", right(qs[0]), "q", "r", "1", "u", "", "q", "x", "y"])
    mock.run_mock(ui, seed=5)
    assert "f flags it for review before you submit (r in the section menu) and after the exam" in ui.text
    assert "Flagged: Q1" in ui.text and "flag removed" in ui.text and ui.text.count("Flagged: none") == 1


def test_questions_stay_locked_after_a_confirmed_submission():
    qs = drawn(5)
    ui = ScriptedUI(["q", "f", wrong(qs[0]), "q", "s", "y", "r", "q", "1", "back", "f", "y", "y", "n", "n", "n"])
    mock.run_mock(ui, seed=5)
    after = ui.text.split("Section 2 of 2")[1]
    assert after.count("question section is closed") == 2 and "Review before submitting" not in after
    assert last_exam()["correct"] == 0
    assert "Your answer:" in after  # the flagged question still shows in the review after the exam


# ---- 5. the banner


def test_labs_only_banner_describes_a_labs_only_result():
    ui = ScriptedUI(["x", "y"])
    mock.run_mock(ui, seed=7, labs_only=True)
    banner = ui.text.split("=== Section")[0]
    assert "50/50" not in banner and "weighting" not in banner
    assert "40 minutes" in banner and "70% target" in banner and "average of the labs" in banner


def test_full_mock_banner_keeps_its_weighting_note():
    ui = ScriptedUI(["x", "y"])
    mock.run_mock(ui, seed=7)
    banner = ui.text.split("=== Section")[0]
    assert "50/50 question/lab weighting" in banner and "70% target" in banner and "120 minutes" in banner


# ---- whole runs


def _solve(sessions: list[LabSession], number: int):
    """The reference solution of lab ``number`` as it was drawn for this run (the mock picks the variant)."""
    for step in sessions[number - 1].scenario_required.solution:
        yield from step.split("\n")


def _script(*parts):
    """Chain plain lines and generators into one input script; the generators are read when their turn comes."""
    for part in parts:
        if isinstance(part, str):
            yield part
        else:
            yield from part


def test_a_full_mock_with_all_three_labs_scores_reviews_and_records(sessions):
    qs = drawn(11)
    labs = mock.pick_labs(11)
    answers = [right(q) if i % 5 else wrong(q) for i, q in enumerate(qs)]  # 28 right, 7 wrong
    ui = ScriptedUI(
        _script(
            *["q", *answers, "y"],
            *[
                "1",
                _solve(sessions, 1),
                "back",
                "2",
                _solve(sessions, 2),
                "back",
                "3",
                "back",
                "f",
                "y",
            ],  # lab 3 opened only
            *["y", "n", "n", "n"],  # read the review, decline the one solution offered
        )
    )
    before = progress.load()
    mock.run_mock(ui, seed=11)
    record = last_exam()
    assert record["kind"] == "mock" and record["total"] == 35 and record["correct"] == 28
    assert record["labs"] == {labs[0].id: 1.0, labs[1].id: 1.0, labs[2].id: 0.0}
    expected = 0.5 * (100 * 28 / 35) + 0.5 * (100 * 2 / 3)
    assert abs(record["score"] - expected) < 0.06 and f"Result: {expected:.1f}%" in ui.text
    assert "Your answer:" in ui.text and ui.text.count("Correct:") == 7  # the seven missed questions are reviewed
    # each lab ran in its own cluster, and practice records are untouched by an exam
    assert len({id(s.shell.ctx.state) for s in sessions}) == 3
    assert not sessions[2].shell.ctx.executed and sessions[0].shell.ctx.executed != sessions[1].shell.ctx.executed
    assert progress.load()["scenarios"] == before["scenarios"]


def test_a_labs_only_mock_scores_the_average_of_the_labs(sessions):
    labs = mock.pick_labs(11)
    ui = ScriptedUI(_script("2", _solve(sessions, 2), "back", "f", "y", "n", "n", "n"))
    mock.run_mock(ui, seed=11, labs_only=True)
    record = last_exam()
    assert record["kind"] == "labs" and record["total"] == 0
    assert record["labs"] == {labs[0].id: 0.0, labs[1].id: 1.0, labs[2].id: 0.0}
    assert abs(record["score"] - 100 / 3) < 0.06 and "Result: 33.3%" in ui.text and ui.text.count("not attempted") == 2


@pytest.mark.parametrize(
    ("script", "labs_only"),
    [
        (["q", "a", "q", "x", "y"], False),  # from the question section
        (["q", "s", "s", "q", "s", "y", "1", "ls", "back", "x", "y"], False),  # from the lab section
        (["1", "ls", "back", "x", "y"], True),
    ],
    ids=["questions", "labs", "labs-only"],
)
def test_an_abandoned_mock_records_nothing(script, labs_only):
    before = progress.load()
    ui = ScriptedUI(script)
    mock.run_mock(ui, seed=5, labs_only=labs_only)
    assert "Exam abandoned." in ui.text and "Result:" not in ui.text
    assert progress.load() == before


def test_exam_input_stays_out_of_the_persistent_history(tmp_path, monkeypatch):
    from greenray.app import ui as ui_module

    made: list = []

    class Reader:  # stands in for the terminal prompt, which needs a console that CI on Windows does not have
        def __init__(self, history_file, **_options):
            self.history_file = history_file
            made.append(self)

    monkeypatch.setattr(ui_module, "LineReader", Reader)
    history = tmp_path / "history"
    ui = ui_module.UI(color=False, history_file=history)
    ui.interactive = True  # as on a terminal: only there is any history kept
    lab_reader = ui._line_reader()
    assert lab_reader.history_file == history
    with ui.private_history():
        private = ui._line_reader()
        assert private is not lab_reader and private.history_file is None  # answers and commands stay in memory
    assert ui._line_reader() is lab_reader and [r.history_file for r in made] == [history, None]
    assert not history.exists()
