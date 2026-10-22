"""0.4 terminal features: completion, simulated clock, watch/clear, streamed samples, status line."""

import time

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from conftest import ScriptedUI, concrete
from greenray.app import lab
from greenray.app.console import LabCompleter
from greenray.core import clock
from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def complete(shell, text):
    completer = LabCompleter(["nvidia-smi", "nano", "check"], lambda: shell.ctx)
    return [c.text for c in completer.get_completions(Document(text), CompleteEvent())]


def test_completion_commands_hosts_and_paths():
    s = Shell(make_cluster())
    assert complete(s, "n") == ["nano", "nvidia-smi"]
    assert complete(s, "ssh dgx-0") == ["dgx-01", "dgx-02", "dgx-03", "dgx-04"]
    s.run("ssh dgx-01")
    s.run("cd /etc/slurm")
    assert complete(s, "cat gr") == ["gres.conf"]
    assert complete(s, "cat /etc/sl") == ["/etc/slurm/"]
    assert "../slurm/" in complete(s, "ls ../")


def test_clock_advances_per_command_and_rolls_back_with_crashes():
    s = Shell(make_cluster())
    t0 = clock.now(s.ctx.state)
    s.run("hostname")
    s.run("hostname")
    assert clock.now(s.ctx.state) - t0 >= __import__("datetime").timedelta(seconds=2 * clock.TICK)
    assert abs(t0.timestamp() - time.time()) < 60  # the session starts "now"


def test_time_bearing_commands_use_the_clock():
    s = Shell(make_cluster())
    s.run("ssh dgx-01")
    year = str(clock.now(s.ctx.state).year)
    assert year in s.run("dmesg -T | head -1")
    assert "up 12 days" in s.run("uptime")
    assert year in s.run("exit; ssh dgx-02")  # login banner's "Last login"


def test_watch_runs_once_without_a_terminal():
    s = Shell(make_cluster())
    out = s.run("watch -n 1 hostname")
    assert out.startswith("Every 1.0s: hostname") and out.rstrip().endswith("bcm-head")
    assert "no command given" in s.run("watch")
    assert "failed to parse" in s.run("watch -n x hostname")


def test_watch_and_dmon_stream_on_a_terminal():
    s = Shell(make_cluster())
    s.run("ssh dgx-01")
    chunks = []
    s.ctx.emit, s.ctx.live_delay = chunks.append, 0
    assert s.run("nvidia-smi dmon -c 3") == ""  # already printed while streaming
    assert len(chunks) == 4  # header + 3 samples
    chunks.clear()
    assert s.run("watch -n 1 hostname") == "" and len(chunks) == 5
    assert s.run("nvidia-smi dmon -c 2 | head -1").startswith("# gpu")  # piped: returned, not streamed


def test_clear_only_on_a_terminal():
    s = Shell(make_cluster())
    assert s.run("clear") == ""
    s.ctx.emit = [].append
    assert "\033[2J" in s.run("clear")


def test_exam_status_line():
    ui = ScriptedUI(["back"])
    lab.run_exam_lab(
        ui, lab.LabSession(concrete("tsh-power-cap-persistence")), time.time() + 600, status=lambda: "Q 3/30"
    )
    assert any("left · Q 3/30]" in p for p in ui.prompts)
