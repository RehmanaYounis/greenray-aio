"""P0 regressions: crashing commands, rollback, realistic input errors, Ctrl-C, validation, evidence."""

import os
import pickle

import pytest

from aiolab import commands
from aiolab.app import lab, progress
from aiolab.content import scenarios as sc
from aiolab.content.validate import check_scenario
from aiolab.core.shell import INCONSISTENT_FLAG, Shell
from aiolab.core.state import make_cluster
from conftest import ScriptedUI


@pytest.fixture
def boom_command(monkeypatch):
    """A command that changes state, moves the shell, then crashes, like a buggy handler would."""

    def boom(ctx, args):
        ctx.state.nodes["dgx-01"].gpus[0].power_limit_w = 1
        ctx.node.files["/tmp/half-written"] = "x"
        ctx.host = "dgx-04"
        raise RuntimeError("handler bug")

    commands._load_all()
    monkeypatch.setitem(commands.REGISTRY, "boom", boom)


def test_internal_error_restores_state_and_keeps_session(boom_command):
    sh = Shell(make_cluster())
    sh.run("ssh dgx-01")
    before = pickle.dumps(sh.ctx.state)
    out = sh.run("boom && echo should-not-run")
    assert "internal simulator error while running 'boom'" in out
    assert "restored to its state before this command" in out
    assert "should-not-run" not in out
    assert pickle.dumps(sh.ctx.state) == before
    assert sh.ctx.host == "dgx-01"
    assert not any(argv[0] == "boom" for _, argv in sh.ctx.executed)  # a crash is never evidence
    assert sh.run("hostname") == "dgx-01"  # the shell keeps working


def test_unrestorable_crash_blocks_grading(boom_command, monkeypatch):
    s = sc.find("tsh-power-cap-persistence")
    sh = Shell(s.build_state())
    for line in ["ssh dgx-03", "nvidia-smi -q -d PERFORMANCE", "nvidia-smi -pl 700", "nvidia-smi -pm 1"]:
        sh.run(line)
    assert all(ok for _, ok, _ in s.grade(sh))
    monkeypatch.setattr(Shell, "_checkpoint", lambda self: (None, self.ctx.host, [], self.ctx.cwd))
    out = sh.run("boom")
    assert "grading is blocked" in out
    assert sh.ctx.state.flags[INCONSISTENT_FLAG]
    results = s.grade(sh)
    assert not all(ok for _, ok, _ in results)
    assert results[0][0].startswith("cluster state is consistent")


FUZZ = [
    "",
    " -h",
    " --help",
    " --bogus",
    " -i",
    " -pl abc",
    " -i 99",
    " get",
    " show",
    " update",
    " x y z",
    " -n",
    " -c",
    " -- -",
    " ''",
]


@pytest.mark.parametrize("host", ["bcm-head", "dgx-01"])
def test_malformed_input_never_hits_internal_error(host):
    bad = []
    for name in commands.names():
        if name == "reboot":
            continue
        for variant in FUZZ:
            out = Shell(make_cluster(), host).run(name + variant, heredoc_reader=lambda p: None)
            if "internal simulator error" in out:
                bad.append(name + variant)
    assert bad == []


@pytest.mark.parametrize(
    "line,expected",
    [
        ("nvidia-smi -pl abc", "Invalid combination of input arguments"),
        ("scontrol --help", "scontrol [<OPTION>] [<COMMAND>]"),
        ("scontrol --bogus", "unrecognized option '--bogus'"),
        ("cp", "cp: missing file operand"),
        ("cp /etc/hostname", "missing destination file operand"),
        ("mpirun -n", 'option "-n" did not have enough parameters'),
        ("perfquery nope", "can't resolve destination port nope"),
        ("frobnicate", "frobnicate: command not found"),
    ],
)
def test_user_input_errors_are_realistic(line, expected):
    sh = Shell(make_cluster())
    sh.run("ssh dgx-01")
    assert expected in sh.run(line)


def test_power_limit_reports_previous_value():
    sh = Shell(make_cluster())
    sh.run("ssh dgx-01")
    sh.run("nvidia-smi -i 0 -pl 500")
    assert "set to 600.00 W from 500.00 W" in sh.run("nvidia-smi -i 0 -pl 600")


def test_ctrl_c_cancels_the_line_not_the_lab():
    ui = ScriptedUI([KeyboardInterrupt(), "hostname", "quit"])
    lab.run_lab(ui, sc.find("tsh-power-cap-persistence"))
    assert "^C" in ui.output and "bcm-head" in ui.output
    assert progress.load()["scenarios"]["tsh-power-cap-persistence"]["attempts"] == 1


def test_validate_reports_broken_setup_instead_of_crashing():
    broken = sc.Scenario(
        id="tsh-broken",
        title="t",
        domain="Administration",
        briefing="b",
        checks=[{"desc": "d", "expr": "True"}],
        setup=[{"set": "nodes.no-such-node.up", "value": 1}],
    )
    assert check_scenario(broken).startswith("raised KeyError")


def test_check_errors_hidden_from_learner_unless_debug(monkeypatch):
    results = [("maria exists", False, "KeyError: 'maria'")]
    ui = ScriptedUI([])
    monkeypatch.delenv("AIOLAB_DEBUG", raising=False)
    lab.show_results(ui, results)
    assert "KeyError" not in ui.text
    monkeypatch.setenv("AIOLAB_DEBUG", "1")
    lab.show_results(ui, results)
    assert "KeyError" in ui.text
    assert os.environ["AIOLAB_DEBUG"] == "1"
