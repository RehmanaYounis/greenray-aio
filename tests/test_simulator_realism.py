"""The simulated tools refuse what the real ones refuse: unknown options, unknown subcommands, failed remotes."""

from __future__ import annotations

import pytest

from conftest import concrete, requires_solutions
from greenray.content import helppages
from greenray.content import scenarios as sc
from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def run(line: str, shell: Shell | None = None) -> tuple[str, int]:
    shell = shell or Shell(make_cluster())
    out = shell.run(line)
    return out, shell.ctx.last_code


# ---- ssh <host> <command> ends with the status of the remote command


@pytest.mark.parametrize(
    ("line", "code"),
    [
        ("ssh dgx-01 true", 0),
        ("ssh dgx-01 hostname", 0),
        ("ssh dgx-01 false", 1),
        ("ssh dgx-01 'ls /nosuch'", 2),
        ("ssh dgx-01 nvidia-smi -i 99", 6),
        ("ssh dgx-01 nosuchcommand", 127),
        ("ssh nosuchhost true", 255),
    ],
)
def test_ssh_returns_the_remote_exit_status(line, code):
    assert run(line)[1] == code


def test_ssh_status_drives_shell_logic_and_keeps_the_message():
    assert run("ssh dgx-01 true && echo reached")[0].strip() == "reached"
    assert "reached" not in run("ssh dgx-01 false && echo reached")[0]
    assert run("ssh dgx-01 false || echo fallback")[0].strip() == "fallback"
    assert "No devices were found matching the id 99" in run("ssh dgx-01 nvidia-smi -i 99")[0]


def test_a_failed_remote_command_is_not_recorded_as_completed_anywhere():
    shell = Shell(make_cluster())
    shell.run("ssh dgx-01 nvidia-smi -i 99")
    assert [completed for _pos, _host, _argv, _text, completed in shell.ctx.results] == [False, False]
