"""Output evidence is recorded on the host where a command ran, however the learner got there."""

from __future__ import annotations

import pytest

from conftest import concrete
from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def _recorded(shell: Shell, host: str, command: str) -> list[str]:
    return [text for h, argv, text in shell.ctx.outputs if h == host and argv[0] == command]


@pytest.mark.parametrize(
    "line",
    [
        "ssh dgx-03 nvidia-smi -L",
        "ssh dgx-03 'nvidia-smi -L | grep GPU'",
        "ssh root@dgx-03 nvidia-smi -L",
        "pdsh -w dgx-03 nvidia-smi -L",
        'cmsh -c "device; pexec -n dgx-03 nvidia-smi -L"',
    ],
)
def test_a_remote_command_is_recorded_on_the_remote_host(line):
    shell = Shell(make_cluster())
    assert "GPU 0" in shell.run(line)
    assert any("GPU 0" in text for text in _recorded(shell, "dgx-03", "nvidia-smi"))
    assert not _recorded(shell, "bcm-head", "nvidia-smi")


@pytest.mark.parametrize("line", ["bash -c 'sinfo'", "sudo -u root sinfo", "salloc -N 1 sinfo"])
def test_a_nested_command_is_recorded_under_its_own_name(line):
    shell = Shell(make_cluster())
    assert "PARTITION" in shell.run(line)
    assert any("PARTITION" in text for text in _recorded(shell, "bcm-head", "sinfo"))


def test_a_remote_command_that_fails_records_nothing():
    shell = Shell(make_cluster())
    assert "No devices were found" in shell.run("ssh dgx-03 nvidia-smi -i 99")
    assert not _recorded(shell, "dgx-03", "nvidia-smi")


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_tsh2_accepts_nvidia_smi_run_through_ssh_from_the_head_node(variant):
    s = concrete("tsh-which-gpu-is-which", variant)
    shell = Shell(s.build_state())
    desc, ok, err = s.grade(shell)[0]
    assert "nvidia-smi" in desc and not ok and not err
    shell.run(f"ssh {s.params['node']} nvidia-smi")
    assert s.grade(shell)[0][1]


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_tsh2_still_rejects_another_node_and_failed_queries_through_ssh(variant):
    s = concrete("tsh-which-gpu-is-which", variant)
    shell = Shell(s.build_state())
    other = next(n for n in ("dgx-01", "dgx-02") if n != s.params["node"])
    shell.run(f"ssh {other} nvidia-smi")
    shell.run(f"ssh {s.params['node']} nvidia-smi -i 99")
    assert not s.grade(shell)[0][1]
