"""Streamed output reaches the grader, and test workloads are graded on what the briefing specifies."""

from __future__ import annotations

import re

import pytest

from conftest import concrete, requires_solutions
from greenray.content import scenarios as sc
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

pytestmark = requires_solutions


def terminal(state) -> tuple[Shell, list[str]]:
    """A shell that streams to a screen, as the interactive lab does."""
    shell = Shell(state)
    screen: list[str] = []
    shell.ctx.emit = screen.append
    return shell, screen


def solved(code: str, edit=None, skip: str | None = None, shell: Shell | None = None):
    s = concrete(code)
    shell = shell or Shell(s.build_state())
    for step in s.solution:
        if skip and re.search(skip, step):
            continue
        lines = (edit(step) if edit else step).split("\n")
        feed = iter(lines[1:])
        shell.run(lines[0], heredoc_reader=lambda _p, feed=feed: next(feed, None))
    return s, shell


def marks(s: sc.Scenario, shell: Shell) -> dict[str, bool]:
    out = {}
    for desc, ok, err in s.grade(shell):
        assert not err, (desc, err)
        out[desc] = ok
    return out


def one(results: dict[str, bool], wording: str) -> bool:
    found = [ok for desc, ok in results.items() if wording in desc]
    assert len(found) == 1, (wording, list(results))
    return found[0]


# ---- a tool that streams its samples to the terminal still shows the grader what it printed


@pytest.mark.parametrize("command", ["nvidia-smi dmon -c 3", "dcgmi dmon -e 203 -c 3"])
def test_streamed_samples_are_recorded_as_the_commands_output(command):
    shell, screen = terminal(make_cluster())
    shell.run("ssh dgx-02")
    assert shell.run(command) == ""  # everything went to the screen as it was produced
    assert len(screen) >= 3
    recorded = [text for _pos, host, argv, text, done in shell.ctx.results if argv[0] == command.split()[0] and done]
    assert len(recorded) == 1
    assert all(chunk in recorded[0] for chunk in screen)
    assert shell.ctx.emit == screen.append  # the screen is handed back untouched


def test_a_piped_sampling_tool_is_recorded_as_before():
    shell, screen = terminal(make_cluster())
    out = shell.run("ssh dgx-02 nvidia-smi dmon -c 2 | cat")
    assert "# gpu" in out and screen == []


def test_a_streaming_command_that_fails_records_only_its_error():
    shell, _screen = terminal(make_cluster())
    shell.run("ssh dgx-02")
    shell.run("dcgmi dmon")
    assert [done for _pos, _host, argv, _text, done in shell.ctx.results if argv[0] == "dcgmi"] == [False]


@pytest.mark.parametrize("dmon", ["nvidia-smi dmon -c 10", "nvidia-smi dmon -c 4 | cat", "dcgmi dmon -e 203 -c 4"])
def test_tsh18_the_starvation_pattern_counts_however_dmon_is_run_on_a_terminal(dmon):
    s = concrete("tsh-18")
    shell, _screen = terminal(s.build_state())
    solved("tsh-18", edit=lambda st: st.replace("nvidia-smi dmon -c 10", dmon), shell=shell)
    assert all(marks(s, shell).values()), marks(s, shell)


def test_tsh18_without_dmon_the_criterion_stays_open():
    s = concrete("tsh-18")
    shell, _screen = terminal(s.build_state())
    solved("tsh-18", skip=r"dmon", shell=shell)
    results = marks(s, shell)
    assert not one(results, "starvation pattern") and sum(results.values()) == len(results) - 1


# ---- adm-10: the test pod has to use the MIG slice

CPU_ONLY = (
    "apiVersion: v1\nkind: Pod\nmetadata:\n  name: mig-inference\nspec:\n"
    "  nodeSelector:\n    kubernetes.io/hostname: dgx-04\n"
    '  containers:\n  - name: c\n    image: ubuntu:22.04\n    command: ["sleep", "infinity"]\n'
)
MIG_POD = "mig-inference is Running on dgx-04"


def test_adm10_a_cpu_only_pod_with_the_same_name_does_not_count():
    s, shell = solved("adm-10")
    assert all(marks(s, shell).values())
    shell.ctx.state.nodes["bcm-head"].files["/root/cpu.yaml"] = CPU_ONLY
    shell.run("kubectl delete pod mig-inference")
    shell.run("kubectl apply -f /root/cpu.yaml")
    assert "Running" in shell.run("kubectl get pod mig-inference -o wide")
    results = marks(s, shell)
    assert not one(results, MIG_POD) and sum(results.values()) == len(results) - 1


def test_adm10_the_original_test_pod_recreated_still_counts():
    s, shell = solved("adm-10")
    shell.run("kubectl delete pod mig-inference")
    shell.run("kubectl apply -f /root/mig-inference.yaml")
    assert all(marks(s, shell).values()), marks(s, shell)


# ---- tsh-12: the health check has to come with the watches active

CHECKED = "DCGM health check on dgx-03"


def _tsh12(order: list[str]):
    s = concrete("tsh-12")
    shell = Shell(s.build_state())
    for step in s.solution:
        for line in order if step == "dcgmi health -g 2 -s mn" else ([] if step == "dcgmi health -g 2 -c" else [step]):
            shell.run(line)
    return s, shell


@pytest.mark.parametrize(
    "order",
    [
        ["dcgmi health -g 2 -c", "dcgmi health -g 2 -s mn"],  # the reported case: checked before any watch
        ["dcgmi health -g 2 -s mn"],  # never checked
        ["dcgmi health -g 2 -s mn", "dcgmi health -g 99 -c"],  # a group that does not exist
    ],
    ids=["check-before-watches", "no-check", "wrong-group"],
)
def test_tsh12_a_check_without_active_watches_earns_nothing(order):
    s, shell = _tsh12(order)
    results = marks(s, shell)
    assert not one(results, CHECKED) and sum(results.values()) == len(results) - 1


@pytest.mark.parametrize(
    "order",
    [
        ["dcgmi health -g 2 -s mn", "dcgmi health -g 2 -c"],
        ["dcgmi health -g 2 -c", "dcgmi health -g 2 -s mn", "dcgmi health -g 2 -c"],
        ["dcgmi health -g 2 -s a", "dcgmi health -g 2 -c"],
        ["dcgmi health -g 2 -s mn", "dcgmi health -g 2 -c | grep -i health"],
    ],
    ids=["documented", "check-twice", "all-watches", "filtered"],
)
def test_tsh12_a_check_after_the_watches_passes(order):
    s, shell = _tsh12(order)
    assert all(marks(s, shell).values()), marks(s, shell)


# ---- tsh-17: the topology of the node the incident is on

TOPOLOGY = "GPU/CPU topology of dgx-03"


def test_tsh17_topology_of_a_healthy_node_is_not_the_diagnosis():
    s = concrete("tsh-17")
    shell = Shell(s.build_state())
    for line in ("ssh dgx-01 nvidia-smi topo -m", "ssh dgx-01 numactl -H", "ssh dgx-01 lscpu"):
        shell.run(line)
    assert not one(marks(s, shell), TOPOLOGY)
    shell.run("ssh dgx-03 nvidia-smi topo -m")  # one half on the right node is not enough either
    assert not one(marks(s, shell), TOPOLOGY)
    shell.run("ssh dgx-03 lscpu")
    assert one(marks(s, shell), TOPOLOGY)


@pytest.mark.parametrize("second", ["numactl -H", "numactl --hardware", "lscpu", "lscpu | grep -i numa"])
def test_tsh17_topology_of_dgx03_passes_with_or_without_a_comparison(second):
    s = concrete("tsh-17")
    shell = Shell(s.build_state())
    for line in ("ssh dgx-01 nvidia-smi topo -m", "ssh dgx-03", "nvidia-smi topo -m", second):
        shell.run(line)
    assert one(marks(s, shell), TOPOLOGY)


@pytest.mark.parametrize("code", ["tsh-18", "adm-10", "tsh-12", "tsh-17"])
def test_reference_solution_still_earns_every_criterion(code):
    for s in sc.find(code).all_variants():
        shell = Shell(s.build_state())
        sc.replay_solution(s, shell)
        assert all(ok for _desc, ok, _err in s.grade(shell)), (s.variant_label, s.grade(shell))
