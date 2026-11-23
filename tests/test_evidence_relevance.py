"""Evidence has to be about the thing that is broken: the affected GPU, port, job, pod, node or package.

Each case runs a query that succeeds and prints plausible output about something else, and expects the
criterion to stay open; then the queries that do show the affected object, and expects it to be met.
"""

from __future__ import annotations

import re

import pytest

from conftest import concrete, requires_solutions
from greenray.content import scenarios as sc
from greenray.core.shell import Shell


def lab(code: str, variant: int = 0) -> tuple[sc.Scenario, Shell]:
    s = concrete(code, variant)
    return s, Shell(s.build_state())


def met(s: sc.Scenario, shell: Shell, index: int) -> bool:
    _desc, ok, err = s.grade(shell)[index]
    assert not err, err
    return ok


def after(code: str, index: int, lines: list[str], variant: int = 0) -> bool:
    """Whether criterion ``index`` is met after ``lines`` in a fresh attempt."""
    s, shell = lab(code, variant)
    for line in lines:
        shell.run(line.format(**s.params) if s.params else line)
    return met(s, shell, index)


def solution_without(s: sc.Scenario, shell: Shell, skip: str) -> None:
    """Replay the reference solution, leaving out the steps matching ``skip``."""
    for step in s.solution:
        lines = step.split("\n")
        if re.search(skip, re.sub(r"^ssh \S+ (?=\S)", "", lines[0])):
            continue
        feed = iter(lines[1:])
        shell.run(lines[0], heredoc_reader=lambda _p, feed=feed: next(feed, None))


# ---- tsh-6: both current versions, before the service is started

START = "systemctl enable --now nvidia-fabricmanager"


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize(
    "looked_at",
    [
        ["journalctl -u nvidia-fabricmanager", "dpkg -l docker-ce"],  # the reported case
        ["journalctl -u nvidia-fabricmanager", "systemctl status nvidia-fabricmanager"],  # yesterday's mismatch only
        ["nv-fabricmanager --version"],  # one of the two versions
        ["nvidia-smi"],
        ["dpkg -l nvidia-fabricmanager-550"],
        ["dpkg -l nvidia-driver-550-server", "dpkg -l docker-ce"],
    ],
    ids=["unrelated-package", "journal-only", "fm-only", "driver-only", "fm-package-only", "driver-package-only"],
)
def test_tsh6_needs_both_current_versions(variant, looked_at):
    assert not after("tsh-6", 4, ["ssh {node}", *looked_at, START, "deviceQuery"], variant)


@pytest.mark.parametrize(
    "looked_at",
    [
        ["nv-fabricmanager --version", "nvidia-smi"],  # the documented workflow
        ["nvidia-smi", "nv-fabricmanager -v"],
        ["dpkg -l"],
        ["dpkg -l | grep -i -E 'fabric|nvidia-driver'"],
        ["cat /var/log/apt/history.log"],
        ["dpkg -s nvidia-fabricmanager-550", "nvidia-smi --query-gpu=driver_version --format=csv,noheader"],
        ["apt-cache policy nvidia-fabricmanager-550", "cat /proc/driver/nvidia/version"],
        ["nv-fabricmanager --version", "modinfo nvidia"],
        ["grep fabricmanager /var/log/apt/history.log", "nvidia-smi -q"],
    ],
)
def test_tsh6_accepts_every_way_to_see_both_versions(looked_at):
    s, shell = lab("tsh-6")
    for line in [f"ssh {s.params['node']}", "journalctl -u nvidia-fabricmanager", *looked_at, START, "deviceQuery"]:
        shell.run(line)
    assert all(ok for _desc, ok, _err in s.grade(shell)), s.grade(shell)


def test_tsh6_versions_read_after_the_start_do_not_count():
    assert not after("tsh-6", 4, ["ssh {node}", START, "nv-fabricmanager --version", "nvidia-smi"])


# ---- tsh-5 and tsh-10: the query has to cover the affected GPU

GPU_LABS = [
    ("tsh-5", 0, "-d ROW_REMAPPER", "remapped_rows.pending"),
    ("tsh-10", 4, "-d TEMPERATURE", "temperature.gpu"),
]


@pytest.mark.parametrize(("code", "index", "display", "field"), GPU_LABS, ids=["tsh-5", "tsh-10"])
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_a_query_for_a_healthy_gpu_earns_nothing(code, index, display, field, variant):
    s = concrete(code, variant)
    other = (s.params["gpu"] + 1) % 8
    peer = "dgx-01" if s.params["node"] != "dgx-01" else "dgx-02"
    for line in (
        f"ssh {{node}} nvidia-smi -i {other} -q {display}",
        f"ssh {{node}} nvidia-smi -q {display} -i {other}",
        f"ssh {{node}} nvidia-smi --query-gpu={field} --format=csv -i {other}",
        f"ssh {{node}} nvidia-smi --id={other} --query-gpu=index,{field} --format=csv",
        "ssh {node} nvidia-smi",
        f"ssh {peer} nvidia-smi -q {display}",
    ):
        assert not after(code, index, [line], variant), line


@pytest.mark.parametrize(("code", "index", "display", "field"), GPU_LABS, ids=["tsh-5", "tsh-10"])
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_a_query_that_covers_the_affected_gpu_earns_evidence(code, index, display, field, variant):
    other = (concrete(code, variant).params["gpu"] + 1) % 8
    for line in (
        f"ssh {{node}} nvidia-smi -q {display}",  # every GPU
        f"ssh {{node}} nvidia-smi -q {display} -i {{gpu}}",
        "ssh {node} nvidia-smi -i {gpu} -q",
        "ssh {node} nvidia-smi -q",
        f"ssh {{node}} nvidia-smi --query-gpu=index,{field} --format=csv",
        f"ssh {{node}} nvidia-smi --query-gpu={field} --format=csv,noheader -i {{gpu}}",
        f"ssh {{node}} nvidia-smi -i {other},{{gpu}} --query-gpu={field} --format=csv",
    ):
        assert after(code, index, [line], variant), line


def test_tsh5_the_reported_query_leaves_only_the_safety_criterion_met():
    s, shell = lab("tsh-5", 2)
    assert s.params["node"] == "dgx-03" and s.params["gpu"] == 5
    out = shell.run("ssh dgx-03 nvidia-smi -i 0 -q -d ROW_REMAPPER")
    assert "Pending                           : No" in out
    assert [ok for _desc, ok, _err in s.grade(shell)] == [False, False, False, True, False, False, False]


# ---- wkl-8: the reason of each pending job


@pytest.mark.parametrize(
    "lines",
    [
        ["squeue -t RUNNING -l"],  # the reported case: a header and no jobs
        ["squeue -u alice", "sinfo"],  # one job only
        ["squeue -j 2001", "scontrol show partition batch"],
        ["scontrol show job 2001"],
        ["squeue -h -o %r"],  # reasons that name no job
        ["sprio", "sinfo"],
        ["squeue -u carol", "scontrol show job 999"],
    ],
)
def test_wkl8_needs_the_reason_of_both_pending_jobs(lines):
    assert not after("wkl-8", 4, lines)


@pytest.mark.parametrize(
    "lines",
    [
        ["squeue"],
        ["squeue -l"],
        ["squeue -o '%i %r'"],
        ["squeue -h -o '%i %r'"],
        ["squeue -t PD"],
        ["squeue --start"],
        ["squeue -u alice", "squeue -u bob"],
        ["squeue -p batch", "squeue -p short"],
        ["squeue -j 2001", "scontrol show job 2002"],
        ["scontrol show job"],
        ["scontrol show job 2001", "scontrol show job 2002"],
    ],
)
def test_wkl8_accepts_squeue_and_scontrol_views_of_both_jobs(lines):
    assert after("wkl-8", 4, lines)


def test_wkl8_the_reported_query_earns_nothing():
    s, shell = lab("wkl-8")
    out = shell.run("squeue -t RUNNING -l")
    assert "NODELIST(REASON)" in out and "2001" not in out
    assert not any(ok for _desc, ok, _err in s.grade(shell))


# ---- tsh-8 and tsh-15: the port that is down or slow


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_tsh8_a_healthy_port_earns_nothing(variant):
    rail = concrete("tsh-8", variant).params["rail"]
    other = (rail + 1) % 8
    for index, line in (
        (2, f"ssh {{node}} ibstat mlx5_{other}"),
        (2, f"ssh {{node}} ibstatus mlx5_{other}"),
        (2, "ssh {peer} ibstat"),
        (3, f"ssh {{node}} mlxlink -d mlx5_{other}"),
        (3, f"ssh {{node}} mlxlink -d mlx5_{other} -m"),
        (3, "ssh {peer} mlxlink -d mlx5_{rail} -m"),
    ):
        assert not after("tsh-8", index, [line], variant), line


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_tsh8_the_affected_port_earns_evidence(variant):
    for index, line in (
        (2, "ssh {node} ibstat"),
        (2, "ssh {node} ibstat mlx5_{rail}"),
        (2, "ssh {node} ibstatus mlx5_{rail}"),
        (2, "ssh {node} ibv_devinfo -d mlx5_{rail}"),
        (3, "ssh {node} mlxlink -d mlx5_{rail}"),
        (3, "ssh {node} mlxlink -d mlx5_{rail} -m"),
    ):
        assert after("tsh-8", index, [line], variant), line


def test_tsh8_diagnostics_of_a_healthy_port_before_the_reseat_do_not_count():
    s, shell = lab("tsh-8")
    node, rail = s.params["node"], s.params["rail"]
    shell.run(f"ssh {node} mlxlink -d mlx5_{(rail + 1) % 8} -m")
    shell.run(f"reseat-cable {node} mlx5_{rail}")
    assert not met(s, shell, 4)


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_tsh15_needs_the_rate_of_the_slow_port(variant):
    params = concrete("tsh-15", variant).params
    other = (params["rail"] + 1) % 8
    peer = "dgx-03" if params["slow"] != "dgx-03" else "dgx-04"
    for line in (f"ssh {{slow}} ibstat mlx5_{other}", f"ssh {{slow}} mlxlink -d mlx5_{other}", f"ssh {peer} ibstat"):
        assert not after("tsh-15", 2, [line], variant), line
    for line in (
        "ssh {slow} ibstat",
        "ssh {slow} ibstat mlx5_{rail}",
        "ssh {slow} ibstatus mlx5_{rail}",
        "ssh {slow} mlxlink -d mlx5_{rail}",
    ):
        assert after("tsh-15", 2, [line], variant), line


# ---- tsh-13: the log of a plugin pod that is crashing


def test_tsh13_needs_the_log_of_a_crashing_plugin_pod():
    s, shell = lab("tsh-13")
    pods = [p for p in shell.ctx.state.k8s.pods if p.name.startswith("nvidia-device-plugin-daemonset-")]
    healthy = next(p.name for p in pods if p.status == "Running")
    crashing = next(p.name for p in pods if p.status != "Running")
    assert "Registered device plugin" in shell.run(f"kubectl logs -n kube-system {healthy}")
    assert not met(s, shell, 5)
    assert "error starting plugins" in shell.run(f"kubectl logs -n kube-system {crashing}")
    assert met(s, shell, 5)
    assert after("tsh-13", 5, ["kubectl logs -n kube-system -l app=nvidia-device-plugin-daemonset --tail=8"])


# ---- tsh-16, tsh-19, adm-15: "looked before changing" has to be a look at the right thing


@requires_solutions
@pytest.mark.parametrize(
    ("code", "index", "skip", "unrelated", "related"),
    [
        (
            "tsh-16",
            1,
            r'^cmsh -c "(device; (use \S+; show|list)|category; list)',
            ['cmsh -c "partition; list"', 'cmsh -c "device; use dgx-04; show"'],
            'cmsh -c "device; use {node}; get category"',
        ),
        ("tsh-19", 0, r"^nvsm show", ["ssh dgx-01 nvsm show health"], "ssh {node} nvsm show health"),
        (
            "adm-15",
            5,
            r"latestmetricdata|nvidia-smi|ipmitool",
            ['cmsh -c "device; latestmetricdata -n dgx-04"'],
            'cmsh -c "device; latestmetricdata -n {node}"',
        ),
    ],
    ids=["tsh-16", "tsh-19", "adm-15"],
)
def test_a_look_at_something_else_before_the_change_does_not_count(code, index, skip, unrelated, related):
    s, shell = lab(code)
    for line in unrelated:
        shell.run(line.format(**s.params))
    solution_without(s, shell, skip)
    assert not met(s, shell, index)

    s, shell = lab(code)
    shell.run(related.format(**s.params))
    solution_without(s, shell, skip)
    assert met(s, shell, index)


@requires_solutions
@pytest.mark.parametrize(
    "code", ["tsh-5", "tsh-6", "tsh-8", "tsh-10", "tsh-13", "tsh-15", "tsh-16", "tsh-19", "adm-15", "wkl-8"]
)
def test_reference_solution_still_earns_every_criterion(code):
    for s in sc.find(code).all_variants():
        shell = Shell(s.build_state())
        sc.replay_solution(s, shell)
        assert all(ok for _desc, ok, _err in s.grade(shell)), (s.variant_label, s.grade(shell))


# ---- tsh-9: the evidence has to be the authentication failure, not any log of any node

SAW, SAW_FIRST = "you saw the authentication failure", "you saw it before"
DIAGNOSTIC = r"journalctl|unmunge|md5sum|systemctl status"


def _tsh9(variant: int, looks: list[str]) -> tuple[sc.Scenario, Shell]:
    """The repair of tsh-9 without the solution's own diagnostic steps, after ``looks``."""
    s, shell = lab("tsh-9", variant)
    for line in looks:
        shell.run(line.format(**s.params, peer=s.params["others"].split("'")[1]))
    solution_without(s, shell, DIAGNOSTIC)
    return s, shell


def _open(s: sc.Scenario, shell: Shell) -> list[str]:
    return [desc for desc, ok, err in s.grade(shell) if not ok and not err]


@requires_solutions
@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize(
    "looks",
    [
        ["ssh {peer} journalctl -u slurmd"],  # the reported case: a healthy peer's startup log
        ["ssh {peer} journalctl -u munge", "journalctl -u munge"],
        ["ssh {node} journalctl -u munge"],  # munged started fine on the node: no failure in this log
        ["ssh {peer} munge -n | unmunge"],  # a credential test between two healthy hosts succeeds
        ["ssh {node} 'munge -n | unmunge'"],  # the node decodes its own credential
        ["journalctl -u slurmd"],  # the head node runs no slurmd
        ["ssh {node} journalctl -u slurmd | grep -c slurmd"],
    ],
    ids=[
        "peer-slurmd",
        "healthy-munge-logs",
        "node-munge-log",
        "healthy-pair",
        "same-node-test",
        "head-slurmd",
        "count",
    ],
)
def test_tsh9_logs_without_the_authentication_failure_earn_nothing(variant, looks):
    s, shell = _tsh9(variant, looks)
    still_open = _open(s, shell)
    assert len(still_open) == 2 and SAW in still_open[0] and SAW_FIRST in still_open[1], still_open


@requires_solutions
@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize(
    "looks",
    [
        ["ssh {node} journalctl -u slurmd"],
        ["ssh {node} systemctl status slurmd"],
        ["journalctl -u slurmctld"],
        ["journalctl -u slurmctld | grep -i munge"],
        ["ssh {node} munge -n | unmunge"],  # a credential from the node does not decode on the head node
        ["ssh {peer} journalctl -u slurmd", "ssh {node} journalctl -u slurmd"],  # a comparison with a healthy peer
    ],
    ids=["node-slurmd", "node-status", "controller", "controller-filtered", "cross-node-test", "peer-then-node"],
)
def test_tsh9_every_view_of_the_authentication_failure_passes(variant, looks):
    s, shell = _tsh9(variant, looks)
    assert _open(s, shell) == [], _open(s, shell)


@requires_solutions
def test_tsh9_evidence_gathered_after_the_repair_is_too_late():
    s, shell = _tsh9(2, [])
    shell.run("journalctl -u slurmctld")  # the old errors are still in the journal
    still_open = _open(s, shell)
    assert len(still_open) == 1 and SAW_FIRST in still_open[0], still_open


@requires_solutions
def test_tsh9_reference_solution_passes_in_every_variant():
    for s in sc.find("tsh-9").all_variants():
        shell = Shell(s.build_state())
        sc.replay_solution(s, shell)
        assert all(ok for _desc, ok, _err in s.grade(shell)), (s.variant_label, s.grade(shell))
