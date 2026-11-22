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
