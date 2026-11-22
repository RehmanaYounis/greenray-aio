"""A lab grades what its briefing asks for: the completed job, the settings to keep, the ticket's password."""

from __future__ import annotations

import pytest

from conftest import concrete, requires_solutions
from greenray.content import scenarios as sc
from greenray.core.shell import Shell


def lab(code: str) -> tuple[sc.Scenario, Shell]:
    s = concrete(code)
    return s, Shell(s.build_state())


def criterion(s: sc.Scenario, shell: Shell, wording: str) -> bool:
    found = [(ok, err) for desc, ok, err in s.grade(shell) if wording in desc]
    assert len(found) == 1 and not found[0][1], (wording, found)
    return found[0][0]


def do(shell: Shell, *lines: str) -> None:
    for line in lines:
        shell.run(line)


# ---- ins-6: the diagnosis covers the DGX mounts and the head node's exports

INS6_SEEN = "you checked what a DGX node has mounted"


@pytest.mark.parametrize(
    "lines",
    [
        ["df -h /"],  # the reported case: the head node's root filesystem
        ["df -h", "showmount -e master"],  # nothing from a DGX node
        ["ssh dgx-02 df -h /", "showmount -e master"],  # a DGX node, but only its root filesystem
        ["ssh dgx-02 df -h"],  # the mounts without the exports
        ["ssh dgx-02 findmnt -t nfs,nfs4"],
        ["showmount -e master"],  # the exports without the mounts
        ["ssh dgx-02 findmnt --bogus /home", "showmount -e master"],  # a refused query is not an empty result
        ["ssh dgx-02 df -h", "showmount -e nosuchhost"],
    ],
)
def test_ins6_diagnosis_needs_the_dgx_mounts_and_the_head_node_exports(lines):
    s, shell = lab("ins-6")
    do(shell, *lines)
    assert not criterion(s, shell, INS6_SEEN)


@pytest.mark.parametrize(
    "lines",
    [
        ["ssh dgx-02 df -h", "showmount -e master"],
        ["ssh dgx-02 findmnt -t nfs,nfs4", "showmount -e"],  # no NFS mounts: an empty answer is the finding
        ["ssh dgx-01 findmnt /home", "exportfs -v"],
        ["ssh dgx-03 'mount | grep nfs'", "cat /etc/exports"],
        ["ssh dgx-04 mount", "ssh dgx-04 showmount -e master"],
        ["ssh dgx-02 df -h /home", "showmount -e master"],
        ["ssh dgx-02", "findmnt", "exit", "showmount -e master"],
    ],
)
def test_ins6_accepts_every_way_to_compare_mounts_with_exports(lines):
    s, shell = lab("ins-6")
    do(shell, *lines)
    assert criterion(s, shell, INS6_SEEN)


# ---- ins-6: "show that it completes"

INS6_REPAIR = (
    'cmsh -c "category; use dgx-h100; fsmounts; add /home; set device master:/home; set filesystem nfs; '
    'set mountoptions rsize=32768,wsize=32768,hard,async; use /cm/shared; set device master:/cm/shared; commit"'
)
INS6_SUBMIT = 'su - alice -c "sbatch /home/alice/llm/train.sbatch"'
INS6_JOB = "llm-ft job completed"


def test_ins6_a_running_job_is_not_a_completed_job():
    s, shell = lab("ins-6")
    do(shell, "ssh dgx-02 df -h", "showmount -e master", INS6_REPAIR)
    assert "Submitted batch job 4002" in shell.run(INS6_SUBMIT)
    assert " R " in shell.run("squeue -u alice")
    results = [ok for _desc, ok, _err in s.grade(shell)]
    assert results == [True, True, True, True, True, False]

    for _ in range(8):  # every command moves the simulated clock on
        if "COMPLETED" in shell.run("sacct -j 4002 -n -X -o State"):
            break
    assert all(ok for _desc, ok, _err in s.grade(shell))


def test_ins6_a_job_that_fails_again_earns_nothing():
    s, shell = lab("ins-6")
    shell.run(INS6_SUBMIT)  # before the repair: the home directory is still missing on the node
    for _ in range(8):
        shell.run("squeue -u alice")
    assert "FAILED" in shell.run("sacct -j 4002 -n -X -o State")
    assert not criterion(s, shell, INS6_JOB)


@requires_solutions
@pytest.mark.parametrize("code", ["ins-6"])
def test_reference_solution_still_earns_every_criterion(code):
    s, shell = lab(code)
    sc.replay_solution(s, shell)
    assert all(ok for _desc, ok, _err in s.grade(shell)), s.grade(shell)
