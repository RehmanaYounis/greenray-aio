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


# ---- tsh-4: the settings to keep are read from the parsed daemon.json

TSH4_KEPT = "daemon.json is valid and still has"
MIRROR = "https://registry-mirror.corp.example:5000"


def _tsh4(daemon_json: str) -> tuple[sc.Scenario, Shell]:
    """tsh-4 solved the reference way, with ``daemon_json`` written before nvidia-ctk and the restart."""
    s, shell = lab("tsh-4")
    for step in s.solution:
        lines = step.split("\n")
        if lines[0].startswith("cat > /etc/docker/daemon.json"):
            lines = [lines[0], *daemon_json.strip().split("\n"), "EOF"]
        feed = iter(lines[1:])
        shell.run(lines[0], heredoc_reader=lambda _p, feed=feed: next(feed, None))
    return s, shell


def _daemon(driver: str = "json-file", size: str = "100m", files: str | None = "5", mirror: str | None = MIRROR) -> str:
    opts = [f'"max-size": "{size}"'] + ([f'"max-file": "{files}"'] if files is not None else [])
    mirrors = f'["{mirror}"]' if mirror else "[]"
    return f'{{"log-driver": "{driver}", "log-opts": {{{", ".join(opts)}}}, "registry-mirrors": {mirrors}}}'


@pytest.mark.parametrize(
    "daemon_json",
    [
        _daemon(files="1"),  # the reported case
        _daemon(files=None),
        _daemon(size="10m"),
        _daemon(driver="local"),
        _daemon(mirror=None),
        _daemon(mirror="https://mirror.gcr.io"),
        '{"registry-mirrors": ["' + MIRROR + '"], "comment": "max-size 100m max-file 5 json-file"}',
    ],
    ids=["max-file-1", "no-max-file", "max-size-10m", "other-driver", "no-mirror", "other-mirror", "words-only"],
)
def test_tsh4_a_changed_or_dropped_setting_fails_the_criterion(daemon_json):
    s, shell = _tsh4(daemon_json)
    assert not criterion(s, shell, TSH4_KEPT)
    assert criterion(s, shell, "infer-triton-a-1") and criterion(s, shell, "infer-triton-b-1")


@pytest.mark.parametrize(
    "daemon_json",
    [
        _daemon(),  # one line
        '{\n  "registry-mirrors": [\n    "' + MIRROR + '"\n  ],\n  "log-opts": {"max-file": "5", "max-size": "100m"},\n'
        '  "log-driver": "json-file"\n}',  # reordered and re-indented
        _daemon(mirror=MIRROR + "/"),
        _daemon()[:-1] + ', "live-restore": true}',  # an extra setting
    ],
    ids=["compact", "reordered", "trailing-slash", "extra-key"],
)
def test_tsh4_json_formatting_does_not_matter(daemon_json):
    s, shell = _tsh4(daemon_json)
    assert all(ok for _desc, ok, _err in s.grade(shell)), s.grade(shell)


def test_tsh4_a_file_broken_after_the_restart_fails():
    s, shell = _tsh4(_daemon())
    files = shell.ctx.state.nodes["dgx-02"].files
    files["/etc/docker/daemon.json"] = files["/etc/docker/daemon.json"].rstrip().rstrip("}") + ",\n}\n"
    assert not criterion(s, shell, TSH4_KEPT)


# ---- adm-4: the ticket's initial password

ADM4_REST = (
    'cmsh -c "group; use mlteam; append members maria; commit"',
    "sacctmgr -i add user maria account=vision",
    'su - maria -c "sbatch --wrap=hostname"',
)
ADM4_PASSWORD = "initial password"


def _adm4(password_clause: str) -> tuple[sc.Scenario, Shell]:
    s, shell = lab("adm-4")
    do(shell, f'cmsh -c "user; add maria; set id 2010; {password_clause}set profile readonly; commit"', *ADM4_REST)
    return s, shell


@pytest.mark.parametrize("clause", ["set password Different-Pass1; ", "set password welcome-2026!; ", ""])
def test_adm4_another_password_or_none_leaves_one_criterion_open(clause):
    s, shell = _adm4(clause)
    assert not criterion(s, shell, ADM4_PASSWORD)
    assert sum(not ok for _desc, ok, _err in s.grade(shell)) == (1 if clause else 2)  # none: the user check fails too


@pytest.mark.parametrize("clause", ["set password Welcome-2026!; ", "set password 'Welcome-2026!'; "])
def test_adm4_the_tickets_password_passes(clause):
    s, shell = _adm4(clause)
    assert all(ok for _desc, ok, _err in s.grade(shell)), s.grade(shell)


def test_adm4_grading_output_does_not_show_the_password():
    s, shell = _adm4("set password Different-Pass1; ")
    for desc, _ok, err in s.grade(shell):
        assert "Welcome" not in desc and "Welcome" not in err
    assert "Welcome-2026!" in s.briefing  # the ticket is where the learner reads it


@requires_solutions
@pytest.mark.parametrize("code", ["ins-6", "tsh-4", "adm-4"])
def test_reference_solution_still_earns_every_criterion(code):
    s, shell = lab(code)
    sc.replay_solution(s, shell)
    assert all(ok for _desc, ok, _err in s.grade(shell)), s.grade(shell)
