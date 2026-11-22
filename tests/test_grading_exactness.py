"""Criteria grade what was asked (the role, the restart, the GPU count) and what the learner actually saw."""

from __future__ import annotations

import re

import pytest

from conftest import concrete, requires_solutions
from greenray.content import scenarios as sc
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

pytestmark = requires_solutions


def solved(code: str, variant: int = 0, edit=None, skip: str | None = None, first: tuple[str, ...] = ()):
    """The reference solution, with steps rewritten by ``edit`` or left out when they match ``skip``."""
    s = concrete(code, variant)
    shell = Shell(s.build_state())
    for line in first:
        shell.run(line.format(**s.params) if s.params else line)
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


# ---- adm-7: the requested roles and scopes, nothing more

ALICE = '--user alice@corp.ai --project nlp --role "L1 researcher"'
CAROL_NLP = '--user carol@corp.ai --project nlp --role "Research manager"'
RULE = "runai-ui access-rule create "


@pytest.mark.parametrize(
    ("change", "broken"),
    [
        (lambda st: st.replace(ALICE, ALICE.replace("L1 researcher", "Research manager")), "alice is a researcher"),
        (lambda st: st.replace(ALICE, ALICE.replace("nlp", "vision")), "alice is a researcher"),
        (lambda st: st.replace(ALICE, "--user alice@corp.ai,bob@corp.ai --project nlp"), "alice is a researcher"),
        (lambda st: st.replace(CAROL_NLP, CAROL_NLP.replace("Research manager", "L1 researcher")), "carol is Research"),
    ],
    ids=["alice-manager", "alice-on-vision", "bob-also-on-nlp", "carol-researcher"],
)
def test_adm7_a_wrong_role_or_scope_fails(change, broken):
    s, shell = solved("adm-7", edit=change)
    results = marks(s, shell)
    assert not one(results, broken)
    assert one(results, "vision-smoke is Running")


@pytest.mark.parametrize(
    ("extra", "broken"),
    [
        (RULE + "--user alice@corp.ai --project vision", "alice is a researcher"),
        (RULE + '--user carol@corp.ai --project sandbox --role "Research manager"', "carol is Research"),
        (RULE + '--user carol@corp.ai --project nlp --role "L1 researcher"', "carol is Research"),
    ],
    ids=["alice-extra-project", "carol-extra-project", "carol-second-role"],
)
def test_adm7_extra_access_fails(extra, broken):
    s, shell = solved("adm-7")
    assert all(marks(s, shell).values())
    shell.run(extra)
    assert not one(marks(s, shell), broken)


def test_adm7_either_researcher_role_and_the_default_role_pass():
    for change in (
        lambda st: st.replace(ALICE, ALICE.replace("L1 researcher", "L2 researcher")),
        lambda st: st.replace(ALICE, "--user alice@corp.ai --project nlp"),  # the role defaults to L1 researcher
    ):
        s, shell = solved("adm-7", edit=change)
        assert all(marks(s, shell).values()), marks(s, shell)


# ---- adm-13: the repaired queue has been through a restart of cmd

RESTART = "restart of cmd"


@pytest.mark.parametrize("variant", [0, 1])
def test_adm13_needs_a_restart_after_the_repair(variant):
    s, shell = solved("adm-13", variant, skip=r"^systemctl restart cmd")
    results = marks(s, shell)
    assert not one(results, RESTART) and sum(results.values()) == len(results) - 1

    shell.run("systemctl restart cmd")
    assert shell.run("scontrol show partition " + s.params["queue"]).startswith("PartitionName=")
    assert all(marks(s, shell).values())


@pytest.mark.parametrize("variant", [0, 1])
def test_adm13_a_restart_before_the_repair_does_not_count(variant):
    s, shell = solved("adm-13", variant, skip=r"^systemctl restart cmd", first=("systemctl restart cmd",))
    assert not one(marks(s, shell), RESTART)


def test_adm13_a_restart_alone_does_not_count():
    s = concrete("adm-13")
    shell = Shell(s.build_state())
    shell.run("systemctl restart cmd")
    assert not one(marks(s, shell), RESTART)


# ---- wkl-5: exactly two GPUs

TWO = "--gres=gpu:2"


@pytest.mark.parametrize("request_", ["--gres=gpu:4", "--gres=gpu:1", "--gpus=8", "-G 3", "--gres=gpu:2 -N 2"])
def test_wkl5_another_gpu_count_fails(request_):
    s, shell = solved("wkl-5", edit=lambda st: st.replace(TWO, request_))
    results = marks(s, shell)
    assert not one(results, "exactly 2 GPUs") and one(results, "spank_pyxis.so")


@pytest.mark.parametrize("request_", ["--gres=gpu:2", "--gres=gpu:h100:2", "--gpus=2", "-G 2", "--gpus-per-node=2"])
def test_wkl5_every_spelling_of_two_gpus_passes(request_):
    s, shell = solved("wkl-5", edit=lambda st: st.replace(TWO, request_))
    assert all(marks(s, shell).values()), marks(s, shell)


# ---- tsh-11: the Xid, seen before the reboot

KERNEL, BEFORE = "you read the kernel log", "before rebooting or power-cycling"
XID_STEP = "dmesg | grep -i xid"


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize(
    "look",
    [
        'dmesg | grep -i "Linux version"',  # the reported case
        "dmesg | head -1",
        "dmesg | wc -l",
        "dmesg > /root/dmesg.txt",
        "journalctl -k | grep -c NVRM",
        "uname -r",
    ],
)
def test_tsh11_a_look_that_did_not_show_the_xid_earns_nothing(variant, look):
    s, shell = solved("tsh-11", variant, edit=lambda st: st.replace(XID_STEP, look))
    results = marks(s, shell)
    assert not one(results, KERNEL) and not one(results, BEFORE)
    assert sum(results.values()) == len(results) - 2  # the recovery itself is complete


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize(
    "look",
    [
        "dmesg",
        "dmesg -T",
        "dmesg | grep -i xid",
        "dmesg | grep -i 'fallen off'",
        "journalctl -k",
        "journalctl -k | grep NVRM",
    ],
)
def test_tsh11_every_kernel_log_path_that_shows_the_xid_passes(variant, look):
    s, shell = solved("tsh-11", variant, edit=lambda st: st.replace(XID_STEP, look))
    assert all(marks(s, shell).values()), marks(s, shell)


def test_tsh11_the_xid_read_through_ssh_from_the_head_node_passes():
    s = concrete("tsh-11")
    shell = Shell(s.build_state())
    node = s.params["node"]
    shell.run(f'ssh {node} dmesg | grep -i "Linux version"')
    assert not one(marks(s, shell), KERNEL)
    shell.run(f"ssh {node} dmesg | grep -i xid")
    assert one(marks(s, shell), KERNEL)


def test_tsh11_bug_report_and_diagnostic_safeguards_are_unchanged():
    s, shell = solved("tsh-11", skip=r"^nvidia-bug-report")
    results = marks(s, shell)
    assert not one(results, "bug report was collected") and one(results, KERNEL)

    s, shell = solved("tsh-11", skip=r"^dcgmi diag")
    results = marks(s, shell)
    assert not one(results, "diagnostic had passed before you resumed")


# ---- a command piped into a filter, or written to a file, shows only what comes out at the end


def texts(shell: Shell, command: str) -> list[str]:
    return [text for _pos, _host, argv, text, _done in shell.ctx.results if argv[0] == command]


def test_a_piped_command_is_credited_with_what_the_pipeline_displayed():
    shell = Shell(make_cluster())
    shell.run("dmesg | grep -i 'Linux version'")
    assert texts(shell, "dmesg") == texts(shell, "grep") and "NVRM" not in texts(shell, "dmesg")[0]

    shell = Shell(make_cluster())
    shell.run("dmesg")
    assert "NVRM" in texts(shell, "dmesg")[0]

    shell = Shell(make_cluster())
    shell.run("ssh dgx-01 dmesg | grep -c Linux")
    assert texts(shell, "dmesg") == ["1\n"] or texts(shell, "dmesg")[0].strip() == "1"

    shell = Shell(make_cluster())
    shell.run("dmesg > /root/d.txt")
    assert texts(shell, "dmesg") == [""] and "NVRM" in shell.ctx.state.nodes["bcm-head"].files["/root/d.txt"]


FILTERED = [
    (
        "adm-2",
        "scontrol showed you",
        ["scontrol show partition defq | grep MaxTime"],
        ["scontrol show partition | wc -l"],
    ),
    (
        "wkl-8",
        "pending reasons",
        ["scontrol show job 2001 | grep Reason", "squeue | grep 2002"],
        ["squeue | wc -l", "squeue > /root/q.txt", "scontrol show job | grep -c JobId"],
    ),
    (
        "wkl-4",
        "why the scheduler rejects",
        ["kubectl describe pod bert-pretrain | grep -A3 Events"],
        ["kubectl describe pod bert-pretrain | grep Image", "kubectl get events | wc -l"],
    ),
    (
        "tsh-3",
        "clock event reasons",
        ["ssh {node} 'nvidia-smi -q -d PERFORMANCE | grep -i \"power cap\"'"],
        ["ssh {node} 'nvidia-smi -q -d PERFORMANCE | grep -c GPU'"],
    ),
    (
        "tsh-5",
        "row remapper",
        ["ssh {node} 'nvidia-smi -q -d ROW_REMAPPER -i {gpu} | grep -A4 Remapped'"],
        ["ssh {node} 'nvidia-smi -q -d ROW_REMAPPER | grep -c GPU'", "ssh {node} 'nvidia-smi -q > /root/q.txt'"],
    ),
    (
        "tsh-8",
        "you checked port state",
        ["ssh {node} 'ibstat mlx5_{rail} | grep -i state'"],
        ["ssh {node} 'ibstat | grep -c CA'", "ssh {node} 'ibstat mlx5_{rail} | grep Firmware'"],
    ),
    (
        "tsh-15",
        "loaded kernel modules",
        ["ssh dgx-01 'lsmod | grep nvidia_peermem'"],  # nothing comes out: the module is not loaded
        ["ssh dgx-01 cat /proc/modules/nosuch"],
    ),
]


@pytest.mark.parametrize(("code", "wording", "shown", "hidden"), FILTERED, ids=[f[0] for f in FILTERED])
def test_filtered_output_counts_only_when_the_answer_is_still_in_it(code, wording, shown, hidden):
    s = concrete(code)
    shell = Shell(s.build_state())
    for line in hidden:
        shell.run(line.format(**s.params) if s.params else line)
    assert not one(marks(s, shell), wording), hidden
    for line in shown:
        shell.run(line.format(**s.params) if s.params else line)
    assert one(marks(s, shell), wording), shown


@pytest.mark.parametrize("code", ["adm-7", "adm-13", "wkl-5", "tsh-11"])
def test_reference_solution_still_earns_every_criterion(code):
    for s in sc.find(code).all_variants():
        shell = Shell(s.build_state())
        sc.replay_solution(s, shell)
        assert all(ok for _desc, ok, _err in s.grade(shell)), (s.variant_label, s.grade(shell))
