"""A dry run counts when it came first, and a proof job when it proves what the briefing asks."""

from __future__ import annotations

import pytest

from conftest import concrete, requires_solutions
from greenray.content import scenarios as sc
from greenray.core.shell import Shell

pytestmark = requires_solutions


def lab(code: str) -> tuple[sc.Scenario, Shell]:
    s = concrete(code)
    return s, Shell(s.build_state())


def do(shell: Shell, *lines: str) -> None:
    for line in lines:
        shell.run(line)


def results(s: sc.Scenario, shell: Shell) -> dict[str, bool]:
    out = {}
    for desc, ok, err in s.grade(shell):
        assert not err, (desc, err)
        out[desc] = ok
    return out


def one(found: dict[str, bool], wording: str) -> bool:
    hits = [ok for desc, ok in found.items() if wording in desc]
    assert len(hits) == 1, (wording, list(found))
    return hits[0]


# ---- ins-7: the imageupdate dry run comes before the live update of the same node

INS7_DRY = "dry run before its live update"
IMAGEUPDATE = 'cmsh -c "device; imageupdate {flags}"'


def _ins7_prepared() -> tuple[sc.Scenario, Shell]:
    """New image built, category switched, dgx-02's own image override cleared: everything but the rollout."""
    s, shell = lab("ins-7")
    do(shell, *(step for step in s.solution if "imageupdate" not in step and "synclog" not in step))
    return s, shell


def test_ins7_a_dry_run_after_the_rollout_does_not_count():
    s, shell = _ins7_prepared()
    do(shell, IMAGEUPDATE.format(flags="-w -c dgx-h100"), IMAGEUPDATE.format(flags="-c dgx-h100"))
    found = results(s, shell)
    assert not one(found, INS7_DRY) and sum(found.values()) == len(found) - 1


def test_ins7_no_dry_run_at_all_does_not_count():
    s, shell = _ins7_prepared()
    do(shell, IMAGEUPDATE.format(flags="-w -c dgx-h100"))
    assert not one(results(s, shell), INS7_DRY)


@pytest.mark.parametrize(
    "rollout",
    [
        ["-c dgx-h100", "-w -c dgx-h100"],
        ["-n dgx-01,dgx-02", "-w -n dgx-01,dgx-02", "-n dgx-03,dgx-04", "-w -n dgx-03,dgx-04"],  # two batches
        ["-c dgx-h100", "-w -n dgx-01", "-w -n dgx-02..dgx-04"],  # one preview, a canary, then the rest
        ["-c dgx-h100", "-w -c dgx-h100", "-c dgx-h100"],  # a look afterwards does no harm
    ],
    ids=["all-at-once", "two-batches", "canary", "extra-dry-run-after"],
)
def test_ins7_a_dry_run_before_each_nodes_update_passes(rollout):
    s, shell = _ins7_prepared()
    do(shell, *(IMAGEUPDATE.format(flags=flags) for flags in rollout))
    assert all(results(s, shell).values()), results(s, shell)


def test_ins7_a_batch_without_its_own_dry_run_fails():
    s, shell = _ins7_prepared()
    flags = ["-n dgx-01,dgx-02", "-w -n dgx-01,dgx-02", "-w -n dgx-03,dgx-04"]
    do(shell, *(IMAGEUPDATE.format(flags=f) for f in flags))
    found = results(s, shell)
    assert not one(found, INS7_DRY) and one(found, "every DGX node runs")


def test_ins7_rollback_image_and_no_reboot_are_still_graded():
    s, shell = _ins7_prepared()
    do(shell, IMAGEUPDATE.format(flags="-c dgx-h100"), IMAGEUPDATE.format(flags="-w -c dgx-h100"))
    found = results(s, shell)
    assert one(found, "rollback target untouched") and one(found, "no DGX node was rebooted")
    do(shell, 'cmsh -c "device; reboot -n dgx-03"')
    assert not one(results(s, shell), "no DGX node was rebooted")


# ---- adm-9: the grabimage dry run comes before the real grab

ADM9_DRY = "grabimage dry run of dgx-01"
EXCLUDE = "cmsh -c \"category; use dgx-h100; append excludelistgrab '- /root/.ssh/id_*'; commit\""
GRAB = 'cmsh -c "device; grabimage {flags}"'
ROLL_OUT = ('cmsh -c "device; imageupdate -n dgx-02..dgx-04"', 'cmsh -c "device; imageupdate -w -n dgx-02..dgx-04"')


@pytest.mark.parametrize(
    "grabs",
    [
        ["-w dgx-01", "dgx-01"],  # the reported case: the preview came after the write
        ["-w dgx-01"],
        ["dgx-02", "-w dgx-01"],  # a preview of another node
    ],
    ids=["dry-run-after", "no-dry-run", "dry-run-of-another-node"],
)
def test_adm9_a_real_grab_without_a_prior_dry_run_fails(grabs):
    s, shell = lab("adm-9")
    do(shell, EXCLUDE, *(GRAB.format(flags=f) for f in grabs), *ROLL_OUT)
    found = results(s, shell)
    assert not one(found, ADM9_DRY) and sum(found.values()) == len(found) - 1


@pytest.mark.parametrize(
    "grabs", [["dgx-01", "-w dgx-01"], ["dgx-01", "-w dgx-01", "dgx-01"]], ids=["before", "and-after"]
)
def test_adm9_a_dry_run_before_the_real_grab_passes(grabs):
    s, shell = lab("adm-9")
    do(shell, EXCLUDE, *(GRAB.format(flags=f) for f in grabs), *ROLL_OUT)
    assert all(results(s, shell).values()), results(s, shell)


def test_adm9_private_key_and_no_reboot_are_still_graded():
    s, shell = lab("adm-9")
    do(shell, GRAB.format(flags="dgx-01"), GRAB.format(flags="-w dgx-01"), *ROLL_OUT)  # no exclude list entry
    found = results(s, shell)
    assert not one(found, "private SSH key is not in the image") and one(found, ADM9_DRY)
    do(shell, 'cmsh -c "device; reboot -n dgx-02"')
    assert not one(results(s, shell), "no DGX node was rebooted")


@pytest.mark.parametrize("code", ["ins-7", "adm-9"])
def test_reference_solution_still_earns_every_criterion(code):
    s, shell = lab(code)
    sc.replay_solution(s, shell)
    assert all(results(s, shell).values()), results(s, shell)
