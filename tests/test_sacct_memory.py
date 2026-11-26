"""sacct memory accounting: the request on the job line, sampled usage on the step lines, and tsh-1's records agreeing."""

from __future__ import annotations

import re

import pytest

from conftest import ScriptedUI, concrete
from greenray.app import lab
from greenray.app.lab import LabSession
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

VARIANTS = range(3)


def table(shell: Shell, command: str) -> dict[str, list[str]]:
    """A parsable sacct query as {job or step id: the other fields}."""
    lines = shell.run(command + " -P -n").strip().splitlines()
    return {line.split("|")[0]: line.split("|")[1:] for line in lines}


def oom_lab(variant: int):
    scenario = concrete("tsh-first-look-at-logs", variant)
    job, node = (scenario.variants[variant][key] for key in ("job", "node"))
    return LabSession(scenario).shell, job, node


def test_a_memory_request_shows_on_the_job_line_and_in_both_tres_columns():
    s = Shell(make_cluster())
    s.run("sbatch --mem=64G --gres=gpu:1 --wrap='sleep 600'")
    rows = table(s, "sacct -j 1001 --format=JobID,ReqMem,ReqTRES,AllocTRES")
    assert rows["1001"][0] == "64G"
    assert "mem=64G" in rows["1001"][1] and "mem=64G" in rows["1001"][2]
    assert "mem=64G" in s.run("scontrol show job 1001")


def test_steps_carry_no_request_of_their_own():
    s = Shell(make_cluster())
    s.run("sbatch --mem=64G --wrap='sleep 600'")
    rows = table(s, "sacct -j 1001 --format=JobID,ReqMem,ReqTRES,AllocTRES")
    for step in ("1001.batch", "1001.extern"):
        assert rows[step][0] == "" and rows[step][1] == ""
        assert "mem=64G" in rows[step][2]
    assert "billing" not in rows["1001.batch"][2] and "billing" in rows["1001.extern"][2]


def test_a_job_without_a_memory_request_or_recorded_usage_keeps_empty_usage_columns():
    s = Shell(make_cluster())
    s.run("sbatch --wrap='sleep 600'")
    rows = table(s, "sacct -j 1001 --format=JobID,ReqMem,MaxRSS,MaxVMSize,ReqTRES")
    assert rows["1001"] == ["0M", "", "", "billing=1,cpu=1,mem=0,node=1"]
    assert rows["1001.batch"][1:3] == ["", ""]


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_oom_job_has_its_request_on_the_job_line_and_usage_on_the_steps(variant):
    s, job, _ = oom_lab(variant)
    rows = table(s, f"sacct -j {job} --format=JobID,State,ReqMem,MaxRSS,MaxVMSize")
    assert rows[str(job)] == ["OUT_OF_MEMORY", "200G", "", ""]
    assert rows[f"{job}.batch"][0] == "OUT_OF_MEMORY" and rows[f"{job}.batch"][1] == ""
    assert re.fullmatch(r"\d+K", rows[f"{job}.batch"][2]) and re.fullmatch(r"\d+K", rows[f"{job}.batch"][3])
    assert rows[f"{job}.extern"] == ["COMPLETED", "", "0", "0"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_oom_jobs_accounting_agrees_with_the_kernel_log(variant):
    s, job, node = oom_lab(variant)
    rows = table(s, f"sacct -j {job} --format=JobID,ReqMem,MaxRSS,MaxVMSize,AllocTRES,AllocCPUS")
    requested_kb = int(rows[str(job)][0].removesuffix("G")) * 1024 * 1024
    sampled_kb = int(rows[f"{job}.batch"][1].removesuffix("K"))
    kernel = s.run(f"ssh {node} 'journalctl -k'")
    limit_kb = int(re.search(r"memory: usage \d+kB, limit (\d+)kB", kernel).group(1))
    killed_rss_kb = int(re.search(r"anon-rss:(\d+)kB", kernel).group(1))
    killed_vm_kb = int(re.search(r"total-vm:(\d+)kB", kernel).group(1))
    assert limit_kb == requested_kb  # the cgroup limit is the request
    assert (
        killed_rss_kb < sampled_kb < limit_kb
    )  # the step is the python process plus its workers, sampled before the kill
    assert int(rows[f"{job}.batch"][2].removesuffix("K")) <= killed_vm_kb
    assert f"job_{job}" in kernel and "mem=200G" in rows[str(job)][3]
    # the controller's allocation line and accounting name the same CPU count
    allocated = re.search(r"#CPUs=(\d+)", s.run("journalctl -u slurmctld")).group(1)
    assert rows[str(job)][4] == allocated
    assert "MinMemoryNode=200G" in s.run(f"scontrol show job {job}")


def test_max_vm_size_is_a_listed_field_and_an_unknown_one_is_still_refused():
    s = Shell(make_cluster())
    assert "MaxVMSize" in s.run("sacct --help") and "ReqMem" in s.run("sacct --help")
    assert "Invalid field requested" in s.run("sacct --format=JobID,MaxPages")


def test_the_whole_investigation_still_earns_every_criterion():
    session = LabSession(concrete("tsh-first-look-at-logs", 0))
    for command in (
        "sacct -j 2210",
        "sacct -j 2210 --format=JobID,State%20,ReqMem,MaxRSS",
        "sacct -j 2210 --format=JobID,ReqTRES%60,AllocTRES%60",
        "sacct -j 2210 --format=JobID,MaxRSS,MaxVMSize",
        "ssh dgx-02 'journalctl -u slurmd | grep 2210'",
        "ssh dgx-02 'journalctl -k | grep -i oom'",
        "echo 'Job 2210 ended OUT_OF_MEMORY: it asked for 200G and the kernel OOM killer ended python' > /root/ticket-4471.txt",
    ):
        session.shell.run(command)
    assert [desc for desc, ok, _ in session.grade() if not ok] == []


def test_the_extra_memory_queries_work_in_a_guided_session():
    job = concrete("tsh-first-look-at-logs", 0).variants[0]["job"]
    ui = ScriptedUI([f"sacct -j {job} --format=JobID,State%20,ReqMem,MaxRSS", "check", "quit"])
    lab.run_lab(ui, concrete("tsh-first-look-at-logs", 0))
    assert "200G" in ui.text and "208347136K" in ui.text
