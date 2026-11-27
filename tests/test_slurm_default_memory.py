"""Memory a job is allocated when it names none: taken from the cluster's configuration, shown the same everywhere."""

from __future__ import annotations

import re

import pytest

from greenray.commands.slurm.scheduler import _alloc_mem_mb, _ensure, _holds_memory
from greenray.content import scenarios as sc
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

PER_CPU = 8192  # DefMemPerCPU in the simulated slurm.conf


def cluster(*edits: str) -> Shell:
    """A fresh cluster, after these shell edits of slurm.conf and a reconfigure."""
    s = Shell(make_cluster())
    for edit in edits:
        s.run(edit)
    if edits:
        s.run("scontrol reconfigure")
    return s


NO_DEFAULT = "sed -i '/^DefMemPerCPU/d' /etc/slurm/slurm.conf"
NOT_TRACKED = "sed -i 's/CR_Core_Memory/CR_Core/' /etc/slurm/slurm.conf"


def real_memory(shell: Shell, node: str) -> int:
    return int(re.search(r"RealMemory=(\d+)", shell.run(f"scontrol show node {node}")).group(1))


def sacct(shell: Shell, job: int | str, fields: str) -> list[str]:
    return shell.run(f"sacct -j {job} -X -P -n --format={fields}").strip().split("|")


def steps(shell: Shell, job: int, fields: str) -> dict[str, list[str]]:
    lines = shell.run(f"sacct -j {job} -P -n --format=JobID,{fields}").strip().splitlines()
    return {line.split("|")[0]: line.split("|")[1:] for line in lines}


def test_the_cluster_sets_a_default_per_cpu_and_reports_it():
    s = cluster()
    assert f"DefMemPerCPU={PER_CPU}" in s.run("cat /etc/slurm/slurm.conf")
    assert f"DefMemPerCPU={PER_CPU}" in s.run("scontrol show partition defq")


def test_a_job_without_mem_gets_the_default_for_its_cpus_everywhere():
    s = cluster()
    s.run("sbatch -w dgx-01 -c 4 --gres=gpu:1 --wrap='sleep 600'")
    assert sacct(s, 1001, "ReqMem,ReqTRES,AllocTRES") == [
        "32G",
        "billing=4,cpu=4,gres/gpu=1,mem=32G,node=1",
        "billing=4,cpu=4,gres/gpu=1,mem=32G,node=1",
    ]
    show = s.run("scontrol show job 1001")
    assert "MinMemoryCPU=8G" in show and "AllocTRES=cpu=4,mem=32G,node=1" in show
    node = s.run("scontrol show node dgx-01")
    assert f"AllocMem={4 * PER_CPU}" in node and "AllocTRES=cpu=4,mem=32G,gres/gpu=1" in node


def test_jobs_without_mem_share_a_node():
    s = cluster()
    s.run("sbatch -w dgx-01 --gres=gpu:1 --wrap='sleep 600'")
    s.run("sbatch -w dgx-01 --gres=gpu:1 --wrap='sleep 600'")
    assert sacct(s, 1001, "State")[0] == sacct(s, 1002, "State")[0] == "RUNNING"
    assert "dgx-01" in s.run("srun -w dgx-01 hostname")


def test_an_exclusive_job_is_given_the_default_for_every_cpu_of_the_node():
    s = cluster()
    s.run("sbatch --exclusive -w dgx-02 --wrap='sleep 600'")
    cpus = int(re.search(r"CPUTot=(\d+)", s.run("scontrol show node dgx-02")).group(1))
    assert f"AllocMem={cpus * PER_CPU}" in s.run("scontrol show node dgx-02")
    assert f"cpu={cpus}," in sacct(s, 1001, "AllocTRES")[0] and "mem=1792G" in sacct(s, 1001, "AllocTRES")[0]


def test_without_a_default_a_job_asks_for_nothing_and_is_allocated_the_whole_node():
    s = cluster(NO_DEFAULT)
    s.run("sbatch -w dgx-01 --gres=gpu:1 --wrap='sleep 600'")
    node_mb = real_memory(s, "dgx-01")
    req_mem, req_tres, alloc_tres = sacct(s, 1001, "ReqMem,ReqTRES,AllocTRES")
    assert req_mem == "0" and "mem=0," in req_tres
    assert f"mem={node_mb}M" in alloc_tres
    show = s.run("scontrol show job 1001")
    assert "MinMemoryNode=0M" in show and f"AllocTRES=cpu=1,mem={node_mb}M" in show
    node = s.run("scontrol show node dgx-01")
    assert f"AllocMem={node_mb}" in node and f"AllocTRES=cpu=1,mem={node_mb}M" in node
    assert "DefMemPerNode=UNLIMITED" in s.run("scontrol show partition defq")


def test_the_whole_node_allocation_follows_the_nodes_configured_memory():
    s = cluster(NO_DEFAULT)
    s.ctx.state.slurm.nodes["dgx-02"].real_memory = 1500000
    s.run("sbatch -w dgx-02 --wrap='sleep 600'")
    assert "mem=1500000M" in sacct(s, 1001, "AllocTRES")[0]
    assert "AllocMem=1500000" in s.run("scontrol show node dgx-02")


def test_a_node_whose_memory_is_all_allocated_takes_no_second_job():
    s = cluster(NO_DEFAULT)
    s.run("sbatch -w dgx-01 --gres=gpu:1 --wrap='sleep 600'")
    s.run("sbatch -w dgx-01 --gres=gpu:1 --wrap='sleep 600'")
    assert sacct(s, 1002, "State,AllocTRES") == ["PENDING", ""]
    assert "(Resources)" in s.run("squeue -j 1002")
    s.run("scancel 1001")
    assert sacct(s, 1002, "State")[0] == "RUNNING"  # the memory came back with the first job


def test_a_two_node_job_and_its_batch_step_report_their_own_shares():
    s = cluster(NO_DEFAULT)
    s.run("sbatch -N2 -w dgx-0[2-3] --exclusive --wrap='sleep 600'")
    one, both = real_memory(s, "dgx-02"), real_memory(s, "dgx-02") + real_memory(s, "dgx-03")
    rows = steps(s, 1001, "AllocTRES,NodeList")
    assert rows["1001"] == [f"billing=448,cpu=448,mem={both}M,node=2", "dgx-[02-03]"]
    assert rows["1001.batch"] == [f"cpu=224,mem={one}M,node=1", "dgx-02"]  # the first node only, no billing
    assert rows["1001.extern"][0] == rows["1001"][0]
    assert f"mem={both}M" in s.run("scontrol show job 1001")
    for node in ("dgx-02", "dgx-03"):
        assert f"AllocMem={one}" in s.run(f"scontrol show node {node}")


def test_jobs_that_name_their_memory_are_allocated_exactly_that():
    s = cluster()
    s.run("sbatch -w dgx-01 --gres=gpu:1 --mem=64G --wrap='sleep 600'")
    s.run("sbatch -w dgx-01 --gres=gpu:1 --mem=1536 --wrap='sleep 600'")
    assert sacct(s, 1001, "State,ReqMem") == ["RUNNING", "64G"]
    # ReqMem and the TRES columns print the same figure in the same unit
    assert sacct(s, 1002, "ReqMem,AllocTRES") == ["1.50G", "billing=1,cpu=1,gres/gpu=1,mem=1.50G,node=1"]
    assert f"AllocMem={64 * 1024 + 1536}" in s.run("scontrol show node dgx-01")
    assert "MinMemoryNode=64G" in s.run("scontrol show job 1001")


def test_an_idle_node_has_no_memory_allocated():
    node = cluster().run("scontrol show node dgx-04")
    assert "AllocMem=0" in node and re.search(r"AllocTRES=\s*$", node, re.M)


@pytest.mark.parametrize(
    ("line", "submit", "req_mem", "shown"),
    [
        ("DefMemPerNode=65536", "sbatch -w dgx-01 -c 8 --wrap='sleep 600'", "64G", "MinMemoryNode=64G"),
        ("DefMemPerGPU=16384", "sbatch -w dgx-01 --gres=gpu:2 --wrap='sleep 600'", "32G", "MinMemoryGPU=16G"),
    ],
)
def test_the_other_defaults_in_slurm_conf_are_honoured(line, submit, req_mem, shown):
    s = cluster(NO_DEFAULT, f"echo '{line}' >> /etc/slurm/slurm.conf")
    s.run(submit)
    assert sacct(s, 1001, "State,ReqMem") == ["RUNNING", req_mem]
    assert shown in s.run("scontrol show job 1001")
    assert line in s.run("scontrol show partition defq")


def test_memory_is_not_allocated_when_the_cluster_does_not_track_it():
    s = cluster(NO_DEFAULT, NOT_TRACKED)
    s.run("sbatch -w dgx-01 --wrap='sleep 600'")
    s.run("sbatch -w dgx-01 --wrap='sleep 600'")
    assert sacct(s, 1002, "State")[0] == "RUNNING"
    assert "mem=0," in sacct(s, 1001, "AllocTRES")[0]
    assert "AllocMem=0" in s.run("scontrol show node dgx-01")


def test_a_later_edit_of_slurm_conf_does_not_rewrite_a_job():
    s = cluster()
    s.run("sbatch -w dgx-01 -c 2 --wrap='sleep 600'")
    s.run("sbatch -w dgx-01 -c 2 --wrap='sleep 30'")
    s.run("sleep 100")
    before = [sacct(s, job, "State,ReqMem,ReqTRES,AllocTRES") for job in (1001, 1002)]
    assert before[0][:2] == ["RUNNING", "16G"] and before[1][:2] == ["COMPLETED", "16G"]
    s.run("sed -i 's/^DefMemPerCPU=.*/DefMemPerCPU=1024/' /etc/slurm/slurm.conf")
    s.run("scontrol reconfigure")
    assert [sacct(s, job, "State,ReqMem,ReqTRES,AllocTRES") for job in (1001, 1002)] == before
    assert f"AllocMem={2 * PER_CPU}" in s.run("scontrol show node dgx-01")
    s.run("sbatch -w dgx-01 -c 2 --wrap='sleep 600'")
    assert sacct(s, 1003, "ReqMem")[0] == "2G"  # a job submitted after the edit follows it


def test_a_suspended_job_keeps_its_memory():
    s = cluster(NO_DEFAULT)
    s.run("sbatch -w dgx-01 --wrap='sleep 600'")
    s.run("sbatch -w dgx-01 --wrap='sleep 600'")
    s.run("scontrol suspend 1001")
    assert sacct(s, 1002, "State")[0] == "PENDING"
    node_mb = real_memory(s, "dgx-01")
    assert f"AllocMem={node_mb}" in s.run("scontrol show node dgx-01")
    s.run("scontrol resume 1001")
    assert f"AllocMem={node_mb}" in s.run("scontrol show node dgx-01")


def test_array_tasks_and_srun_steps_carry_the_allocation():
    s = cluster()
    s.run("sbatch --array=0-1 -c 2 --wrap='srun hostname; sleep 600'")
    rows = s.run("sacct -P -n --format=JobID,ReqMem,AllocTRES").strip().splitlines()
    tasks = [row.split("|") for row in rows if re.fullmatch(r"\d+_\d", row.split("|")[0])]
    assert len(tasks) == 2 and all(task[1] == "16G" and "mem=16G" in task[2] for task in tasks)
    assert "dgx-" in s.run("srun -c 2 hostname")


def lab_states():
    for raw in sc.load_all():
        yield from raw.all_variants()


def test_no_lab_starts_with_more_memory_allocated_than_a_node_has():
    for scenario in lab_states():
        state = scenario.build_state()
        _ensure(state)
        for name, node in state.slurm.nodes.items():
            held = sum(
                _alloc_mem_mb(state, j, name)
                for j in state.slurm.jobs
                if _holds_memory(j) and name in (j.nodelist or [])
            )
            assert held <= int(node.real_memory), f"{scenario.code}: {name} has more memory allocated than it has"


def test_tracking_memory_does_not_stop_a_learners_small_job_in_any_lab():
    """In every lab's starting state, a job without --mem runs wherever it would run if memory were not tracked."""
    differences = []
    for scenario in lab_states():
        outcomes = []
        for tracked in (True, False):
            shell = Shell(scenario.build_state())
            if not tracked:
                shell.run(NOT_TRACKED)
            shell.run("scontrol reconfigure")  # in both, so the only difference is the memory setting
            nodes = sorted(shell.ctx.state.slurm.nodes)
            outcomes.append([node in shell.run(f"srun -w {node} hostname") for node in nodes])
        if outcomes[0] != outcomes[1]:
            differences.append(scenario.code)
    assert differences == []
