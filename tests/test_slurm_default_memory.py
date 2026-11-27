"""Memory a job is allocated when it names none: taken from the cluster's configuration, shown the same everywhere."""

from __future__ import annotations

import re

from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def cluster() -> Shell:
    return Shell(make_cluster())


def real_memory(shell: Shell, node: str) -> int:
    return int(re.search(r"RealMemory=(\d+)", shell.run(f"scontrol show node {node}")).group(1))


def sacct(shell: Shell, job: int, fields: str) -> list[str]:
    return shell.run(f"sacct -j {job} -X -P -n --format={fields}").strip().split("|")


def test_a_job_without_mem_asks_for_nothing_and_is_allocated_the_whole_node():
    s = cluster()
    s.run("sbatch -w dgx-01 --gres=gpu:1 --wrap='sleep 600'")
    node_mb = real_memory(s, "dgx-01")
    req_mem, req_tres, alloc_tres = sacct(s, 1001, "ReqMem,ReqTRES,AllocTRES")
    assert req_mem == "0" and "mem=0," in req_tres
    assert f"mem={node_mb}M" in alloc_tres
    show = s.run("scontrol show job 1001")
    assert "MinMemoryNode=0M" in show and f"AllocTRES=cpu=1,mem={node_mb}M" in show
    node = s.run("scontrol show node dgx-01")
    assert f"AllocMem={node_mb}" in node and re.search(rf"AllocTRES=cpu=1,mem={node_mb}M", node)


def test_the_allocation_follows_the_nodes_configured_memory():
    s = cluster()
    s.ctx.state.slurm.nodes["dgx-02"].real_memory = 1500000
    s.run("sbatch -w dgx-02 --wrap='sleep 600'")
    assert "mem=1500000M" in sacct(s, 1001, "AllocTRES")[0]
    assert "AllocMem=1500000" in s.run("scontrol show node dgx-02")


def test_a_job_over_two_nodes_is_allocated_both_nodes_memory():
    s = cluster()
    s.run("sbatch -N2 -w dgx-0[2-3] --wrap='sleep 600'")
    total = real_memory(s, "dgx-02") + real_memory(s, "dgx-03")
    assert f"mem={total}M" in sacct(s, 1001, "AllocTRES")[0]


def test_a_node_whose_memory_is_all_allocated_takes_no_second_job():
    s = cluster()
    s.run("sbatch -w dgx-01 --gres=gpu:1 --wrap='sleep 600'")
    s.run("sbatch -w dgx-01 --gres=gpu:1 --wrap='sleep 600'")
    assert sacct(s, 1002, "State,AllocTRES") == ["PENDING", ""]
    assert "(Resources)" in s.run("squeue -j 1002")
    s.run("scancel 1001")
    assert sacct(s, 1002, "State")[0] == "RUNNING"  # the memory came back with the first job


def test_jobs_that_name_their_memory_share_a_node():
    s = cluster()
    s.run("sbatch -w dgx-01 --gres=gpu:1 --mem=64G --wrap='sleep 600'")
    s.run("sbatch -w dgx-01 --gres=gpu:1 --mem=32G --wrap='sleep 600'")
    assert sacct(s, 1001, "State,ReqMem")[:2] == ["RUNNING", "64G"]
    assert sacct(s, 1002, "State,ReqMem")[:2] == ["RUNNING", "32G"]
    assert "mem=32G" in sacct(s, 1002, "AllocTRES")[0]
    assert f"AllocMem={96 * 1024}" in s.run("scontrol show node dgx-01")


def test_an_idle_node_has_no_memory_allocated():
    s = cluster()
    node = s.run("scontrol show node dgx-04")
    assert "AllocMem=0" in node and re.search(r"AllocTRES=\s*$", node, re.M)


def test_a_default_in_slurm_conf_becomes_the_request():
    s = cluster()
    s.run("echo 'DefMemPerCPU=4096' >> /etc/slurm/slurm.conf")
    s.run("scontrol reconfigure")
    s.run("sbatch -w dgx-01 -c 8 --wrap='sleep 600'")
    s.run("sbatch -w dgx-01 -c 8 --wrap='sleep 600'")
    assert sacct(s, 1001, "State,ReqMem") == ["RUNNING", "32G"]
    assert sacct(s, 1002, "State")[0] == "RUNNING"  # 32G each, so they share the node
    assert "mem=32G" in sacct(s, 1001, "AllocTRES")[0]
    assert "DefMemPerCPU=4096" in s.run("scontrol show partition defq")
    assert "MinMemoryNode=32G" in s.run("scontrol show job 1001")


def test_without_a_default_the_partition_reports_unlimited():
    assert "DefMemPerNode=UNLIMITED" in cluster().run("scontrol show partition defq")


def test_memory_is_not_allocated_when_the_cluster_does_not_track_it():
    s = cluster()
    s.run("sed -i 's/CR_Core_Memory/CR_Core/' /etc/slurm/slurm.conf")
    s.run("scontrol reconfigure")
    s.run("sbatch -w dgx-01 --wrap='sleep 600'")
    s.run("sbatch -w dgx-01 --wrap='sleep 600'")
    assert sacct(s, 1002, "State")[0] == "RUNNING"
    assert "mem=0," in sacct(s, 1001, "AllocTRES")[0]
    assert "AllocMem=0" in s.run("scontrol show node dgx-01")


def test_no_lab_starts_with_a_node_held_whole_or_over_allocated():
    """A running job in a lab's starting state names its memory, so the learner's own jobs can still share its node."""
    from greenray.commands.slurm.scheduler import _alloc_mem_mb, _ensure
    from greenray.content import scenarios as sc

    for raw in sc.load_all():
        for scenario in raw.all_variants():
            state = scenario.build_state()
            _ensure(state)
            running = [j for j in state.slurm.jobs if j.state == "RUNNING" and j.nodelist]
            unnamed = [j.id for j in running if not j.get("min_memory_mb")]
            assert not unnamed, f"{scenario.code}: running job(s) {unnamed} have no memory request"
            for name, node in state.slurm.nodes.items():
                held = sum(_alloc_mem_mb(state, j, name) for j in running if name in j.nodelist)
                assert held <= int(node.real_memory), f"{scenario.code}: {name} has more memory allocated than it has"
