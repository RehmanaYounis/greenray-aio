"""A test workload counts when it is the workload the lab specifies, and node evidence when it is from that node."""

from __future__ import annotations

import pytest

from conftest import concrete, requires_solutions
from greenray.content import scenarios as sc
from greenray.core.shell import Shell

pytestmark = requires_solutions


def solved(code: str, skip: str | None = None, first: tuple[str, ...] = ()):
    s = concrete(code)
    shell = Shell(s.build_state())
    for line in first:
        shell.run(line.format(**s.params) if s.params else line)
    for step in s.solution:
        if skip and step.startswith(skip):
            continue
        lines = step.split("\n")
        feed = iter(lines[1:])
        shell.run(lines[0], heredoc_reader=lambda _p, feed=feed: next(feed, None))
    return s, shell


def open_criteria(s: sc.Scenario, shell: Shell) -> list[str]:
    out = []
    for desc, ok, err in s.grade(shell):
        assert not err, (desc, err)
        if not ok:
            out.append(desc)
    return out


def cpu_only(name: str, namespace: str, node: str | None) -> str:
    selector = f"  nodeSelector:\n    kubernetes.io/hostname: {node}\n" if node else ""
    return (
        f"apiVersion: v1\nkind: Pod\nmetadata:\n  name: {name}\n  namespace: {namespace}\nspec:\n{selector}"
        '  containers:\n  - name: c\n    image: ubuntu:22.04\n    command: ["sleep", "infinity"]\n'
    )


PODS = [
    ("ins-9", "rdma-test", "default", None, "/root/rdma-test.yaml"),
    ("ins-4", "nemo-finetune", "llm", None, None),
    ("tsh-13", "nccl-test", "default", "dgx-03", None),
    ("wkl-2", "cuda-test", "default", "dgx-01", "/root/cuda-test.yaml"),
]


@pytest.mark.parametrize(("code", "name", "namespace", "node", "manifest"), PODS, ids=[p[0] for p in PODS])
def test_a_cpu_only_pod_with_the_test_pods_name_does_not_count(code, name, namespace, node, manifest):
    s, shell = solved(code)
    assert open_criteria(s, shell) == []
    shell.ctx.state.nodes["bcm-head"].files["/root/cpu.yaml"] = cpu_only(name, namespace, node)
    shell.run(f"kubectl delete pod {name} -n {namespace}")
    shell.run("kubectl apply -f /root/cpu.yaml")
    assert "Running" in shell.run(f"kubectl get pod {name} -n {namespace}")
    still_open = open_criteria(s, shell)
    assert len(still_open) == 1 and name in still_open[0], still_open

    if manifest:  # the lab's own manifest, applied again, is the valid way back
        shell.run(f"kubectl delete pod {name} -n {namespace}")
        shell.run(f"kubectl apply -f {manifest}")
        shell.run(f"kubectl get pod {name} -n {namespace}")
        assert open_criteria(s, shell) == []


# ---- tsh-11: the bug report has to come from the node that failed


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_tsh11_a_bug_report_from_another_node_does_not_count(variant):
    s = concrete("tsh-11", variant)
    peer = "dgx-03" if s.params["node"] != "dgx-03" else "dgx-02"
    shell = Shell(s.build_state())
    assert "nvidia-bug-report.log.gz" in shell.run(f"ssh {peer} nvidia-bug-report.sh")
    for step in s.solution:
        if not step.startswith("nvidia-bug-report"):
            shell.run(step)
    still_open = open_criteria(s, shell)
    assert len(still_open) == 1 and "bug report" in still_open[0], still_open


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_tsh11_a_bug_report_through_ssh_to_the_node_counts(variant):
    s = concrete("tsh-11", variant)
    shell = Shell(s.build_state())
    shell.run(f"ssh {s.params['node']} nvidia-bug-report.sh")
    for step in s.solution:
        if not step.startswith("nvidia-bug-report"):
            shell.run(step)
    assert open_criteria(s, shell) == []


# ---- tsh-7, tsh-15: the modules of a DGX node, not of the head node

MODULES = [("tsh-7", "kernel modules are loaded"), ("tsh-15", "loaded kernel modules")]


@pytest.mark.parametrize(("code", "wording"), MODULES, ids=[m[0] for m in MODULES])
def test_lsmod_on_the_head_node_is_not_the_nodes_module_list(code, wording):
    s = concrete(code)

    def met(shell: Shell) -> bool:
        return not any(wording in desc for desc in open_criteria(s, shell))

    shell = Shell(s.build_state())
    assert "Module" in shell.run("lsmod")
    assert not met(shell)

    for line in ("ssh dgx-01 lsmod", "ssh dgx-02 'lsmod | grep nvidia_peermem'"):
        shell = Shell(s.build_state())
        shell.run(line)
        assert met(shell), line


@pytest.mark.parametrize("code", ["ins-9", "ins-4", "tsh-13", "wkl-2", "tsh-11", "tsh-7", "tsh-15"])
def test_reference_solution_still_earns_every_criterion(code):
    for s in sc.find(code).all_variants():
        shell = Shell(s.build_state())
        sc.replay_solution(s, shell)
        assert all(ok for _desc, ok, _err in s.grade(shell)), (s.variant_label, s.grade(shell))
