"""Evidence criteria need output, not an attempt.

One row per criterion that used to accept "the command ran": the failed or empty lookups in the row must
earn nothing, even when the rest of the reference solution is carried out without its diagnostic steps.
The full reference solution earning every criterion is covered by ``greenray validate``.
"""

from __future__ import annotations

import re

import pytest

from conftest import concrete, requires_solutions
from greenray.content import scenarios as sc
from greenray.core.shell import Shell

# (lab code, criterion index, solution steps to leave out, lookups that must earn nothing)
ROWS: list[tuple[str, int, str, list[str]]] = [
    ("adm-3", 3, r'^cmsh -c "device; (status|power status)', ['cmsh -c "device; power status -n missing"']),
    (
        "adm-11",
        7,
        r"^kubectl (describe pod|get events)",
        ["kubectl describe pod missing -n inference", "kubectl describe deploy embedder -n inference"],
    ),
    ("adm-12", 7, r"^sshare", ["sshare --help"]),
    (
        "adm-13",
        4,
        r'^journalctl|^cmsh -c "main; events|^(diff|grep|cat|less|head) .*slurm\.conf',
        ["journalctl -u cmd -u", "cat /nonexistent/slurm.conf", "grep nosuchword /etc/slurm/slurm.conf"],
    ),
    (
        "adm-15",
        5,
        r"latestmetricdata|nvidia-smi|ipmitool",
        ['cmsh -c "device; latestmetricdata -n missing"', "ssh {node} nvidia-smi -i 99"],
    ),
    ("ins-4", 5, r"^kubectl (describe pod|get events)", ["kubectl describe pod missing -n llm"]),
    ("ins-5", 5, r"^(sinfo|scontrol show node)", ["scontrol show node missing"]),
    ("ins-6", 0, r"^(df|findmnt|mount|showmount)", ["showmount -e nosuchhost", "ssh dgx-02 findmnt /nosuch"]),
    ("ins-8", 0, r"^mlxconfig .*\bq\b", ["ssh dgx-01 mlxconfig -d nosuch q"]),
    (
        "ins-10",
        5,
        r'^cmsh -c "(main; events|configurationoverlay; (list|use \S+; show))',
        ['cmsh -c "configurationoverlay; use nosuch; show"'],
    ),
    ("tsh-1", 1, r"^(journalctl|dmesg)", ["ssh {node} journalctl -k -u"]),
    ("tsh-4", 4, r"^(journalctl|systemctl status docker)", ["ssh dgx-02 journalctl -u docker -u"]),
    ("tsh-5", 0, r"^nvidia-smi (-q|--query)|^dcgmi", ["ssh {node} nvidia-smi -q -d ROW_REMAPPER -i 99"]),
    (
        "tsh-5",
        2,
        r"^scontrol update nodename=\S+ state=drain",
        ["scontrol update nodename=missing state=drain reason=x"],
    ),
    ("tsh-6", 3, r"^(journalctl|systemctl status)", ["ssh {node} journalctl -u nvidia-fabricmanager -u"]),
    ("tsh-6", 4, r"^(nv-fabricmanager|dpkg|cat /var/log/apt|apt)", ["ssh {node} dpkg -s nosuch-fabricmanager"]),
    ("tsh-7", 3, r"NCCL_DEBUG=INFO", ["export NCCL_DEBUG=INFO", "echo NCCL_DEBUG=INFO"]),
    ("tsh-7", 4, r"lsmod", ["ssh dgx-01 cat /proc/modules/nosuch"]),
    ("tsh-8", 2, r"^(ibstat|ibstatus|ibv_devinfo)", ["ssh {node} ibstat nosuch"]),
    ("tsh-8", 3, r"^mlxlink -d \S+( -m)?$", ["ssh {node} mlxlink -d nosuch"]),
    ("tsh-8", 4, r"^mlxlink -d \S+( -m)?$", ["ssh {node} mlxlink -d nosuch"]),
    ("tsh-9", 0, r"journalctl|unmunge|md5sum", ["journalctl -u", "unmunge --bogus"]),
    ("tsh-9", 1, r"journalctl|unmunge|md5sum", ["journalctl -u", "md5sum /nosuch"]),
    ("tsh-10", 4, r"^nvidia-smi (-q|--query)", ["ssh {node} nvidia-smi -q -d TEMPERATURE -i 99"]),
    ("tsh-10", 5, r"^ipmitool sel", ["ssh {peer} ipmitool sel list", "ssh {node} ipmitool sdr"]),
    ("tsh-11", 0, r"^(dmesg|journalctl)", ["ssh {node} journalctl -k -u"]),
    ("tsh-11", 1, r"^nvidia-bug-report", ["ssh {node} cat nvidia-bug-report.log.gz"]),
    ("tsh-11", 2, r"^(dmesg|journalctl)", ["ssh {node} journalctl -k -u"]),
    ("tsh-12", 1, r"^dcgmi health .*-c", ["ssh dgx-03 dcgmi health -g 99 -c"]),
    ("tsh-13", 5, r"^kubectl logs", ["kubectl logs -n kube-system nvidia-device-plugin-missing"]),
    ("tsh-14", 3, r"NCCL_DEBUG=INFO", ["export NCCL_DEBUG=INFO"]),
    ("tsh-15", 0, r"NCCL_DEBUG=INFO", ["export NCCL_DEBUG=INFO"]),
    ("tsh-15", 1, r"lsmod", ["ssh dgx-01 cat /proc/modules/nosuch"]),
    ("tsh-15", 2, r"^(ibstat|ibstatus|mlxlink|ibv_devinfo)", ["ssh {slow} ibstat nosuch"]),
    ("tsh-16", 0, r"^(nvidia-smi|lsmod|dpkg|journalctl|uname|dmesg)", ["ssh {node} dpkg -s nosuchpkg"]),
    (
        "tsh-16",
        1,
        r'^cmsh -c "(device; (use \S+; show|list)|category; list)',
        ['cmsh -c "device; use missing; show"'],
    ),
    ("tsh-17", 0, r"^(nvidia-smi topo|numactl|lscpu)", ["ssh dgx-03 nvidia-smi -i 99", "ssh dgx-03 numactl"]),
    ("tsh-18", 4, r"dmon", ["ssh dgx-02 dcgmi dmon"]),
    (
        "tsh-18",
        5,
        r"gdscheck",
        ["ssh dgx-01 /usr/local/cuda/gds/tools/gdscheck -p", "ssh dgx-02 cat /nosuch/gdscheck -p"],
    ),
    ("tsh-19", 0, r"^nvsm show", ["ssh {node} nvsm show bogus"]),
    ("wkl-5", 2, r"plugstack", ["cat /etc/slurm/plugstack.conf.d/nosuch.conf"]),
    ("wkl-6", 5, r"describe", ["runai describe job jlab-b", "kubectl describe pod jlab-b"]),
    ("wkl-8", 4, r"^(squeue|scontrol show|sinfo)", ["squeue --bogus", "scontrol show job 999999"]),
    ("wkl-9", 0, r"^kubectl describe pod", ["kubectl describe pod missing"]),
    ("wkl-9", 1, r"^kubectl (get nodes|describe node)", ["kubectl describe node missing"]),
]


def _grade(s: sc.Scenario, shell: Shell, index: int) -> bool:
    _desc, ok, err = s.grade(shell)[index]
    assert not err, err
    return ok


@requires_solutions
@pytest.mark.parametrize(("code", "index", "skip", "lookups"), ROWS, ids=[f"{r[0]}[{r[1]}]" for r in ROWS])
def test_failed_lookups_earn_no_evidence(code, index, skip, lookups):
    s = concrete(code)
    assert s.kinds[index] in ("evidence", "safety")
    shell = Shell(s.build_state())
    for line in lookups:
        shell.run(line.format(**s.params) if s.params else line)
    assert not _grade(s, shell, index)
    left_out = 0
    for step in s.solution:
        lines = step.split("\n")
        if re.search(skip, re.sub(r"^ssh \S+ (?=\S)", "", lines[0])):  # a step may reach the node in one ssh call
            left_out += 1
            continue
        feed = iter(lines[1:])
        shell.run(lines[0], heredoc_reader=lambda _p, feed=feed: next(feed, None))
    assert left_out, "the row leaves out no solution step"
    assert not _grade(s, shell, index)


@requires_solutions
@pytest.mark.parametrize("code", sorted({r[0] for r in ROWS}))
def test_reference_solution_still_earns_every_criterion(code):
    for s in sc.find(code).all_variants():
        shell = Shell(s.build_state())
        sc.replay_solution(s, shell)
        assert all(ok for _desc, ok, _err in s.grade(shell)), (s.variant_label, s.grade(shell))


def test_no_criterion_is_met_by_an_attempt_alone():
    """New checks must use printed()/printed_before(); ran() stays for safety checks that forbid a command."""
    offenders = [
        f"{s.code}: {c.get('desc', c['expr'])}"
        for s in sc.load_all()
        for c in s.checks
        if re.search(r"(?<!not )\bran(_before)?\(", c["expr"])
    ]
    assert offenders == []


# ---- the tool's own error can be the finding


def test_an_error_message_counts_only_where_the_check_asks_for_it():
    s = concrete("tsh-16")
    shell = Shell(s.build_state())
    out = shell.run(f"ssh {s.params['node']} nvidia-smi")
    assert "couldn't communicate with the NVIDIA driver" in out
    assert _grade(s, shell, 0)

    s = concrete("adm-3")
    shell = Shell(s.build_state())
    assert "FAILED" in shell.run('cmsh -c "device; power status -n dgx-02"')
    assert _grade(s, shell, 3)


def test_printed_before_orders_output_against_a_completed_command():
    s = concrete("tsh-8")
    node, rail = s.params["node"], s.params["rail"]
    shell = Shell(s.build_state())
    shell.run(f"reseat-cable {node} mlx5_{rail}")
    shell.run(f"ssh {node} mlxlink -d mlx5_{rail} -m")
    assert _grade(s, shell, 3) and not _grade(s, shell, 4)  # the diagnostics came after the reseat

    shell = Shell(s.build_state())
    shell.run(f"ssh {node} mlxlink -d mlx5_{rail} -m")
    shell.run(f"reseat-cable {node} mlx5_{rail}")
    assert _grade(s, shell, 4)


# ---- wkl-k8s-pod-pending-nodeselector: the node selector is graded as parsed YAML

SELECTOR = "nodeSelector:\n    nvidia.com/gpu.product: NVIDIA-H100-80GB\n"


@pytest.mark.parametrize(
    "selector",
    [
        "nodeSelector:\n    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3\n",
        'nodeSelector:\n    "nvidia.com/gpu.product": NVIDIA-H100-80GB-HBM3\n',
        "nodeSelector:\n    nvidia.com/gpu.product: 'NVIDIA-H100-80GB-HBM3'   # the label GFD sets\n",
        "nodeSelector: {nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3}\n",
        "nodeSelector:\n    kubernetes.io/os: linux\n    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3\n",
    ],
    ids=["plain", "quoted-key", "quoted-value-comment", "flow", "two-labels"],
)
def test_wkl2_every_yaml_spelling_of_the_selector_passes(selector):
    s = concrete("wkl-2")
    shell = Shell(s.build_state())
    files = shell.ctx.state.nodes["bcm-head"].files
    assert SELECTOR in files["/root/cuda-test.yaml"]
    files["/root/cuda-test.yaml"] = files["/root/cuda-test.yaml"].replace(SELECTOR, selector)
    for line in (
        "kubectl describe pod cuda-test",
        "kubectl get nodes -L nvidia.com/gpu.product",
        "kubectl delete pod cuda-test",
        "kubectl apply -f /root/cuda-test.yaml",
    ):
        shell.run(line)
    assert all(ok for _desc, ok, _err in s.grade(shell)), s.grade(shell)


@pytest.mark.parametrize(
    "selector",
    [
        SELECTOR,
        "",
        "# nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3\n  nodeSelector:\n    nvidia.com/gpu.product: NVIDIA-A100\n",
    ],
    ids=["unchanged", "removed", "fix-only-in-a-comment"],
)
def test_wkl2_a_selector_that_does_not_match_fails(selector):
    s = concrete("wkl-2")
    shell = Shell(s.build_state())
    files = shell.ctx.state.nodes["bcm-head"].files
    files["/root/cuda-test.yaml"] = files["/root/cuda-test.yaml"].replace(SELECTOR, selector)
    desc, ok, err = s.grade(shell)[2]
    assert "nodeSelector" in desc and not ok and not err


def test_tsh13_containerd_config_is_matched_whatever_its_spacing():
    s = concrete("tsh-13")
    shell = Shell(s.build_state())
    sc.replay_solution(s, shell)
    index = next(i for i, c in enumerate(s.checks) if "containerd config" in c["desc"])
    assert _grade(s, shell, index)
    for node in ("dgx-03", "dgx-04"):
        files = shell.ctx.state.nodes[node].files
        assert 'default_runtime_name = "nvidia"' in files["/etc/containerd/config.toml"]
        files["/etc/containerd/config.toml"] = files["/etc/containerd/config.toml"].replace(
            'default_runtime_name = "nvidia"', 'default_runtime_name="nvidia"'
        )
    assert _grade(s, shell, index)
