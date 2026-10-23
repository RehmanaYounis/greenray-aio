"""Exact output of the everyday tools on the healthy cluster, compared with files under tests/golden.

A difference here means a tool's layout changed. Compare the new text with the real tool's
output before accepting it: run with UPDATE_GOLDEN=1 to rewrite the files, then review the diff.
Regenerating without that review would silently bless a regression, which is what this suite
exists to catch.
"""

import os
from pathlib import Path

import pytest

from greenray.core.shell import Shell
from greenray.core.state import make_cluster

GOLDEN = Path(__file__).parent / "golden"

# (file name, host, command); host None means the head node.
CASES = [
    ("nvidia-smi", "dgx-01", "nvidia-smi"),
    ("nvidia-smi-i0", "dgx-01", "nvidia-smi -i 0"),
    ("nvidia-smi-L", "dgx-01", "nvidia-smi -L"),
    ("nvidia-smi-q-i0", "dgx-01", "nvidia-smi -q -i 0"),
    ("nvidia-smi-query", "dgx-01", "nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv"),
    (
        "nvidia-smi-mig",
        "dgx-01",
        "nvidia-smi -mig 1 -i 1 && nvidia-smi mig -cgi 1g.10gb,1g.10gb -C -i 1 && nvidia-smi -i 1",
    ),
    ("sinfo", None, "sinfo"),
    ("squeue", None, "squeue"),
    ("sacct", None, "sacct"),
    ("kubectl-get-nodes", None, "kubectl get nodes"),
    ("kubectl-get-pods", None, "kubectl get pods -A"),
    ("cmsh-device-list", None, 'cmsh -c "device list"'),
    ("ibstat", "dgx-01", "ibstat"),
]


def run(host, command):
    shell = Shell(make_cluster())
    if host:
        shell.run(f"ssh {host}")
    return shell.run(command)


@pytest.mark.parametrize(("name", "host", "command"), CASES, ids=[c[0] for c in CASES])
def test_output_matches_golden_file(name, host, command):
    out = run(host, command)
    path = GOLDEN / f"{name}.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(out, encoding="utf-8")
    assert out == path.read_text(encoding="utf-8"), f"{command!r} changed; see the docstring of {__file__}"


def test_nvidia_smi_table_is_rectangular_and_honours_i():
    full = run("dgx-01", "nvidia-smi")
    widths = {len(line) for line in full.splitlines() if line}
    assert widths == {91}, widths
    one = run("dgx-01", "nvidia-smi -i 0")
    assert one.count("NVIDIA H100") == 1 and "00000000:18:00.0" in one
    assert {len(line) for line in one.splitlines() if line} == {91}
    two = run("dgx-01", "nvidia-smi -i 2,5")
    assert "00000000:3A:00.0" in two and "00000000:AB:00.0" in two and two.count("NVIDIA H100") == 2
    assert "No devices were found matching the id 9" in run("dgx-01", "nvidia-smi -i 9")


def test_mig_table_is_rectangular():
    out = run("dgx-01", "nvidia-smi -mig 1 -i 1 && nvidia-smi mig -cgi 1g.10gb -C -i 1 && nvidia-smi -i 1")
    table = out[out.index("| MIG devices:") - 92 :]
    assert {len(line) for line in table.splitlines() if line} == {91}
    assert "|   1    1   0   0 |" in table
