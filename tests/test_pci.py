"""PCI devices: lspci filters as pciutils does, addresses come from one inventory, and selective nvidia-smi queries need --format."""

from __future__ import annotations

import re

import pytest

from conftest import concrete
from greenray.app.lab import LabSession
from greenray.commands.pci import pci_devices
from greenray.content import scenarios as sc
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

GPU_BUSES = ["18", "2a", "3a", "5d", "9a", "ab", "ba", "db"]
GPU_LINE = "3D controller: NVIDIA Corporation GH100 [H100 SXM5 80GB] (rev a1)"
NEEDS_FORMAT = "Invalid combination of input arguments. Please run 'nvidia-smi -h' for help."
TSH2 = [(i, v) for i, v in enumerate(sc.find("tsh-2").variants)]


def on(node: str = "dgx-01") -> Shell:
    s = Shell(make_cluster())
    s.run(f"ssh {node}")
    return s


def addresses(output: str) -> list[str]:
    return [line.split()[0] for line in output.splitlines() if line and not line.startswith("\t")]


# ---------------------------------------------------------------- lspci -s


@pytest.mark.parametrize(
    "spec",
    [
        "0000:2a:00.0",
        "2a:00.0",
        "2A:00.0",
        "0000:2A:00.0",
        "2a:",
        "2a:*.*",
        "2a:0",
        "2a:00",
        "*:2a:00.0",
        ":2a:00.0",
        "0:2a:0.0",
    ],
)
def test_a_slot_filter_shows_only_the_device_at_that_address(spec):
    assert on().run(f"lspci -s {spec}") == f"2a:00.0 {GPU_LINE}"


def test_parts_of_a_slot_filter_that_are_left_out_match_anything():
    s = on()
    everything = s.run("lspci")
    assert s.run("lspci -s 00.0") == s.run("lspci -s .0") == s.run("lspci -s '*'") == s.run("lspci -s 0") == everything
    assert s.run("lspci -s .1") == "" and s.run("lspci -s 01.0") == ""  # every device here is device 0, function 0


def test_a_slot_filter_that_matches_nothing_prints_nothing():
    s = on()
    assert s.run("lspci -s 99:01.0") == ""
    assert s.run("lspci -s 0001:2a:00.0") == ""  # the right bus in another domain


@pytest.mark.parametrize(
    ("spec", "problem"),
    [
        ("zz:00.0", "Invalid bus number"),
        ("100:00.0", "Invalid bus number"),
        ("2a:20.0", "Invalid slot number"),
        ("2a:00.8", "Invalid function number"),
        ("2a:00.0.0", "Invalid slot/function number"),
        ("g000:2a:00.0", "Invalid domain number"),
        ("0:0:2a:00.0", "Too many fields"),
    ],
)
def test_a_malformed_slot_filter_is_refused_with_lspcis_message(spec, problem):
    assert on().run(f"lspci -s {spec}").strip() == f"lspci: -s: {problem}"


def test_the_filter_value_may_be_glued_to_the_option_or_combined_with_others():
    s = on()
    one = f"2a:00.0 {GPU_LINE}"
    assert s.run("lspci -s2a:00.0") == one
    assert s.run("lspci -Ds 2a:00.0") == "0000:" + one
    assert "option requires an argument" in s.run("lspci -s")


# ---------------------------------------------------------------- lspci output and other options


def test_devices_are_listed_in_address_order_in_lowercase_without_duplicates():
    listed = addresses(on().run("lspci"))
    assert listed == sorted(listed, key=lambda a: int(a[:2], 16))
    assert all(re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", a) for a in listed)
    assert len(listed) == len(set(listed))
    assert [a[:2] for a in listed if a[:2] in GPU_BUSES] == GPU_BUSES


def test_the_filters_and_formats_of_lspci():
    s = on()
    assert addresses(s.run("lspci -d 10de:2330")) == [f"{bus}:00.0" for bus in GPU_BUSES]
    assert len(addresses(s.run("lspci -d 15b3:"))) == 8 and len(addresses(s.run("lspci -d ::0207"))) == 8
    assert s.run("lspci -d 10DE:2330 -s 2a:") == f"2a:00.0 {GPU_LINE}"  # both filters have to match
    assert s.run("lspci -n -s 2a:00.0") == "2a:00.0 0302: 10de:2330 (rev a1)"
    assert s.run("lspci -nn -s 2a:00.0") == (
        "2a:00.0 3D controller [0302]: NVIDIA Corporation GH100 [H100 SXM5 80GB] [10de:2330] (rev a1)"
    )
    assert s.run("lspci -k -s 2a:00.0").splitlines()[1:] == [
        "\tKernel driver in use: nvidia",
        "\tKernel modules: nvidia",
    ]
    assert "NUMA node 0" in s.run("lspci -v -s 2a:00.0") and "NUMA node 1" in s.run("lspci -v -s db:00.0")
    assert "Kernel driver in use: nvidia" in s.run("lspci -v -s 2a:00.0")  # -v includes what -k shows
    assert "At least two fields must be given" in s.run("lspci -d 10de")
    assert "Invalid vendor ID" in s.run("lspci -d xyz:")
    assert "invalid option -- 'Z'" in s.run("lspci -Z")
    assert s.run("lspci | grep -i nvidia | wc -l").strip() == "12"  # eight GPUs and four NVSwitches, as before


def test_a_gpu_that_fell_off_the_bus_is_gone_from_lspci_but_its_switch_port_stays():
    s = on()
    s.ctx.node.gpus[1].fallen_off_bus = True
    assert s.run("lspci -s 2a:00.0") == ""
    assert "PCI bridge" in s.run("lspci -s 29:00.0")


# ---------------------------------------------------------------- one inventory for every tool


def test_infiniband_ports_have_pci_addresses_and_every_tool_agrees_on_them():
    s = on()
    listed = [line.split()[0] for line in s.run("lspci -d 15b3:1021").splitlines()]
    assert len(listed) == 8 and not any("mlx5" in a for a in listed)
    verbose = s.run("ibdev2netdev -v").splitlines()
    for i, address in enumerate(listed):
        bus = int(address[:2], 16)
        assert verbose[i].startswith(f"0000:{address} mlx5_{i} ")
        assert f"==> ibp{bus}s0 " in verbose[i]  # the interface is named after its bus, as the kernel does it
    kernel = s.run("dmesg | grep mlx5_core")
    assert [m for m in re.findall(r"mlx5_core 0000:(\S+):", kernel)] == listed
    s.run("mst start")
    status = s.run("mst status -v")
    assert all(address in status for address in listed)
    assert f"ibp{int(listed[0][:2], 16)}s0" in s.run("ip -br a")


@pytest.mark.parametrize(
    "scenario", [v for raw in sc.load_all() for v in raw.all_variants()], ids=lambda s: f"{s.code}-{s.variant_index}"
)
def test_no_node_in_any_lab_has_two_devices_at_one_address(scenario):
    state = scenario.build_state()
    for name, node in state.nodes.items():
        found = [d.full_address for d in pci_devices(node)]
        assert len(found) == len(set(found)), f"{name}: {sorted(a for a in found if found.count(a) > 1)}"


def test_gpu_identity_is_what_it_was():
    s = on("dgx-04")
    rows = s.run("nvidia-smi --query-gpu=index,pci.bus_id,uuid,serial --format=csv,noheader").splitlines()
    assert rows[1] == "1, 00000000:2A:00.0, GPU-7f3a0401-1c2e-4b5a-9d0e-0401a1b2c3d4, 1652304001417"
    assert [row.split(", ")[1] for row in rows] == [f"00000000:{bus.upper()}:00.0" for bus in GPU_BUSES]


# ---------------------------------------------------------------- nvidia-smi selective queries


@pytest.mark.parametrize(
    "command",
    [
        "nvidia-smi --query-gpu=index,pci.bus_id",
        "nvidia-smi -i 0000:2a:00.0 --query-gpu=index,pci.bus_id",
        "nvidia-smi --query-gpu=index --format=noheader",
        "nvidia-smi --query-gpu=index --format=json",
        "nvidia-smi --query-gpu=index --format=csv,pretty",
        "nvidia-smi --query-compute-apps=pid,used_memory",
    ],
)
def test_a_selective_query_without_format_csv_is_refused(command):
    out = on().run(command)
    assert out.strip() == NEEDS_FORMAT


@pytest.mark.parametrize(
    ("fmt", "first"),
    [
        ("csv", "index, pci.bus_id, memory.total [MiB]"),
        ("csv,noheader", "0, 00000000:18:00.0, 81559 MiB"),
        ("csv,nounits", "index, pci.bus_id, memory.total"),
        ("csv,noheader,nounits", "0, 00000000:18:00.0, 81559"),
        ("noheader,csv", "0, 00000000:18:00.0, 81559 MiB"),
    ],
)
def test_the_csv_formats_still_work(fmt, first):
    out = on().run(f"nvidia-smi --query-gpu=index,pci.bus_id,memory.total --format={fmt}")
    assert out.splitlines()[0] == first


def test_the_gpu_selector_takes_an_index_a_uuid_or_a_bus_id_in_either_case():
    s = on("dgx-04")
    want = "1, 00000000:2A:00.0"
    query = "--query-gpu=index,pci.bus_id --format=csv,noheader"
    for target in ("1", "0000:2a:00.0", "00000000:2A:00.0", "2a:00.0", "GPU-7f3a0401-1c2e-4b5a-9d0e-0401a1b2c3d4"):
        assert s.run(f"nvidia-smi -i {target} {query}") == want
    assert 'Field "nope" is not a valid field to query.' in s.run("nvidia-smi --query-gpu=nope --format=csv")


# ---------------------------------------------------------------- tsh-2, every variant


@pytest.mark.parametrize(("index", "v"), TSH2, ids=[v["node"] for _, v in TSH2])
def test_tsh2_the_alert_address_resolves_to_one_gpu_with_both_tools(index, v):
    s = LabSession(concrete("tsh-which-gpu-is-which", index)).shell
    s.run(f"ssh {v['node']}")
    assert s.run(f"lspci -s 0000:{v['busl']}:00.0") == f"{v['busl']}:00.0 {GPU_LINE}"
    # the port the kernel log names as the reporter is the switch port above that GPU
    assert s.run(f"lspci -s {v['up']}:00.0").startswith(f"{v['up']}:00.0 PCI bridge")
    assert f"pcieport 0000:{v['up']}:00.0" in s.run("dmesg")
    query = f"nvidia-smi -i 0000:{v['busl']}:00.0 --query-gpu=index,pci.bus_id,uuid,serial"
    assert s.run(query).strip() == NEEDS_FORMAT
    assert s.run(query + " --format=csv,noheader").strip() == (
        f"{v['gpu']}, 00000000:{v['bus']}:00.0, {v['uuid']}, {v['serial']}"
    )


@pytest.mark.parametrize(("index", "v"), TSH2, ids=[v["node"] for _, v in TSH2])
def test_tsh2_is_still_graded_the_same(index, v):
    session = LabSession(concrete("tsh-which-gpu-is-which", index))
    shell = session.shell
    shell.run(f"ssh {v['node']} 'lspci -s {v['busl']}:00.0'")
    assert not next(ok for desc, ok, _ in session.grade() if "nvidia-smi on" in desc)  # lspci alone is not the evidence
    shell.run(f"ssh {v['node']} 'nvidia-smi --query-gpu=index,pci.bus_id,uuid,serial'")
    assert not next(ok for desc, ok, _ in session.grade() if "nvidia-smi on" in desc)  # a refused query showed nothing
    shell.run(f"ssh {v['node']} 'nvidia-smi --query-gpu=index,pci.bus_id,uuid,serial --format=csv'")
    shell.run(f"echo 'GPU index {v['gpu']} uuid {v['uuid']} serial {v['serial']}' > /root/gpu-report-{v['node']}.txt")
    assert [desc for desc, ok, _ in session.grade() if not ok] == []
