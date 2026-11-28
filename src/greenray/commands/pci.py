"""PCI devices of a node, from its inventory, and lspci over them.

Every address a tool prints for a GPU, an NVSwitch, a ConnectX port or a DPU comes from
``pci_devices``, so lspci, the InfiniBand tools and the kernel log cannot disagree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.errors import CommandError
from . import command


@dataclass(frozen=True)
class PciDevice:
    bus: int
    slot: int
    function: int
    class_id: int
    class_name: str
    vendor_id: int
    device_id: int
    name: str
    revision: int = 0
    driver: str = ""
    domain: int = 0

    @property
    def address(self) -> str:
        """``bb:dd.f``, lowercase, the way lspci and the kernel write it."""
        return f"{self.bus:02x}:{self.slot:02x}.{self.function}"

    @property
    def full_address(self) -> str:
        return f"{self.domain:04x}:{self.address}"

    @property
    def numa_node(self) -> int:
        return 0 if self.bus < 0x80 else 1  # the first socket's root ports hold the low buses


def parse_address(text: str) -> tuple[int, int, int]:
    """(bus, slot, function) of an address such as ``00000000:2A:00.0`` or ``c1:00.0``."""
    bus, slot, function = re.fullmatch(r"(?:[0-9a-fA-F]+:)?([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", text).groups()
    return int(bus, 16), int(slot, 16), int(function)


def ib_address(port) -> str:
    """``bb:dd.f`` of a ConnectX port. Ports built before the inventory carried an address get it from their name."""
    if port.get("bus_id"):
        return "{:02x}:{:02x}.{}".format(*parse_address(port.bus_id))
    named = re.search(r"ibp(\d+)s(\d+)", port.get("netdev", ""))
    return f"{int(named.group(1)):02x}:{int(named.group(2)):02x}.0" if named else "00:00.0"


def pci_devices(node) -> list[PciDevice]:
    """What is on the node's PCI buses, in the order lspci lists it."""
    found: list[PciDevice] = []
    for gpu in node.get("gpus", []):
        bus, slot, function = parse_address(gpu.bus_id)
        # the PCIe switch port above a GPU stays when the GPU itself drops off the bus
        found.append(
            PciDevice(
                bus - 1,
                0,
                0,
                0x0604,
                "PCI bridge",
                0x1000,
                0xC030,
                "Broadcom / LSI PEX890xx PCIe Gen 5 Switch",
                0xB0,
                "pcieport",
            )
        )
        if not gpu.get("fallen_off_bus"):
            driver = "nvidia" if node.get("driver_loaded", True) else ""
            found.append(
                PciDevice(
                    bus,
                    slot,
                    function,
                    0x0302,
                    "3D controller",
                    0x10DE,
                    0x2330,
                    "NVIDIA Corporation GH100 [H100 SXM5 80GB]",
                    0xA1,
                    driver,
                )
            )
    for i in range(int(node.get("nvswitches", 0) or 0)):
        found.append(
            PciDevice(
                0x05 + i,
                0,
                0,
                0x0680,
                "Bridge",
                0x10DE,
                0x22A3,
                "NVIDIA Corporation Device 22a3",
                0xA1,
                "nvidia-nvswitch",
            )
        )
    for port in node.get("ib", []):
        bus, slot, function = parse_address(ib_address(port))
        found.append(
            PciDevice(
                bus,
                slot,
                function,
                0x0207,
                "Infiniband controller",
                0x15B3,
                0x1021,
                "Mellanox Technologies MT2910 Family [ConnectX-7]",
                0,
                "mlx5_core",
            )
        )
    for dpu in node.get("dpus", []):
        bus, slot, function = parse_address(dpu["pci"])
        name = "Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller"
        found.append(
            PciDevice(bus, slot, function, 0x0200, "Ethernet controller", 0x15B3, 0xA2DC, name, 1, "mlx5_core")
        )
    return sorted(found, key=lambda d: (d.domain, d.bus, d.slot, d.function))


def _hex(text: str, limit: int, what: str, flag: str) -> int | None:
    """One part of a filter: None for 'any' (empty or ``*``), else a hexadecimal number up to ``limit``."""
    if text in ("", "*"):
        return None
    if not re.fullmatch(r"[0-9a-fA-F]+", text) or int(text, 16) > limit:
        raise CommandError(f"lspci: {flag}: Invalid {what}", 1)
    return int(text, 16)


def _slot_filter(spec: str):
    """``[[[[<domain>]:]<bus>]:][<device>][.[<func>]]`` as lspci -s reads it."""
    fields = spec.split(":")
    if len(fields) > 3:
        raise CommandError("lspci: -s: Too many fields", 1)
    domain = _hex(fields[0], 0x7FFFFFFF, "domain number", "-s") if len(fields) == 3 else None
    bus = _hex(fields[-2], 0xFF, "bus number", "-s") if len(fields) >= 2 else None
    parts = fields[-1].split(".")
    if len(parts) > 2:
        raise CommandError("lspci: -s: Invalid slot/function number", 1)
    slot = _hex(parts[0], 0x1F, "slot number", "-s")
    function = _hex(parts[1], 7, "function number", "-s") if len(parts) == 2 else None
    wanted = (domain, bus, slot, function)
    return lambda d: all(w is None or w == have for w, have in zip(wanted, (d.domain, d.bus, d.slot, d.function)))


def _id_filter(spec: str):
    """``[<vendor>]:[<device>][:<class>]`` as lspci -d reads it."""
    fields = spec.split(":")
    if len(fields) < 2:
        raise CommandError("lspci: -d: At least two fields must be given", 1)
    if len(fields) > 4:
        raise CommandError("lspci: -d: Too many fields", 1)
    vendor = _hex(fields[0], 0xFFFF, "vendor ID", "-d")
    device = _hex(fields[1], 0xFFFF, "device ID", "-d")
    class_id = _hex(fields[2], 0xFFFF, "class code", "-d") if len(fields) > 2 else None
    wanted = (vendor, device, class_id)
    return lambda d: all(w is None or w == have for w, have in zip(wanted, (d.vendor_id, d.device_id, d.class_id)))


USAGE = "Usage: lspci [-s [[[[<domain>]:]<bus>]:][<slot>][.[<func>]]] [-d [<vendor>]:[<device>][:<class>]] [-n|-nn] [-k] [-v] [-D]"
NOT_SIMULATED = "tmxbqiHFMAGOpPQ"  # real lspci options that change the output into something not modelled here


@command("lspci", help="lspci [-s BUS:DEV.FN] [-d VENDOR:DEVICE] [-nn] [-k] [-v] [-D]: PCI devices of this node")
def lspci(ctx, args):
    filters, numeric, show_driver, verbose, show_domain = [], 0, False, False, False
    rest = list(args)
    if "--help" in rest or "-h" in rest:
        return USAGE + "\n"
    while rest:
        arg = rest.pop(0)
        if not arg.startswith("-") or arg == "-":
            raise CommandError(USAGE, 1)
        letters = arg[1:]
        while letters:
            letter, letters = letters[0], letters[1:]
            if letter in "sd":
                value = letters or (rest.pop(0) if rest else None)
                if value is None:
                    raise CommandError(f"lspci: option requires an argument -- '{letter}'\n{USAGE}", 1)
                filters.append(_slot_filter(value) if letter == "s" else _id_filter(value))
                letters = ""
            elif letter == "n":
                numeric += 1
            elif letter == "k":
                show_driver = True
            elif letter == "v":
                verbose = show_driver = True
            elif letter == "D":
                show_domain = True
            elif letter in NOT_SIMULATED:
                raise CommandError(f"lspci: -{letter} is not simulated here; -s, -d, -n, -nn, -k, -v and -D are", 1)
            else:
                raise CommandError(f"lspci: invalid option -- '{letter}'\n{USAGE}", 1)
    lines = []
    for d in pci_devices(ctx.node):
        if not all(wanted(d) for wanted in filters):
            continue
        ids = f"{d.vendor_id:04x}:{d.device_id:04x}"
        if numeric == 1:
            what = f"{d.class_id:04x}: {ids}"
        elif numeric:
            what = f"{d.class_name} [{d.class_id:04x}]: {d.name} [{ids}]"
        else:
            what = f"{d.class_name}: {d.name}"
        rev = f" (rev {d.revision:02x})" if d.revision else ""
        lines.append(f"{d.full_address if show_domain else d.address} {what}{rev}")
        if verbose:
            lines.append(f"\tFlags: bus master, fast devsel, latency 0, NUMA node {d.numa_node}")
        if show_driver and d.driver:
            lines.append(f"\tKernel driver in use: {d.driver}")
            lines.append(f"\tKernel modules: {d.driver.replace('-', '_')}")
        if verbose:
            lines.append("")
    return "\n".join(lines) + "\n" if lines else ""
