"""InfiniBand / RDMA diagnostics, IPoIB netdevs and NCCL performance tests.

Port data comes from ``node.ib`` entries (see state.make_ib_port):
state, phys_state, rate_gbps, expected_rate_gbps, symbol_errors, link_downed,
rcv_errors, cable_ok, netdev, lid, fw. Optional per-port ``width`` ("4X"/"2X"/"1X").

Commands: ibstat, ibstatus, ibv_devinfo, iblinkinfo, ibdiagnet, perfquery, sminfo,
mst, mlxlink, ip, all_reduce_perf (and the other nccl-tests binaries), mpirun.

NCCL settings are read like real NCCL does: /etc/nccl.conf on each host, then
environment variables (which win). With ``mpirun`` only the launching host's
ranks inherit your exported variables; remote ranks get only what you pass
with ``-x``.
"""

from __future__ import annotations

import re

from ..core.errors import CommandError
from ..core.state import HEAD
from . import HOOKS_REBOOT, HOOKS_SERVICE, command
from .system import known_dirs

STATE_NUM = {"Down": 1, "Initializing": 2, "Armed": 3, "Active": 4}
PHYS_NUM = {
    "Sleep": 1,
    "Polling": 2,
    "Disabled": 3,
    "PortConfigurationTraining": 4,
    "LinkUp": 5,
    "LinkErrorRecovery": 6,
}
LANE_NAME = {100: "NDR", 50: "HDR", 25: "EDR", 14: "FDR", 10: "QDR", 2.5: "SDR"}
LANE_SIGNAL = {100: "106.25", 50: "53.125", 25: "25.78125", 14: "14.0625", 10: "10.0", 2.5: "2.5"}
MGMT_IF = "enp83s0f0np0"  # in-band Ethernet on the DGX (BCM internalnet)
HEAD_IF = "eno1"
SM_GUID = "0xfc6a1c0300a1b2c0"


NO_GDR_FACTOR = 0.52  # inter-node busbw fraction left when NCCL must bounce through host memory

# ---------------------------------------------------------------- helpers


def norm_module(name: str) -> str:
    """modprobe/lsmod treat '-' and '_' in module names as the same character."""
    return name.strip().replace("-", "_")


def module_loaded(node, name: str) -> bool:
    want = norm_module(name)
    return any(norm_module(m) == want for m in node.get("modules", []))


def peermem_loaded(node) -> bool:
    """GPUDirect RDMA for NCCL over IB needs nvidia_peermem (the legacy nv_peer_mem also works)."""
    return module_loaded(node, "nvidia_peermem") or module_loaded(node, "nv_peer_mem")


def modules_persisted(node) -> set[str]:
    """Modules that systemd-modules-load would load at boot (/etc/modules-load.d/*.conf, /etc/modules)."""
    out = set()
    for path, text in node.files.items():
        if (path.startswith("/etc/modules-load.d/") and path.endswith(".conf")) or path == "/etc/modules":
            for line in str(text).splitlines():
                line = line.split("#", 1)[0].strip()
                if line and not line.startswith(";"):
                    out.add(norm_module(line.split()[0]))
    return out


def node_no(host: str) -> int:
    m = re.search(r"(\d+)$", host)
    return int(m.group(1)) if m and host != HEAD else 0


def port_guid(host: str, idx: int) -> str:
    return f"a088c20300{node_no(host):02x}{idx:02x}f0"


def _fmt_guid(g: str) -> str:
    return ":".join(g[i : i + 4] for i in range(0, 16, 4))


def _width(p) -> int:
    return int(str(p.get("width", "4X")).rstrip("Xx") or 4)


def _active(p) -> bool:
    return p.state == "Active" and p.phys_state == "LinkUp"


def _rate(p) -> int:
    """Rate as reported by ibstat: 10 when the link is not up."""
    return p.rate_gbps if _active(p) else 10


def _rate_label(p) -> str:
    if not _active(p):
        return "10 Gb/sec (4X SDR)"
    w = _width(p)
    lane = p.rate_gbps / w
    lane = int(lane) if float(lane).is_integer() else lane
    return f"{p.rate_gbps} Gb/sec ({w}X {LANE_NAME.get(lane, 'NDR')})"


def _pci(p) -> str:
    m = re.search(r"ibp(\d+)s0", p.netdev)
    return f"{int(m.group(1)):02x}:00.0" if m else "18:00.0"


def _ib_ready(ctx):
    n = ctx.node
    if not n.ib:
        raise CommandError("ibpanic: [1201] main: no IB devices found")
    if n.services.get("openibd", "active") != "active" or "mlx5_ib" not in n.modules:
        raise CommandError(f"ibpanic: [1201] main: stat of IB device '{n.ib[0].ca}' failed: No such file or directory")


def _find_ca(ctx, name):
    for p in ctx.node.ib:
        if p.ca == name:
            return p
    raise CommandError(f"ibpanic: [1201] main: stat of IB device '{name}' failed: No such file or directory")


def _all_ports(state):
    for host, node in state.nodes.items():
        for i, p in enumerate(node.ib):
            yield host, node, i, p


def _link_up(p):
    """Bring a port up if physically possible (cable OK and not admin-disabled)."""
    if p.phys_state == "Disabled":
        p.state = "Down"
        return False
    if p.get("cable_ok", True):
        p.state, p.phys_state = "Active", "LinkUp"
        if p.rate_gbps < 50 or not p.rate_gbps:
            p.rate_gbps = p.expected_rate_gbps
        return True
    p.state, p.phys_state = "Down", "Polling"
    return False


# ---------------------------------------------------------------- ibstat / ibstatus / ibv_devinfo


def _ibstat_port(host, i, p):
    down = not _active(p)
    return [
        f"\tPort {p.port}:",
        f"\t\tState: {p.state}",
        f"\t\tPhysical state: {p.phys_state}",
        f"\t\tRate: {_rate(p)}",
        f"\t\tBase lid: {65535 if down else p.lid}",
        "\t\tLMC: 0",
        f"\t\tSM lid: {0 if down else p.sm_lid}",
        "\t\tCapability mask: 0xa751e848",
        f"\t\tPort GUID: 0x{port_guid(host, i)}",
        f"\t\tLink layer: {p.link_layer}",
    ]


@command("ibstat", help="ibstat [-l|-s|-p] [ca] [port] — HCA and port state, rate, LID")
def ibstat(ctx, args):
    _ib_ready(ctx)
    flags = {a for a in args if a.startswith("-")}
    pos = [a for a in args if not a.startswith("-")]
    ports = list(enumerate(ctx.node.ib))
    if "-l" in flags or "--list_of_cas" in flags:
        return "".join(p.ca + "\n" for _, p in ports)
    if "-p" in flags or "--port_list" in flags:
        return "".join(f"0x{port_guid(ctx.host, i)}\n" for i, _ in ports)
    if pos:
        p = _find_ca(ctx, pos[0])
        ports = [(ctx.node.ib.index(p), p)]
    out = []
    for i, p in ports:
        g = port_guid(ctx.host, i)
        out += [
            f"CA '{p.ca}'",
            "\tCA type: MT4129",
            "\tNumber of ports: 1",
            f"\tFirmware version: {p.fw}",
            "\tHardware version: 0",
            f"\tNode GUID: 0x{g}",
            f"\tSystem image GUID: 0x{g}",
        ]
        if "-s" not in flags:
            out += _ibstat_port(ctx.host, i, p)
    return "\n".join(out) + "\n"


@command("ibstatus", help="ibstatus [ca[:port]] — compact port state / rate")
def ibstatus(ctx, args):
    _ib_ready(ctx)
    want = [a.split(":")[0] for a in args if not a.startswith("-")]
    out = []
    for i, p in enumerate(ctx.node.ib):
        if want and p.ca not in want:
            continue
        g = port_guid(ctx.host, i)
        st = "ACTIVE" if p.state == "Active" else p.state.upper()
        down = not _active(p)
        out += [
            f"Infiniband device '{p.ca}' port {p.port} status:",
            f"\tdefault gid:\t fe80:0000:0000:0000:{_fmt_guid(g)}",
            f"\tbase lid:\t 0x{0xFFFF if down else p.lid:x}",
            f"\tsm lid:\t\t 0x{0 if down else p.sm_lid:x}",
            f"\tstate:\t\t {STATE_NUM.get(p.state, 1)}: {st}",
            f"\tphys state:\t {PHYS_NUM.get(p.phys_state, 2)}: {p.phys_state}",
            f"\trate:\t\t {_rate_label(p)}",
            f"\tlink_layer:\t {p.link_layer}",
            "",
        ]
    if want and not out:
        raise CommandError(f"ibstatus: device {want[0]} not found")
    return "\n".join(out) + "\n"


@command("ibv_devinfo", help="ibv_devinfo [-l] [-d ca] [-v] — verbs view of HCAs")
def ibv_devinfo(ctx, args):
    _ib_ready(ctx)
    if "-l" in args or "--list" in args:
        return f"{len(ctx.node.ib)} HCAs found:\n" + "".join(f"\t{p.ca}\n" for p in ctx.node.ib)
    dev = None
    for i, a in enumerate(args):
        if a in ("-d", "--ib-dev") and i + 1 < len(args):
            dev = args[i + 1]
    out = []
    for i, p in enumerate(ctx.node.ib):
        if dev and p.ca != dev:
            continue
        g = _fmt_guid(port_guid(ctx.host, i))
        st = {
            "Active": "PORT_ACTIVE (4)",
            "Down": "PORT_DOWN (1)",
            "Initializing": "PORT_INIT (2)",
            "Armed": "PORT_ARMED (3)",
        }.get(p.state, "PORT_DOWN (1)")
        out += [
            f"hca_id:\t{p.ca}",
            "\ttransport:\t\t\tInfiniBand (0)",
            f"\tfw_ver:\t\t\t\t{p.fw}",
            f"\tnode_guid:\t\t\t{g}",
            f"\tsys_image_guid:\t\t\t{g}",
            "\tvendor_id:\t\t\t0x02c9",
            "\tvendor_part_id:\t\t\t4129",
            "\thw_ver:\t\t\t\t0x0",
            "\tboard_id:\t\t\tMT_0000000838",
            "\tphys_port_cnt:\t\t\t1",
            f"\t\tport:\t{p.port}",
            f"\t\t\tstate:\t\t\t{st}",
            "\t\t\tmax_mtu:\t\t4096 (5)",
            f"\t\t\tactive_mtu:\t\t{'4096 (5)' if _active(p) else '4096 (5)'}",
            f"\t\t\tsm_lid:\t\t\t{p.sm_lid if _active(p) else 0}",
            f"\t\t\tport_lid:\t\t{p.lid if _active(p) else 65535}",
            "\t\t\tport_lmc:\t\t0x00",
            f"\t\t\tlink_layer:\t\t{p.link_layer}",
            "",
        ]
    if dev and not out:
        raise CommandError(f"IB device '{dev}' wasn't found")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- fabric-wide tools


def _rail_switches(state):
    rails = max((len(n.ib) for n in state.nodes.values()), default=0)
    return [(f"ib-rail-{r}", 200 + r, f"b83fd20300{r:02x}e0a0") for r in range(rails)]


@command("iblinkinfo", help="iblinkinfo — every switch port and the link behind it")
def iblinkinfo(ctx, args):
    _ib_ready(ctx)
    out = []
    for r, (sw, lid, guid) in enumerate(_rail_switches(ctx.state)):
        out.append(f"Switch: 0x{guid} {sw} (MQM9700 NDR 64-port):")
        for host, node, i, p in _all_ports(ctx.state):
            if i != r:
                continue
            swport = node_no(host) if host != HEAD else 64
            if _active(p) and node.up:
                w = _width(p)
                lane = p.rate_gbps / w
                lane = int(lane) if float(lane).is_integer() else lane
                out.append(
                    f"      {lid:>4} {swport:>4}[  ] ==( {w}X {LANE_SIGNAL.get(lane, '106.25'):>9} Gbps Active/  LinkUp)==> "
                    f'{p.lid:>6}    1[  ] "{host} {p.ca}" ( )'
                )
            else:
                ps = p.phys_state if node.up else "Polling"
                out.append(
                    f'      {lid:>4} {swport:>4}[  ] ==(                Down/ {ps:>7})==>             [  ] "" ( )'
                )
    return "\n".join(out) + "\n"


def _counters_warn(host, p):
    rows = []
    for field, name in (
        ("symbol_errors", "symbol_error_counter"),
        ("link_downed", "link_downed_counter"),
        ("rcv_errors", "port_rcv_errors"),
    ):
        v = p.get(field, 0)
        if v:
            rows.append(
                f"-W- lid=0x{p.lid:04x} dev=4129 {host}/{p.ca}/P{p.port}\n"
                f"        Performance Monitor counter     : Value\n"
                f"        {name:<32}: {v} (threshold=1)"
            )
    return rows


@command("ibdiagnet", help="ibdiagnet [--pc] — fabric discovery, link, speed and counter checks")
def ibdiagnet(ctx, args):
    _ib_ready(ctx)
    if not any(_active(p) for p in ctx.node.ib):
        raise CommandError("-E- Failed to find an active local port. Use -i <device> -p <port> to select one.")
    state = ctx.state
    switches = _rail_switches(state)
    cas = [(h, n, i, p) for h, n, i, p in _all_ports(state) if n.up and _active(p)]
    expected = [(h, n, i, p) for h, n, i, p in _all_ports(state)]
    warn_links, warn_speed, warn_pc = [], [], []
    for h, n, i, p in expected:
        if not n.up:
            warn_links.append(
                f"-W- {switches[i][0]}/P{node_no(h) or 64} - link partner {h}/{p.ca} not responding (node is down)"
            )
        elif not _active(p):
            warn_links.append(
                f"-W- {switches[i][0]}/P{node_no(h) or 64} - port is {p.state}/{p.phys_state}, "
                f"expected peer {h}/{p.ca}/P{p.port} not discovered"
            )
        else:
            if p.rate_gbps < p.expected_rate_gbps:
                warn_speed.append(
                    f"-W- Link {h}/{p.ca}/P{p.port} <--> {switches[i][0]}/P{node_no(h) or 64} is running at "
                    f"{p.rate_gbps} Gb/s ({_width(p)}X) but both ends support {p.expected_rate_gbps} Gb/s (4X NDR)"
                )
            warn_pc += _counters_warn(h, p)
    pc = "--pc" in args or "-P" in args or True
    lines = [
        "Load Plugins from:",
        "/usr/share/ibdiagnet2.1.1/plugins/",
        "",
        "Running: ibdiagnet " + " ".join(args),
        'Running version:   "IBDIAG-2.13.0.MLNX20240128"',
        "",
        "---------------------------------------------",
        "Discovery",
        f"-I- Discovering ... {len(cas) + len(switches)} nodes ({len(switches)} Switches & {len(cas)} CAs) discovered.",
        "-I- Fabric Discover finished successfully",
        "",
        "---------------------------------------------",
        "Lids Check",
        "-I- Lids Check finished successfully",
        "",
        "---------------------------------------------",
        "Links Check",
        *(warn_links or ["-I- Links Check finished successfully"]),
        "",
        "---------------------------------------------",
        "Subnet Manager",
        "-I- SM Info retrieving finished successfully",
        f"-I- Subnet: 0xfe80000000000000 SM: ufm-01 lid=1 guid={SM_GUID} priority=15 state=MASTER",
        "",
        "---------------------------------------------",
        "Speed / Width checks",
        "-I- Link Speed Check (Compare to supported link speed)",
        *(warn_speed or ["-I- Links Speed Check finished successfully"]),
        "",
    ]
    if pc:
        lines += [
            "---------------------------------------------",
            "Port Counters",
            *(warn_pc or ["-I- Ports counters value Check finished successfully"]),
            "",
        ]
    lines += [
        "---------------------------------------------",
        "Summary",
        "-I- Stage                     Warnings   Errors     Comment",
        "-I- Discovery                 0          0",
        "-I- Lids Check                0          0",
        f"-I- Links Check               {len(warn_links):<10} 0",
        "-I- Subnet Manager            0          0",
        f"-I- Speed / Width checks      {len(warn_speed):<10} 0",
        f"-I- Port Counters             {len(warn_pc):<10} 0",
        "",
        "-I- You can find detailed errors/warnings in: /var/tmp/ibdiagnet2/ibdiagnet2.log",
    ]
    text = "\n".join(lines) + "\n"
    ctx.node.files["/var/tmp/ibdiagnet2/ibdiagnet2.log"] = text
    ctx.state.flags["ibdiagnet_ran"] = True
    return text


@command("perfquery", help="perfquery [-x] [-r|-R] [-C ca] [lid [port]] — port error/traffic counters")
def perfquery(ctx, args):
    _ib_ready(ctx)
    ca, ext, reset_after, reset_only, pos = None, False, False, False, []
    it = iter(args)
    for a in it:
        if a in ("-C", "--Ca"):
            ca = next(it, None)
        elif a in ("-P", "--Port"):
            next(it, None)
        elif a in ("-x", "--extended"):
            ext = True
        elif a in ("-r", "--reset_after_read"):
            reset_after = True
        elif a in ("-R", "--Reset_only"):
            reset_only = True
        elif not a.startswith("-"):
            pos.append(a)
    if pos:
        try:
            lid = int(pos[0], 0)
        except ValueError:
            raise CommandError(f"perfquery: iberror: failed: can't resolve destination port {pos[0]}")
        match = [(h, p) for h, n, i, p in _all_ports(ctx.state) if p.lid == lid and n.up and _active(p)]
        if not match:
            raise CommandError(
                f"ibwarn: [3012] mad_rpc: _do_madrpc failed; dport (Lid {lid})\n"
                f"perfquery: iberror: failed: perfquery query failed"
            )
        host, p = match[0]
    else:
        p = _find_ca(ctx, ca) if ca else ctx.node.ib[0]
        host = ctx.host
        if not _active(p):
            raise CommandError(
                f"ibwarn: [3012] mad_rpc_open_port: can't open UMAD port ({p.ca}:{p.port}) - port is {p.state}\n"
                "perfquery: iberror: failed: Failed to open port"
            )
    if reset_only:
        p.symbol_errors = p.link_downed = p.rcv_errors = 0
        return ""
    n = node_no(host)
    rows = [
        f"# Port {'extended ' if ext else ''}counters: Lid {p.lid} port {p.port} (CapMask: 0x5A00)",
        f"PortSelect:......................{p.port}",
        "CounterSelect:...................0x0000",
        f"SymbolErrorCounter:..............{p.symbol_errors}",
        "LinkErrorRecoveryCounter:........0",
        f"LinkDownedCounter:...............{p.link_downed}",
        f"PortRcvErrors:...................{p.rcv_errors}",
        "PortRcvRemotePhysicalErrors:.....0",
        "PortRcvSwitchRelayErrors:........0",
        "PortXmitDiscards:................0",
        "PortXmitConstraintErrors:........0",
        "PortRcvConstraintErrors:.........0",
        "CounterSelect2:..................0x00",
        "LocalLinkIntegrityErrors:........0",
        "ExcessiveBufferOverrunErrors:....0",
        "QP1Dropped:......................0",
        "VL15Dropped:.....................0",
        f"PortXmitData:....................{(n + 3) * 18446744 + 9123}",
        f"PortRcvData:.....................{(n + 3) * 18446711 + 1777}",
        f"PortXmitPkts:....................{(n + 3) * 51234 + 99}",
        f"PortRcvPkts:.....................{(n + 3) * 51201 + 57}",
        f"PortXmitWait:....................{12 * n}",
    ]
    if reset_after:
        p.symbol_errors = p.link_downed = p.rcv_errors = 0
    return "\n".join(rows) + "\n"


@command("sminfo", help="sminfo — which Subnet Manager is master")
def sminfo(ctx, args):
    _ib_ready(ctx)
    if not any(_active(p) for p in ctx.node.ib):
        raise CommandError("ibwarn: [2311] mad_rpc_open_port: can't open UMAD port\nsminfo: iberror: failed: query")
    return f"sminfo: sm lid 1 sm guid {SM_GUID}, activity count 5243872 priority 15 state 3 SMINFO_MASTER\n"


# ---------------------------------------------------------------- MFT: mst / mlxlink


@command("mst", help="mst start|status [-v] — Mellanox Software Tools devices")
def mst(ctx, args):
    node = ctx.node
    if not node.ib:
        raise CommandError("-E- No Mellanox devices found")
    if args[:1] == ["start"]:
        node["mst_started"] = True
        return (
            "Starting MST (Mellanox Software Tools) driver set\n"
            "Loading MST PCI module - Success\nLoading MST PCI configuration module - Success\n"
            "Create devices\n"
        )
    if args[:1] == ["stop"]:
        node["mst_started"] = False
        return "Stopping MST (Mellanox Software Tools) driver set\nUnloading MST PCI module - Success\n"
    if args[:1] != ["status"]:
        raise CommandError("Usage: mst start|stop|status [-v]")
    if not node.get("mst_started"):
        return (
            "MST modules:\n------------\n    MST PCI module is not loaded\n"
            "    MST PCI configuration module is not loaded\n\n"
            "MST devices:\n------------\nNo MST devices were found nor MST modules were loaded.\n"
            "You may need to run 'mst start' to load MST modules.\n"
        )
    out = [
        "MST modules:",
        "------------",
        "    MST PCI module is not loaded",
        "    MST PCI configuration module loaded",
        "",
    ]
    if "-v" in args:
        out += ["PCI devices:", "------------", f"{'DEVICE_TYPE':<20}{'MST':<30}{'PCI':<10}{'RDMA':<10}{'NET':<20}NUMA"]
        for i, p in enumerate(node.ib):
            out.append(
                f"{'ConnectX7(rev:0)':<20}{'/dev/mst/mt4129_pciconf' + str(i):<30}{_pci(p):<10}"
                f"{p.ca:<10}{'net-' + p.netdev:<20}{0 if i < 4 else 1}"
            )
        for d in node.get("dpus", []):
            out.append(
                f"{'BlueField3(rev:1)':<20}{d.mst:<30}{d.pci:<10}{d.rdma:<10}{'net-' + d.netdev:<20}{d.get('numa', 1)}"
            )
    else:
        out += ["MST devices:", "------------"]
        for i, p in enumerate(node.ib):
            out += [
                f"/dev/mst/mt4129_pciconf{i}         - PCI configuration cycles access.",
                f"                                   domain:bus:dev.fn=0000:{_pci(p)} addr.reg=88 data.reg=92 cr_bar.gw_offset=-1",
                "                                   Chip revision is: 00",
            ]
        for d in node.get("dpus", []):
            out += [
                f"{d.mst}        - PCI configuration cycles access.",
                f"                                   domain:bus:dev.fn=0000:{d.pci} addr.reg=88 data.reg=92 cr_bar.gw_offset=-1",
                "                                   Chip revision is: 01",
            ]
    return "\n".join(out) + "\n"


def _mlx_dev(ctx, dev):
    for i, p in enumerate(ctx.node.ib):
        if dev in (p.ca, f"/dev/mst/mt4129_pciconf{i}", _pci(p), "0000:" + _pci(p)):
            return i, p
    raise CommandError(f'-E- Failed to open device: "{dev}", No such file or directory')


def _kv(rows):
    return [f"{k:<35}: {v}" for k, v in rows]


@command("mlxlink", help="mlxlink -d <mlx5_N> [-m] [-c] [--port_state UP|DN|TG] — link/cable diagnostics")
def mlxlink(ctx, args):
    dev, port_state, show_module, counters = None, None, False, False
    it = iter(args)
    for a in it:
        if a in ("-d", "--device"):
            dev = next(it, None)
        elif a in ("-a", "--port_state"):
            port_state = (next(it, "") or "").upper()
        elif a in ("-m", "--show_module", "--cable"):
            show_module = True
        elif a in ("-c", "--show_counters"):
            counters = True
    if not dev:
        raise CommandError("-E- Please provide a device name: mlxlink -d <device>")
    i, p = _mlx_dev(ctx, dev)
    if port_state:
        if port_state not in ("UP", "DN", "TG"):
            raise CommandError(f"-E- Invalid port state {port_state}, valid values: UP, DN, TG")
        out = ["", "Configuring Port State..."]
        if port_state == "DN":
            p.state, p.phys_state = "Down", "Disabled"
        else:
            if port_state == "TG" or p.phys_state == "Disabled":
                p.link_downed = p.get("link_downed", 0) + (1 if _active(p) else 0)
            if p.phys_state == "Disabled":
                p.phys_state = "Polling"
            _link_up(p)
        return "\n".join(out) + "\n"
    act = _active(p)
    if p.phys_state == "Disabled":
        trouble = (
            "1",
            "PHY FW",
            "Port is administratively disabled (mlxlink --port_state DN). Enable it with --port_state UP",
        )
    elif not p.get("cable_ok", True):
        trouble = (
            "2",
            "PHY FW",
            "No partner detected / no signal on RX lanes. Check that the cable/transceiver is "
            "seated on both ends (HCA and switch), inspect/clean the fiber, then check the switch port",
        )
    elif act and p.rate_gbps < p.expected_rate_gbps:
        trouble = ("0", "N/A", f"Link is up at reduced speed/width ({p.rate_gbps}G). Check cable type and lane errors")
    elif act and p.get("symbol_errors", 0) > 0:
        trouble = ("0", "N/A", "Physical errors detected. Monitor BER; replace the cable if errors keep increasing")
    else:
        trouble = ("0", "N/A", "No issue was observed")
    if act:
        w = _width(p)
        lane = p.rate_gbps / w
        lane = int(lane) if float(lane).is_integer() else lane
        speed, width, fec = f"IB-{LANE_NAME.get(lane, 'NDR')}", f"{w}x", "Standard_RS-FEC - (544,514)"
    else:
        speed = width = fec = "N/A"
    st = "Active" if act else ("Disable" if p.phys_state == "Disabled" else "Polling")
    out = [
        "",
        "Operational Info",
        "----------------",
        *_kv(
            [
                ("State", st),
                ("Physical state", p.phys_state if act else ("N/A" if p.phys_state == "Disabled" else "Polling")),
                ("Speed", speed),
                ("Width", width),
                ("FEC", fec),
                ("Loopback Mode", "No Loopback"),
                ("Auto Negotiation", "ON"),
            ]
        ),
        "",
        "Supported Info",
        "--------------",
        *_kv(
            [
                ("Enabled Link Speed", "0x000000f0 (NDR,HDR,EDR,FDR)"),
                ("Supported Cable Speed", "0x000000f0 (NDR,HDR,EDR,FDR)" if p.get("cable_ok", True) else "N/A"),
            ]
        ),
        "",
        "Troubleshooting Info",
        "--------------------",
        *_kv([("Status Opcode", trouble[0]), ("Group Opcode", trouble[1]), ("Recommendation", trouble[2])]),
        "",
        "Tool Information",
        "----------------",
        *_kv([("Firmware Version", p.fw), ("amBER Version", "3.6"), ("MFT Version", "mft 4.28.0-92")]),
    ]
    if show_module:
        rx = "1.12,1.05,0.98,1.20" if p.get("cable_ok", True) else "-40.00,-40.00,-40.00,-40.00"
        out += [
            "",
            "Module Info",
            "-----------",
            *_kv(
                [
                    ("Identifier", "OSFP"),
                    ("Compliance", "400GAUI-4 C2M"),
                    ("Cable Technology", "850 nm VCSEL"),
                    ("Cable Type", "Optical Module (separated)"),
                    ("Vendor Name", "NVIDIA"),
                    ("Vendor Part Number", "MMA4Z00-NS400"),
                    ("Vendor Serial Number", f"MT2412FT{node_no(ctx.host):02d}{i:02d}7"),
                    ("Module Temperature [C]", "47"),
                    ("Module Voltage [mV]", "3287"),
                    ("RX Power Current [dBm]", rx),
                    ("TX Power Current [dBm]", "1.31,1.29,1.35,1.27"),
                ]
            ),
        ]
    if counters:
        out += [
            "",
            "Physical Counters and BER Info",
            "------------------------------",
            *_kv(
                [
                    ("Time Since Last Clear [Min]", "1432.7"),
                    ("Effective Physical Errors", p.get("symbol_errors", 0)),
                    ("Effective Physical BER", "15E-255" if not p.get("symbol_errors") else "3E-12"),
                    ("Raw Physical Errors Per Lane", "0,0,0,0"),
                    ("Raw Physical BER", "1E-10"),
                    ("Link Down Counter", p.get("link_downed", 0)),
                    ("Link Error Recovery Counter", 0),
                ]
            ),
        ]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- ip (netdevs incl. IPoIB)


def _ipoib_hw(host, i):
    g = port_guid(host, i)
    return "00:00:10:29:fe:80:00:00:00:00:00:00:" + ":".join(g[k : k + 2] for k in range(0, 16, 2))


def netdevs(host, node):
    """[(name, kind, is_up, ipv4/prefix, port-or-None)]"""
    n = node_no(host)
    down = set(node.get("netdev_admin_down", []))
    out = [("lo", "loopback", True, "127.0.0.1/8", None)]
    mg = HEAD_IF if host == HEAD else MGMT_IF
    out.append((mg, "ether", mg not in down, f"{node.ip}/16", None))
    for i, p in enumerate(node.ib):
        ip = f"10.149.{i}.{n if host != HEAD else 254}/16"
        out.append((p.netdev, "infiniband", p.netdev not in down and _active(p), ip, p))
    return out


@command("ip", help="ip link | ip -br a | ip addr [show dev X] | ip link set dev X up|down")
def ip(ctx, args):
    brief = "-br" in args or "-brief" in args
    rest = [a for a in args if not a.startswith("-")]
    obj = rest[0] if rest else "addr"
    devs = netdevs(ctx.host, ctx.node)
    if obj in ("l", "link") and len(rest) >= 2 and rest[1] == "set":
        words = rest[2:]
        if words and words[0] == "dev":
            words = words[1:]
        if len(words) < 2 or words[1] not in ("up", "down"):
            raise CommandError("Usage: ip link set dev <name> up|down")
        name, action = words[0], words[1]
        if name not in [d[0] for d in devs]:
            raise CommandError(f'Cannot find device "{name}"')
        down = ctx.node.setdefault("netdev_admin_down", [])
        if action == "down" and name not in down:
            down.append(name)
        if action == "up" and name in down:
            down.remove(name)
        return ""
    want = None
    if "dev" in rest:
        want = rest[rest.index("dev") + 1] if rest.index("dev") + 1 < len(rest) else None
    elif len(rest) >= 3 and rest[1] == "show":
        want = rest[2]
    out = []
    for idx, (name, kind, up, addr, _p) in enumerate(devs, start=1):
        if want and name != want:
            continue
        admin_up = name not in ctx.node.get("netdev_admin_down", [])
        state = "UNKNOWN" if kind == "loopback" else ("UP" if up else "DOWN")
        state_col = state
        if brief and ctx.colorize:  # ip -br -c style
            state_col = {"UP": "\033[32mUP\033[0m", "DOWN": "\033[31mDOWN\033[0m"}.get(state, state) + " " * (
                14 - len(state)
            )
        if brief:
            if obj in ("l", "link"):
                mac = "00:00:00:00:00:00" if kind == "loopback" else (f"7c:c2:55:1a:00:{node_no(ctx.host):02x}")
                if kind == "infiniband":
                    mac = _ipoib_hw(ctx.host, idx - 3)
                out.append(f"{name:<16} {state_col if ctx.colorize else f'{state:<14}'} {mac}")
            else:
                out.append(
                    f"{name:<16} {state_col if ctx.colorize else f'{state:<14}'} {addr if (up or kind == 'loopback') else ''}".rstrip()
                )
            continue
        flags = (
            ["LOOPBACK", "UP", "LOWER_UP"]
            if kind == "loopback"
            else (["BROADCAST", "MULTICAST"] + (["UP"] if admin_up else []) + (["LOWER_UP"] if up else []))
        )
        if kind != "loopback" and admin_up and not up:
            flags = ["NO-CARRIER"] + flags
        mtu = 65536 if kind == "loopback" else (4092 if kind == "infiniband" else 9000)
        out.append(
            f"{idx}: {name}: <{','.join(flags)}> mtu {mtu} qdisc {'noqueue' if kind == 'loopback' else 'mq'} "
            f"state {state} group default qlen {256 if kind == 'infiniband' else 1000}"
        )
        if kind == "infiniband":
            out.append(
                f"    link/infiniband {_ipoib_hw(ctx.host, idx - 3)} "
                "brd 00:ff:ff:ff:ff:12:40:1b:ff:ff:00:00:00:00:00:00:ff:ff:ff:ff"
            )
        elif kind == "ether":
            out.append(f"    link/ether 7c:c2:55:1a:00:{node_no(ctx.host):02x} brd ff:ff:ff:ff:ff:ff")
        else:
            out.append("    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00")
        if obj in ("a", "addr", "address") and (up or kind == "loopback"):
            out.append(f"    inet {addr} scope {'host' if kind == 'loopback' else 'global'} {name}")
    if want and not out:
        raise CommandError(f'Device "{want}" does not exist.')
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- NCCL


def nccl_conf(node) -> dict:
    """Parse /etc/nccl.conf (KEY=VALUE lines, '#' comments, optional 'export')."""
    out = {}
    for line in node.files.get("/etc/nccl.conf", "").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def match_list(spec: str, names: list[str]) -> list[str]:
    """NCCL list semantics: '^' excludes, '=' exact match, otherwise prefix match; ':port' suffix allowed."""
    spec = (spec or "").strip()
    if not spec:
        return list(names)
    exclude = spec.startswith("^")
    if exclude:
        spec = spec[1:]
    exact = spec.startswith("=")
    if exact:
        spec = spec[1:]
    entries = [e.split(":")[0].strip() for e in spec.split(",") if e.strip()]

    def hit(n):
        return any((n == e) if exact else n.startswith(e) for e in entries)

    return [n for n in names if hit(n) != exclude]


def _socket_ifs(host, node, env):
    devs = [d for d in netdevs(host, node) if d[2] and d[1] != "loopback"]
    spec = env.get("NCCL_SOCKET_IFNAME")
    if spec:
        names = match_list(spec, [d[0] for d in devs])
    else:  # default: prefer ib*, exclude docker*/lo
        names = [d[0] for d in devs if d[0].startswith("ib")] or [d[0] for d in devs if not d[0].startswith("docker")]
    return [d for d in devs if d[0] in names]


def _ib_selection(node, env):
    if env.get("NCCL_IB_DISABLE", "0") not in ("0", ""):
        return [], "disabled"
    names = match_list(env.get("NCCL_IB_HCA", ""), [p.ca for p in node.ib])
    ports = [p for p in node.ib if p.ca in names and _active(p) and p.link_layer == "InfiniBand"]
    return ports, ""


OP_FACTOR = {  # nccl-tests busBw = algBw * factor(n)
    "all_reduce_perf": lambda n: 2 * (n - 1) / n,
    "all_gather_perf": lambda n: (n - 1) / n,
    "reduce_scatter_perf": lambda n: (n - 1) / n,
    "alltoall_perf": lambda n: (n - 1) / n,
    "broadcast_perf": lambda n: 1,
    "reduce_perf": lambda n: 1,
    "sendrecv_perf": lambda n: 1,
}


def _parse_size(s: str) -> int:
    m = re.fullmatch(r"(\d+)([KkMmGg]?)", s)
    if not m:
        raise CommandError(f"Invalid size {s}")
    mult = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[m.group(2).lower()]
    return int(m.group(1)) * mult


def _test_args(args):
    opts = {"b": "32M", "e": "32M", "f": "2", "g": "1", "i": None, "n": "20", "w": "5", "c": "1", "t": "1"}
    long = {
        "--minbytes": "b",
        "--maxbytes": "e",
        "--stepfactor": "f",
        "--ngpus": "g",
        "--iters": "n",
        "--warmup_iters": "w",
        "--check": "c",
        "--nthreads": "t",
        "--stepbytes": "i",
    }
    it = iter(args)
    for a in it:
        key = long.get(a) or (a[1:] if re.fullmatch(r"-[befgintwcdozpaGCm]", a) else None)
        if key is None:
            continue
        opts[key] = next(it, opts.get(key))
    return opts


def _log(host, rank, msg, level="INFO", local=0):
    return f"{host}:{41270 + local}:{41270 + local} [{local}] NCCL {level} {msg}"


def run_nccl_test(ctx, prog, placements, env_by_host, args):
    """placements: [(host, n_ranks, gpus_per_rank)]. Returns output text."""
    state = ctx.state
    opts = _test_args(args)
    gpr = int(opts["g"])
    minb, maxb = _parse_size(opts["b"]), _parse_size(opts["e"])
    factor = float(opts["f"])
    hosts = [h for h, _, _ in placements]
    nranks_total = sum(n * gpr for _, n, _ in placements)
    out, logs = [], []
    record = {
        "prog": prog,
        "hosts": hosts,
        "ngpus": nranks_total,
        "ok": False,
        "busbw": 0.0,
        "transport": None,
        "launcher": ctx.host,
    }

    def fail(msg):
        state.flags["nccl_last"] = record
        state.flags.setdefault("nccl_runs", []).append(dict(record))
        raise CommandError("\n".join(out + logs + [msg]), 1)

    # ---- per-host CUDA checks
    for host, nr, _ in placements:
        node = state.nodes[host]
        pid = 41270
        if not node.gpus or not node.driver_loaded:
            fail(
                f"{host}: Test CUDA failure common.cu:892 'no CUDA-capable device is detected'\n"
                f" .. {host} pid {pid}: Test failure common.cu:842"
            )
        if node.nvswitches and node.services.get("nvidia-fabricmanager") != "active":
            state.flags.setdefault("last_cuda", {})[host] = "FAIL"
            fail(
                f"{host}: Test CUDA failure common.cu:892 'system not yet initialized'\n"
                f" .. {host} pid {pid}: Test failure common.cu:842"
            )
        if nr * gpr > len(node.gpus):
            fail(
                f"{host}: Test CUDA failure common.cu:892 'invalid device ordinal'\n"
                f" .. {host} pid {pid}: Test failure common.cu:842"
            )
        bad = [g for g in node.gpus[: nr * gpr] if g.fallen_off_bus or g.reset_required]
        if bad:
            fail(
                f"{host}: Test CUDA failure common.cu:892 'unknown error'\n .. {host} pid {pid}: Test failure common.cu:842"
            )
        state.flags.setdefault("last_cuda", {})[host] = "PASS"

    # ---- NCCL init: bootstrap + network selection
    level = {h: (env_by_host[h].get("NCCL_DEBUG", "") or "").upper() for h in hosts}
    multi = len(hosts) > 1
    inter_caps = []
    transports = set()
    gdr_by_host = {}
    for rank_host_idx, (host, _nr, _) in enumerate(placements):
        node, env = state.nodes[host], env_by_host[host]
        lv = level[host]
        info = lv in ("INFO", "TRACE")
        warn = lv in ("WARN", "INFO", "TRACE")
        r0 = sum(n for _, n, _ in placements[:rank_host_idx])
        if rank_host_idx == 0 and lv in ("VERSION", "WARN", "INFO", "TRACE"):
            logs.append("NCCL version 2.21.5+cuda12.4")
        ifs = _socket_ifs(host, node, env)
        if not ifs:
            if warn:
                logs.append(_log(host, r0, "Bootstrap : no socket interface found", "WARN"))
            fail(
                f"{host}: Test NCCL failure common.cu:1005 'unhandled system error (run with NCCL_DEBUG=INFO for details)'\n"
                f" .. {host} pid 41270: Test failure common.cu:891"
            )
        iface = ifs[0]
        if info:
            logs.append(_log(host, r0, f"Bootstrap : Using {iface[0]}:{iface[3].split('/')[0]}<0>"))
            logs.append(_log(host, r0, "NET/Plugin: No plugin found (libnccl-net.so)"))
            logs.append(_log(host, r0, "cudaDriverVersion 12040"))
        ports, why = _ib_selection(node, env)
        if ports:
            if info:
                lst = " ".join(f"[{k}]{p.ca}:{p.port}/IB" for k, p in enumerate(ports))
                logs.append(_log(host, r0, f"NET/IB : Using {lst} ; OOB {iface[0]}:{iface[3].split('/')[0]}<0>"))
                logs.append(_log(host, r0, "Using network IB"))
            gdr = peermem_loaded(node)
            gdr_by_host[host] = gdr
            if info and not gdr:
                for k, p in enumerate(ports):
                    logs.append(_log(host, r0, f"NET/IB : GPU Direct RDMA Disabled for HCA {k} '{p.ca}' (no module)"))
            if info and multi:
                tag = "/GDRDMA" if gdr else ""
                per = [n * gpr for _, n, _ in placements]
                first = sum(per[:rank_host_idx])
                last = first + per[rank_host_idx] - 1
                nxt = sum(per[: rank_host_idx + 1]) % nranks_total
                prev_last = (first - 1) % nranks_total
                logs.append(
                    _log(
                        host,
                        r0,
                        f"Channel 00/0 : {last}[{per[rank_host_idx] - 1}] -> {nxt}[0] [send] via NET/IB/0{tag}",
                    )
                )
                logs.append(
                    _log(
                        host,
                        r0,
                        f"Channel 00/0 : {prev_last}[{per[rank_host_idx - 1] - 1}] -> {first}[0] [receive] via NET/IB/0{tag}",
                    )
                )
            cap = sum(min(p.rate_gbps, p.expected_rate_gbps) for p in ports) / (8 * 400)
            # without GPUDirect RDMA every inter-node transfer is staged through host memory
            inter_caps.append(370.0 * min(cap, 1.0) * (1.0 if gdr else NO_GDR_FACTOR))
            transports.add("IB")
        else:
            spec = env.get("NCCL_IB_HCA")
            if info:
                logs.append(_log(host, r0, "NET/IB : No device found."))
            if warn and spec and why != "disabled" and multi:
                logs.append(
                    _log(
                        host,
                        r0,
                        f"NET/IB : NCCL_IB_HCA={spec} did not match any active IB device; "
                        f"falling back to the Socket transport",
                        "WARN",
                    )
                )
            if info:
                logs.append(_log(host, r0, f"NET/Socket : Using [0]{iface[0]}:{iface[3].split('/')[0]}<0>"))
                logs.append(_log(host, r0, "Using network Socket"))
            inter_caps.append(2.6)
            transports.add("Socket")

    # ---- intra-node capability (NVLink / NVSwitch, thermals)
    intra = []
    for host, nr, _ in placements:
        node = state.nodes[host]
        gpus = node.gpus[: nr * gpr]
        local = len(gpus)
        nv = (
            min((node.gpus[0].nvlink_count - len(g.nvlink_down)) / node.gpus[0].nvlink_count for g in gpus)
            if gpus
            else 1
        )
        thermal = (
            0.62
            if any("HW Thermal Slowdown" in g.throttle_reasons or "HW Slowdown" in g.throttle_reasons for g in gpus)
            else 1.0
        )
        base = 480.0 if local >= 8 else 360.0
        intra.append(base * nv * thermal)
        if level[host] in ("INFO", "TRACE"):
            logs.append(
                _log(
                    host,
                    0,
                    f"comm 0x55d1c2a0 rank {sum(n * gpr for h2, n, _ in placements[: hosts.index(host)])} nRanks {nranks_total} nNodes {len(hosts)} localRanks {nr} "
                    f"localRank 0 MNNVL 0",
                )
            )
            logs.append(
                _log(host, 0, f"{'NVLS multicast support is available' if local >= 8 else 'Channel 00/32 : 0 1'}")
            )
    peak = min(intra)
    if multi:
        peak = min(peak, min(inter_caps))
        transport = "Socket" if "Socket" in transports else "IB"
    else:
        transport = "NVLink"
    fac = OP_FACTOR[prog](nranks_total) if nranks_total > 1 else 0
    lat = 60.0 if transport == "Socket" else (22.0 if multi else 9.0)

    # ---- header
    out.append(
        f"# nThread 1 nGpus {gpr} minBytes {minb} maxBytes {maxb} step: {opts['f']}(factor) "
        f"warmup iters: {opts['w']} iters: {opts['n']} agg iters: 1 validation: {opts['c']} graph: 0"
    )
    out.append("#")
    out.append("# Using devices")
    rank = 0
    for host, nr, _ in placements:
        node = state.nodes[host]
        for r in range(nr):
            for g in range(gpr):
                gpu = node.gpus[r * gpr + g]
                out.append(
                    f"#  Rank {rank:>2} Group  0 Pid {41270 + r:>6} on {host:>10} device {r * gpr + g:>2} "
                    f"[0x{gpu.bus_id[9:11].lower()}] {gpu.name}"
                )
                rank += 1
    out.append("#")
    out.extend(logs)
    logs = []
    out.append(
        "#                                                              out-of-place                       in-place          "
    )
    out.append(
        "#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong"
    )
    out.append(
        "#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       "
    )
    sizes, s = [], minb
    while s <= maxb:
        sizes.append(s)
        if factor <= 1:
            break
        s = int(s * factor)
    bus_vals = []
    peak_alg = (peak / fac) if fac else peak  # GB/s
    for k, size in enumerate(sizes):
        rows = []
        for jitter in (1.0, 0.996 - 0.002 * (k % 3)):
            t = lat + size / (peak_alg * 1e3) if peak_alg else lat
            t = t / jitter
            alg = size / t / 1e3
            bus = alg * fac
            rows.append((t, alg, bus))
        bus_vals.append(rows[0][2])
        redop = "sum" if prog in ("all_reduce_perf", "reduce_scatter_perf", "reduce_perf") else "none"
        root = "0" if prog in ("broadcast_perf", "reduce_perf") else "-1"
        cells = "".join(f" {t:>8.2f} {a:>7.2f} {b:>7.2f} {0:>6}" for t, a, b in rows)
        out.append(f" {size:>12} {size // 4:>13}     float {redop:>7} {root:>7}{cells}")
    avg = sum(bus_vals) / len(bus_vals) if bus_vals else 0
    out.append("# Out of bounds values : 0 OK")
    out.append(f"# Avg bus bandwidth    : {avg:.4f} ")
    out.append("#")
    record.update(
        {
            "ok": True,
            "busbw": round(bus_vals[-1], 2) if bus_vals else 0.0,
            "avg_busbw": round(avg, 3),
            "transport": transport,
            "max_bytes": maxb,
            "gdr": dict(gdr_by_host),
        }
    )
    state.flags["nccl_last"] = record
    state.flags.setdefault("nccl_runs", []).append(dict(record))
    return "\n".join(out) + "\n"


def _local_env(ctx, host):
    env = dict(nccl_conf(ctx.state.nodes[host]))
    env.update(ctx.node.env)
    return env


def _nccl_direct(prog):
    def run(ctx, args):
        if not ctx.node.gpus:
            raise CommandError(f"{ctx.host}: Test CUDA failure common.cu:892 'no CUDA-capable device is detected'")
        g = int(_test_args(args)["g"])
        return run_nccl_test(ctx, prog, [(ctx.host, 1, g)], {ctx.host: _local_env(ctx, ctx.host)}, args)

    return run


for _prog in OP_FACTOR:
    command(
        _prog,
        f"./{_prog}",
        f"./build/{_prog}",
        f"/opt/nccl-tests/build/{_prog}",
        help="nccl-tests: -b <min> -e <max> -f <factor> -g <gpus per process>" if _prog == "all_reduce_perf" else "",
    )(_nccl_direct(_prog))


MPI_VAL_OPTS = {
    "-np",
    "-n",
    "-c",
    "-H",
    "--host",
    "-host",
    "--hostfile",
    "-hostfile",
    "--machinefile",
    "-x",
    "-N",
    "--npernode",
    "-npernode",
    "--map-by",
    "-map-by",
    "--bind-to",
    "-bind-to",
    "--rank-by",
    "--report-bindings",
    "-wdir",
    "--wdir",
    "--prefix",
}
MPI_TWO_VAL = {"--mca", "-mca", "-gmca", "--gmca"}


MPI_POLICIES = {
    "--map-by": {"slot", "node", "socket", "numa", "core", "hwthread", "l1cache", "l2cache", "l3cache", "board"},
    "--bind-to": {"none", "core", "socket", "numa", "hwthread", "l1cache", "l2cache", "l3cache", "board"},
    "--rank-by": {"slot", "node", "socket", "numa", "core", "hwthread", "board"},
}


@command(
    "mpirun",
    "mpiexec",
    help="mpirun -np 16 -H dgx-01:8,dgx-02:8 -x NCCL_DEBUG=INFO all_reduce_perf -b 8 -e 8G -f 2 -g 1",
)
def mpirun(ctx, args):
    np_, hosts, xvars, npernode, allow_root, prog_i = None, [], {}, None, False, None
    flags: set[str] = set()
    policy = {"map-by": "slot", "bind-to": "core", "rank-by": "slot"}
    i = 0
    while i < len(args):
        a = args[i]
        if a in MPI_TWO_VAL:
            i += 3
            continue
        if a in ("--allow-run-as-root",):
            allow_root = True
        elif a in ("--report-bindings", "--oversubscribe", "--tag-output", "-v", "--verbose", "--display-map"):
            flags.add(a.lstrip("-"))
        elif a in MPI_POLICIES:
            val = args[i + 1] if i + 1 < len(args) else ""
            if val.split(":")[0] not in MPI_POLICIES[a] and not val.startswith("ppr:"):
                raise CommandError(
                    "--------------------------------------------------------------------------\n"
                    f"The specified {a.lstrip('-')} directive ({val}) is not recognized.\n"
                    f"Please check the mpirun man page for the accepted values.\n"
                    "--------------------------------------------------------------------------"
                )
            policy[a.lstrip("-")] = val.split(":")[0]
            i += 2
            continue
        elif a in ("--prefix", "-wdir", "--wdir"):
            val = args[i + 1] if i + 1 < len(args) else ""
            if not val.startswith("/"):
                raise CommandError(f'mpirun: Error: option "{a}" requires an absolute path, got "{val}"')
            if a == "--prefix" and val.rstrip("/") not in known_dirs(ctx):
                raise CommandError(
                    "--------------------------------------------------------------------------\n"
                    "mpirun was unable to find the executable orted in the --prefix path\n"
                    f"    {val.rstrip('/')}/bin\n"
                    "--------------------------------------------------------------------------"
                )
            if a != "--prefix" and val.rstrip("/") not in known_dirs(ctx):
                raise CommandError(
                    f"mpirun: Could not change to the working directory {val}: No such file or directory"
                )
            i += 2
            continue
        elif a in MPI_VAL_OPTS:
            val = args[i + 1] if i + 1 < len(args) else ""
            if a in ("-np", "-n", "-c"):
                if not val.isdigit():
                    raise CommandError(f'mpirun: Error: option "{a}" did not have enough parameters (1)')
                np_ = int(val)
            elif a in ("-H", "--host", "-host"):
                for h in val.split(","):
                    name, _, slots = h.partition(":")
                    hosts.append((name, int(slots) if slots else 1))
            elif a in ("--hostfile", "-hostfile", "--machinefile"):
                text = ctx.node.files.get(val)
                if text is None:
                    raise CommandError(f"Open MPI was unable to open the hostfile:\n    {val}")
                for line in text.splitlines():
                    line = line.split("#")[0].strip()
                    if line:
                        m = re.search(r"slots=(\d+)", line)
                        hosts.append((line.split()[0], int(m.group(1)) if m else 1))
            elif a == "-x":
                k, eq, v = val.partition("=")
                xvars[k] = v if eq else ctx.node.env.get(k, "")
            elif a in ("-N", "--npernode", "-npernode"):
                npernode = int(val)
            i += 2
            continue
        elif a.startswith("-"):
            pass
        else:
            prog_i = i
            break
        i += 1
    if prog_i is None:
        raise CommandError("mpirun: no executable specified")
    if not allow_root and not (
        ctx.node.env.get("OMPI_ALLOW_RUN_AS_ROOT") == "1" and ctx.node.env.get("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM") == "1"
    ):
        raise CommandError(
            "--------------------------------------------------------------------------\n"
            "mpirun has detected an attempt to run as root.\n\n"
            "Running as root is *strongly* discouraged as any mistake (e.g., in\n"
            "defining TMPDIR) or bug can result in catastrophic damage to the OS\n"
            "file system, leaving your system in an unusable state.\n\n"
            "We strongly suggest that you run mpirun as a non-root user.\n\n"
            "You can override this protection by adding the --allow-run-as-root option\n"
            "to the cmd line or by setting two environment variables in the following way:\n"
            "the variable OMPI_ALLOW_RUN_AS_ROOT=1 to indicate the desire to override this\n"
            "protection, and OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 to confirm the choice and\n"
            "add one more layer of certainty that you want to do so.\n"
            "--------------------------------------------------------------------------"
        )
    prog = args[prog_i].rsplit("/", 1)[-1]
    pargs = args[prog_i + 1 :]
    if not hosts:
        hosts = [(ctx.host, len(ctx.node.gpus) or 1)]
    if npernode:
        hosts = [(h, npernode) for h, _ in hosts]
    for h, _ in hosts:
        if h not in ctx.state.nodes:
            raise CommandError(
                f"ssh: Could not resolve hostname {h}: Name or service not known\n"
                "--------------------------------------------------------------------------\n"
                "ORTE was unable to reliably start one or more daemons."
            )
        nd = ctx.state.nodes[h]
        if not nd.up or nd.power != "on":
            raise CommandError(
                f"ssh: connect to host {h} port 22: No route to host\n"
                "--------------------------------------------------------------------------\n"
                "ORTE was unable to reliably start one or more daemons.\n"
                "This usually is caused by:\n\n* not finding the required libraries and/or binaries on\n"
                "  one or more nodes.",
                1,
            )
    total = sum(s for _, s in hosts)
    np_ = np_ or total
    if np_ > total and "oversubscribe" in flags:
        hosts = [(h, s + (np_ - total if n == 0 else 0)) for n, (h, s) in enumerate(hosts)]
        total = np_
    if np_ > total:
        raise CommandError(
            "--------------------------------------------------------------------------\n"
            f"There are not enough slots available in the system to satisfy the {np_}\n"
            f"slots that were requested by the application:\n\n  {prog}\n\n"
            "Either request fewer slots for your application, or make more slots\n"
            "available for use.\n"
            "--------------------------------------------------------------------------"
        )
    placements, left = [], np_
    for h, s in hosts:
        take = min(s, left)
        if take:
            placements.append((h, take, 0))
        left -= take
    procs = _rank_order(placements, policy)
    if prog == "hostname":
        out = "".join(f"{h}\n" for h in procs)
        return _launch_report(flags, policy, procs) + _tagged(flags, out, procs)
    if prog not in OP_FACTOR:
        raise CommandError(
            "--------------------------------------------------------------------------\n"
            "mpirun was unable to find the specified executable file, and therefore\n"
            f"did not launch the job.\n\nExecutable: {args[prog_i]}\n"
            "--------------------------------------------------------------------------"
        )
    env_by_host = {}
    for h, _, _ in placements:
        env = dict(nccl_conf(ctx.state.nodes[h]))
        if h == ctx.host:
            env.update(ctx.node.env)
        env.update(xvars)
        env_by_host[h] = env
    out = run_nccl_test(ctx, prog, placements, env_by_host, pargs)
    return _launch_report(flags, policy, procs) + _tagged(flags, out, procs)


def _rank_order(placements, policy):
    """The host of each rank, in rank order: --map-by node spreads consecutive ranks across hosts."""
    by_host = [[h] * n for h, n, _ in placements]
    if policy["map-by"] == "node" or policy["rank-by"] == "node":
        order = []
        while any(by_host):
            for slots in by_host:
                if slots:
                    order.append(slots.pop(0))
        return order
    return [h for slots in by_host for h in slots]


def _binding(policy, rank):
    if policy["bind-to"] == "none":
        return "is not bound (or bound to all available processors)"
    if policy["bind-to"] in ("socket", "numa", "board"):
        return f"bound to socket {rank % 2}[core 0-55[hwt 0-1]]: [BB/BB/BB/BB][../../../..]"
    return f"bound to socket {rank % 2}[core {rank}[hwt 0-1]]: [BB/../../..][../../../..]"


def _launch_report(flags, policy, procs):
    """What --display-map and --report-bindings print before the program starts."""
    out = []
    if "display-map" in flags:
        out += [
            f" Data for JOB [31337,1] offset 0 Total slots allocated {len(procs)}",
            "",
            " ========================   JOB MAP   ========================",
            "",
        ]
        for host in dict.fromkeys(procs):
            ranks = [r for r, h in enumerate(procs) if h == host]
            out.append(f" Data for node: {host}\tNum slots: {len(ranks)}\tMax slots: 0\tNum procs: {len(ranks)}")
            out += [
                f" \tProcess OMPI jobid: [31337,1] App: 0 Process rank: {r} Bound: {_binding(policy, r)}" for r in ranks
            ]
            out.append("")
        out.append(" =============================================================")
    if "report-bindings" in flags:
        out += [f"[{h}:{2000 + r}] MCW rank {r} {_binding(policy, r)}" for r, h in enumerate(procs)]
    return "\n".join(out) + "\n" if out else ""


def _tagged(flags, out, procs):
    """--tag-output prefixes every line with the job id and the rank that printed it."""
    if "tag-output" not in flags:
        return out
    lines = out.rstrip("\n").split("\n") if out else []
    return "".join(f"[31337,{i % max(1, len(procs))}]<stdout>:{line}\n" for i, line in enumerate(lines))


# ---------------------------------------------------------------- hooks


def _on_service(ctx, name, action):
    if name == "openibd" and action in ("start", "restart"):
        for p in ctx.node.ib:
            if p.phys_state == "Disabled":  # admin-down does not survive a driver restart
                p.phys_state = "Polling"
            _link_up(p)


def _on_reboot(ctx, node):
    # Modules that are not auto-loaded by any udev rule / package (e.g. nvidia_peermem, nvidia_fs on
    # some installs) come back after a reboot only if they are listed in /etc/modules-load.d/.
    volatile = {norm_module(m) for m in node.get("volatile_modules", [])}
    if volatile:
        keep = modules_persisted(node)
        node.modules = [m for m in node.modules if norm_module(m) not in volatile or norm_module(m) in keep]
        for m in sorted(volatile & keep):
            if not module_loaded(node, m):
                node.modules.append(m)
    for p in node.ib:
        if p.phys_state == "Disabled":
            p.phys_state = "Polling"
        _link_up(p)
    node.pop("netdev_admin_down", None)


HOOKS_SERVICE.append(_on_service)
HOOKS_REBOOT.append(_on_reboot)
