"""Process, network and system-information tools: pgrep, lsof, ss, dig, ethtool, vmstat, hostnamectl, ..."""

from __future__ import annotations

import re

from ..core import audit, clock
from ..core.errors import CommandError
from ..core.state import HEAD
from . import HOOKS_REBOOT, command, dispatch
from .common import columns, opt
from .ib import netdevs, node_no
from .storage import _resolve

# Service -> the TCP ports it listens on, as ss shows them.
LISTENERS = {
    "sshd": [22],
    "slurmctld": [6817],
    "slurmdbd": [6819],
    "slurmd": [6818],
    "kube-apiserver": [6443],
    "kubelet": [10250],
    "nvidia-dcgm": [5555],
    "mysql": [3306],
    "cmd": [8081],
    "nfs-server": [2049, 111],
    "chronyd": [],
    "opensm": [],
}


def _processes(ctx) -> list[tuple[int, str, str]]:
    """(pid, user, command) for every active service and GPU process on the node."""
    rows = [
        (1000 + abs(hash(svc)) % 900, "root", f"/usr/sbin/{svc}")
        for svc, st in ctx.node.services.items()
        if st == "active"
    ]
    rows.append((900, "root", "/usr/sbin/sshd"))
    rows += [(p.pid, p.get("user", "alice"), p.name) for g in ctx.node.gpus for p in g.processes]
    return sorted(rows)


@command("htop", help="htop: the same snapshot as top")
def htop(ctx, args):
    return dispatch(ctx, ["top"])


@command("pgrep", help="pgrep [-f] [-l] [-u USER] PATTERN")
def pgrep(ctx, args):
    user = opt(args, "-u", "--user")
    pats = [a for a in args if not a.startswith("-") and a != user]
    if not pats:
        raise CommandError("pgrep: no matching criteria specified\nTry `pgrep --help' for more information.", 2)
    full = "-f" in args
    out = []
    for pid, owner, cmdline in _processes(ctx):
        name = cmdline.split()[0].rsplit("/", 1)[-1]
        if user and owner != user:
            continue
        if re.search(pats[0], cmdline if full else name):
            out.append(f"{pid} {name}" if "-l" in args else str(pid))
    if not out:
        raise CommandError("", 1)
    return "\n".join(out) + "\n"


@command("killall", help="killall [-SIGNAL] NAME")
def killall(ctx, args):
    names = [a for a in args if not a.startswith("-")]
    if not names:
        raise CommandError("Usage: killall [OPTION]... [--] NAME...", 1)
    out = []
    for name in names:
        matched = [str(pid) for pid, _, cmdline in _processes(ctx) if cmdline.split()[0].rsplit("/", 1)[-1] == name]
        gpu_procs = [str(p.pid) for g in ctx.node.gpus for p in g.processes if p.name.rsplit("/", 1)[-1] == name]
        if not matched:
            out.append(f"{name}: no process found")
            continue
        if gpu_procs:
            dispatch(ctx, ["kill", *gpu_procs])
    if out:
        raise CommandError("\n".join(out), 1)
    return ""


@command("who", help="who: who is logged in")
def who(ctx, args):
    return f"root     pts/0        {clock.stamp(ctx.state, '%Y-%m-%d %H:%M')} (10.141.255.254)\n"


@command("w", help="w: who is logged in and what they are doing")
def w(ctx, args):
    return (
        f" {clock.stamp(ctx.state, '%H:%M:%S')} up {clock.uptime_text(ctx.state)},  1 user,  load average: 0.08, 0.05, 0.01\n"
        "USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT\n"
        f"root     pts/0    10.141.255.254   {clock.stamp(ctx.state, '%H:%M')}    0.00s  0.03s  0.00s w\n"
    )


@command("lsof", help="lsof /dev/nvidia* | lsof -i :PORT: who holds a device or a port")
def lsof(ctx, args):
    port = opt(args, "-i")
    targets = [a for a in args if not a.startswith("-") and a != port]
    rows = [("COMMAND", "PID", "USER", "FD", "TYPE", "DEVICE", "SIZE/OFF", "NODE", "NAME")]
    if port:
        want = int(port.lstrip(":").split(":")[-1]) if port.lstrip(":").split(":")[-1].isdigit() else None
        for svc, ports in LISTENERS.items():
            if ctx.node.services.get(svc) == "active" and (want is None or want in ports):
                for p in ports:
                    if want is None or p == want:
                        rows.append(
                            (
                                svc[:9],
                                str(1000 + abs(hash(svc)) % 900),
                                "root",
                                "3u",
                                "IPv4",
                                "0x0",
                                "0t0",
                                "TCP",
                                f"*:{p} (LISTEN)",
                            )
                        )
    for t in targets:
        for i, g in enumerate(ctx.node.gpus):
            if t in (f"/dev/nvidia{i}", "/dev/nvidia*", "/dev/nvidiactl"):
                for p in g.processes:
                    rows.append(
                        (
                            p.name.rsplit("/", 1)[-1][:9],
                            str(p.pid),
                            p.get("user", "alice"),
                            "12u",
                            "CHR",
                            "195,0",
                            "0t0",
                            "592",
                            f"/dev/nvidia{i}",
                        )
                    )
        if (
            t.startswith("/dev/nvidia")
            and len(rows) == 1
            and t not in [f"/dev/nvidia{i}" for i in range(len(ctx.node.gpus))] + ["/dev/nvidia*", "/dev/nvidiactl"]
        ):
            raise CommandError(f"lsof: status error on {t}: No such file or directory", 1)
    if len(rows) == 1:
        raise CommandError("", 1)
    return columns(rows, gap=1)


@command("fuser", help="fuser -v /dev/nvidia0: processes using a device")
def fuser(ctx, args):
    targets = [a for a in args if not a.startswith("-")]
    out = []
    for t in targets:
        m = re.fullmatch(r"/dev/nvidia(\d+)", t)
        pids = []
        if m and int(m.group(1)) < len(ctx.node.gpus):
            pids = [str(p.pid) for p in ctx.node.gpus[int(m.group(1))].processes]
        elif t == "/dev/nvidiactl":
            pids = [str(p.pid) for g in ctx.node.gpus for p in g.processes]
        if pids:
            out.append(f"{t}: {' '.join(pids)}")
    if not out:
        raise CommandError("", 1)
    return "\n".join(out) + "\n"


@command("vmstat", "mpstat", help="vmstat / mpstat: a one-line snapshot of CPU and memory")
def vmstat(ctx, args):
    busy = sum(g.utilization for g in ctx.node.gpus) / 100 if ctx.node.gpus else 0.0
    us = min(99, int(12 * busy))
    free_kib = ctx.node.real_memory_mb * 1024 - 12 * 1024 * 1024
    return (
        "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----\n"
        " r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st\n"
        f" {1 + int(busy * 8):>2}  0      0 {free_kib:>8} 204800 1048576    0    0     2     8  412  980 {us:>2}  1 {99 - us:>2}  0  0\n"
    )


@command("hostnamectl", help="hostnamectl: host identity")
def hostnamectl(ctx, args):
    chassis = "server" if ctx.host == HEAD else "server"
    return (
        f"   Static hostname: {ctx.host}\n"
        f"         Icon name: computer-{chassis}\n"
        f"           Chassis: {chassis}\n"
        f"        Machine ID: {abs(hash(ctx.host)) % 10**32:032x}\n"
        f"           Boot ID: {abs(hash(ctx.host + 'boot')) % 10**32:032x}\n"
        "  Operating System: Ubuntu 22.04.4 LTS\n"
        f"            Kernel: Linux {ctx.node.kernel}\n"
        "      Architecture: x86-64\n"
        f"   Hardware Vendor: {'NVIDIA' if ctx.host != HEAD else 'Dell Inc.'}\n"
        f"    Hardware Model: {'DGX H100' if ctx.host != HEAD else 'PowerEdge R750'}\n"
    )


@command("timedatectl", help="timedatectl: clock and NTP status")
def timedatectl(ctx, args):
    now = clock.now(ctx.state)
    synced = ctx.node.services.get("chronyd") == "active"
    return (
        f"               Local time: {now:%a %Y-%m-%d %H:%M:%S} UTC\n"
        f"           Universal time: {now:%a %Y-%m-%d %H:%M:%S} UTC\n"
        f"                 RTC time: {now:%a %Y-%m-%d %H:%M:%S}\n"
        "                Time zone: Etc/UTC (UTC, +0000)\n"
        f"System clock synchronized: {'yes' if synced else 'no'}\n"
        f"              NTP service: {'active' if synced else 'inactive'}\n"
        "          RTC in local TZ: no\n"
    )


@command("modinfo", help="modinfo MODULE")
def modinfo(ctx, args):
    from .system import KNOWN_MODULES

    pos = [a for a in args if not a.startswith("-")]
    if not pos:
        raise CommandError("modinfo: ERROR: missing module or filename.", 1)
    m = pos[0].replace("-", "_")
    if m not in KNOWN_MODULES and m not in ctx.node.modules:
        raise CommandError(f"modinfo: ERROR: Module {m} not found.", 1)
    nvidia = m.startswith("nvidia")
    lines = [
        f"filename:       /lib/modules/{ctx.node.kernel}/{'updates/dkms' if nvidia else 'kernel/drivers/infiniband/hw/mlx5'}/{m}.ko",
        f"version:        {ctx.node.driver_version if nvidia else '24.04-0.6.6'}",
        f"license:        {'NVIDIA' if nvidia else 'Dual BSD/GPL'}",
        f"vermagic:       {ctx.node.kernel} SMP preempt mod_unload modversions",
    ]
    if m == "nvidia_peermem":
        lines.insert(2, "description:    NVIDIA GPUDirect RDMA peer memory client")
        lines.append("depends:        ib_core,nvidia")
    return "\n".join(lines) + "\n"


@command("ss", "netstat", help="ss -tlnp | netstat -tlnp: listening TCP ports")
def ss(ctx, args):
    flags = "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--"))
    show_proc = "p" in flags
    rows = [
        ("Netid", "State", "Recv-Q", "Send-Q", "Local Address:Port", "Peer Address:Port")
        + (("Process",) if show_proc else ())
    ]
    for svc, ports in LISTENERS.items():
        if ctx.node.services.get(svc) != "active":
            continue
        for p in ports:
            row = ("tcp", "LISTEN", "0", "128", f"0.0.0.0:{p}", "0.0.0.0:*")
            if show_proc:
                row += (f'users:(("{svc}",pid={1000 + abs(hash(svc)) % 900},fd=3))',)
            rows.append(row)
    if "u" in flags and "t" not in flags:
        rows = rows[:1]
    return columns(rows, gap=2)


@command("ifconfig", help="ifconfig [IFACE]")
def ifconfig(ctx, args):
    want = next((a for a in args if not a.startswith("-")), None)
    out = []
    for name, kind, up, addr, _p in netdevs(ctx.host, ctx.node):
        if want and name != want:
            continue
        if kind == "loopback":
            out += [
                f"{name}: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536",
                "        inet 127.0.0.1  netmask 255.0.0.0",
                "",
            ]
            continue
        flags = "4163<UP,BROADCAST,RUNNING,MULTICAST>" if up else "4099<UP,BROADCAST,MULTICAST>"
        mtu = 2044 if kind == "infiniband" else 1500
        out.append(f"{name}: flags={flags}  mtu {mtu}")
        if addr and up:
            ip_only = addr.split("/")[0]
            out.append(
                f"        inet {ip_only}  netmask 255.255.0.0  broadcast {'.'.join(ip_only.split('.')[:2])}.255.255"
            )
        mac = (
            f"7c:c2:55:1a:00:{node_no(ctx.host):02x}" if kind != "infiniband" else "80:00:02:08:fe:80:00:00:00:00:00:00"
        )
        out.append(
            f"        {'infiniband' if kind == 'infiniband' else 'ether'} {mac}  txqueuelen {256 if kind == 'infiniband' else 1000}  ({'InfiniBand' if kind == 'infiniband' else 'Ethernet'})"
        )
        out.append("        RX packets 184221  bytes 2918732110 (2.9 GB)")
        out.append("        TX packets 102114  bytes 1109233017 (1.1 GB)")
        out.append("")
    if want and not out:
        raise CommandError(f"{want}: error fetching interface information: Device not found", 1)
    return "\n".join(out).rstrip("\n") + "\n"


@command("ethtool", help="ethtool IFACE: link speed and state")
def ethtool(ctx, args):
    pos = [a for a in args if not a.startswith("-")]
    if not pos:
        raise CommandError("ethtool: bad command line argument(s)\nFor more information run ethtool -h", 1)
    devs = {d[0]: d for d in netdevs(ctx.host, ctx.node)}
    if pos[0] not in devs:
        raise CommandError("Cannot get device settings: No such device", 75)
    name, kind, up, _addr, port = devs[pos[0]]
    speed = (
        f"{port.rate_gbps * 1000}Mb/s"
        if kind == "infiniband" and port is not None and up
        else ("100000Mb/s" if up else "Unknown!")
    )
    return (
        f"Settings for {name}:\n"
        f"\tSupported ports: [ {'FIBRE' if kind == 'infiniband' else 'TP'} ]\n"
        f"\tSpeed: {speed}\n"
        f"\tDuplex: {'Full' if up else 'Unknown! (255)'}\n"
        f"\tPort: {'Other' if kind == 'infiniband' else 'Twisted Pair'}\n"
        "\tAuto-negotiation: on\n"
        f"\tLink detected: {'yes' if up else 'no'}\n"
    )


@command("nslookup", "dig", "host", help="nslookup HOST | dig HOST: resolve a cluster host name")
def nslookup(ctx, args):
    pos = [a for a in args if not a.startswith(("-", "+"))]
    if not pos:
        raise CommandError("usage: nslookup HOST", 1)
    name = pos[0]
    target = _resolve(ctx.state, name.split(".")[0])
    if target is None:
        raise CommandError(f"** server can't find {name}: NXDOMAIN", 1)
    ip = ctx.state.nodes[target].ip
    return f"Server:\t\t10.141.255.254\nAddress:\t10.141.255.254#53\n\nName:\t{name}\nAddress: {ip}\n"


@command("traceroute", help="traceroute HOST")
def traceroute(ctx, args):
    pos = [a for a in args if not a.startswith("-")]
    if not pos:
        raise CommandError("Usage: traceroute [options] host", 2)
    target = _resolve(ctx.state, pos[0])
    if target is None:
        raise CommandError(f"traceroute: unknown host {pos[0]}", 2)
    node = ctx.state.nodes[target]
    head = f"traceroute to {pos[0]} ({node.ip}), 30 hops max, 60 byte packets\n"
    if not node.get("up", True):
        return head + " 1  * * *\n 2  * * *\n 3  * * *\n"
    return head + f" 1  {pos[0]} ({node.ip})  0.118 ms  0.097 ms  0.091 ms\n"


@command("curl", "wget", help="curl [-sk] [-I] URL: reach a cluster service (the kube API, NGC, the helm repo)")
def curl(ctx, args):
    urls = [a for a in args if a.startswith(("http://", "https://"))]
    if not urls:
        raise CommandError("curl: try 'curl --help' for more information", 2)
    url = urls[0]
    m = re.match(r"https?://([^/:]+)(?::(\d+))?(/.*)?", url)
    host, port, path = m.group(1), m.group(2), m.group(3) or "/"
    short = "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--"))
    head_only = "I" in short or "--head" in args
    insecure = "k" in short or "--insecure" in args
    if host in (
        "nvcr.io",
        "helm.ngc.nvidia.com",
        "ngc.nvidia.com",
        "api.ngc.nvidia.com",
        "registry-1.docker.io",
        "nvidia.github.io",
    ):
        if not ctx.state.flags.get("internet", True):
            raise CommandError(f"curl: (6) Could not resolve host: {host}", 6)
        body = "HTTP/2 200\ncontent-type: text/html\n" if head_only else ""
        return (
            body
            if head_only
            else (
                '{"errors":[{"code":"UNAUTHORIZED","message":"authentication required"}]}\n'
                if host == "nvcr.io" and "v2" in path
                else ""
            )
        )
    target = _resolve(ctx.state, host)
    if target is None:
        raise CommandError(f"curl: (6) Could not resolve host: {host}", 6)
    node = ctx.state.nodes[target]
    if not node.get("up", True):
        raise CommandError(
            f"curl: (7) Failed to connect to {host} port {port or 80} after 3003 ms: No route to host", 7
        )
    port_no = int(port or 80)
    service = next((s for s, ports in LISTENERS.items() if port_no in ports), None)
    if service is None or node.services.get(service) != "active":
        raise CommandError(f"curl: (7) Failed to connect to {host} port {port_no} after 0 ms: Connection refused", 7)
    if service == "kube-apiserver":
        if not insecure:
            raise CommandError(
                "curl: (60) SSL certificate problem: unable to get local issuer certificate\n"
                "More details here: https://curl.se/docs/sslcerts.html",
                60,
            )
        if path in ("/healthz", "/livez", "/readyz"):
            return "ok"
        return (
            '{\n  "kind": "Status",\n  "apiVersion": "v1",\n  "status": "Failure",\n  "message": "forbidden: User \\"system:anonymous\\" cannot get path \\"'
            + path
            + '\\"",\n  "reason": "Forbidden",\n  "code": 403\n}\n'
        )
    if service == "kubelet":
        return "Unauthorized\n" if insecure else "curl: (60) SSL certificate problem\n"
    return "HTTP/1.1 200 OK\n" if head_only else ""


@command(
    "shutdown",
    "poweroff",
    help="shutdown -h now | poweroff: power the node off (power it back on with ipmitool or cmsh)",
)
def shutdown(ctx, args):
    if "-r" in args:
        return dispatch(ctx, ["reboot"])
    if "-c" in args:
        return ""
    if ctx.host == HEAD:
        raise CommandError("shutdown: refusing to power off the head node from the lab shell", 1)
    node = ctx.node
    jobs = audit.running_jobs_on(ctx.state, ctx.host)
    if jobs:
        audit.record(ctx.state, audit.POWER_OFF_WITH_JOBS, ctx.host, "shutdown", ", ".join(str(j.id) for j in jobs))
    for fn in HOOKS_REBOOT:
        fn(ctx, node)
    node.power = "off"
    node.up = False
    msg = f"Connection to {ctx.host} closed by remote host.\n"
    if ctx.host_stack:
        ctx.host = ctx.host_stack.pop()
        ctx.cwd = ctx.saved_cwd.pop(len(ctx.host_stack), "/root")
    return msg


@command("tmux", "screen", help="tmux / screen: not available in the lab shell")
def tmux(ctx, args):
    raise CommandError(
        "tmux: the lab shell is a single terminal; run commands one at a time (watch and & are available)", 1
    )
