"""CPU / NUMA topology tools: lscpu, numactl (and taskset for binding checks).

The simulated nodes are 2-socket Intel Xeon Platinum 8480C systems (DGX H100):
56 cores per socket, 2 threads per core, 224 CPUs, 2 NUMA nodes. Linux numbers
the first thread of every core first, so

    NUMA node0 CPUs: 0-55,112-167     (socket 0; GPUs 0-3 and NICs mlx5_0-3)
    NUMA node1 CPUs: 56-111,168-223   (socket 1; GPUs 4-7 and NICs mlx5_4-7)

Inside a Slurm job step, numactl/taskset report the CPUs the task was given
(see ``slurm.task_binding``), which is how a learner sees a bad gres.conf
``Cores=`` mapping.
"""

from __future__ import annotations

from ..core.errors import CommandError
from . import command


def _topo(node):
    cpus = int(node.get("cpus", 224) or 224)
    sockets = 2
    threads = 2
    cores = cpus // threads  # total physical cores
    per_socket = cores // sockets
    return cpus, sockets, threads, cores, per_socket


def _numa_cpus(node, numa: int) -> list[int]:
    _cpus, _s, _t, cores, per_socket = _topo(node)
    first = list(range(numa * per_socket, (numa + 1) * per_socket))
    return first + [c + cores for c in first]


def _ranges(nums) -> str:
    nums = sorted(nums)
    if not nums:
        return ""
    out, start, prev = [], nums[0], nums[0]
    for x in nums[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if x is not None:
            start = prev = x
    return ",".join(out)


def _numa_of_cpu(node, cpu: int) -> int:
    _cpus, _s, _t, cores, per_socket = _topo(node)
    return (cpu % cores) // per_socket


def _bound_cpus(ctx):
    """CPUs this process may run on: the job step's binding, or the whole machine."""
    from . import slurm

    node = ctx.node
    try:
        bound = slurm.task_binding(ctx.state, ctx.host)
    except Exception:
        bound = None
    if bound is None:
        return list(range(_topo(node)[0])), False
    return list(bound), True


def _mem_mb(node, numa):
    total = int(node.get("real_memory_mb", 2063000) or 2063000)
    return total // 2


@command("lscpu", help="lscpu — CPU model, sockets, cores, threads and NUMA node CPU lists")
def lscpu(ctx, args):
    node = ctx.node
    cpus, sockets, threads, cores, per_socket = _topo(node)
    model = "Intel(R) Xeon(R) Platinum 8480C"
    if "-e" in args or "--extended" in args:
        rows = ["CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ"]
        for c in range(cpus):
            core = c % cores
            numa = core // per_socket
            rows.append(f"{c:>3} {numa:>4} {numa:>6} {core:>4} {core}:{core}:{core}:{numa:<10} yes 3800.0000 800.0000")
        return "\n".join(rows) + "\n"
    if "-p" in args or any(a.startswith("--parse") for a in args):
        rows = ["# CPU,Core,Socket,Node"]
        for c in range(cpus):
            core = c % cores
            rows.append(f"{c},{core},{core // per_socket},{core // per_socket}")
        return "\n".join(rows) + "\n"
    lines = [
        ("Architecture:", "x86_64"),
        ("  CPU op-mode(s):", "32-bit, 64-bit"),
        ("  Address sizes:", "46 bits physical, 57 bits virtual"),
        ("  Byte Order:", "Little Endian"),
        ("CPU(s):", str(cpus)),
        ("  On-line CPU(s) list:", f"0-{cpus - 1}"),
        ("Vendor ID:", "GenuineIntel"),
        ("  Model name:", model),
        ("    CPU family:", "6"),
        ("    Model:", "143"),
        ("    Thread(s) per core:", str(threads)),
        ("    Core(s) per socket:", str(per_socket)),
        ("    Socket(s):", str(sockets)),
        ("    Stepping:", "8"),
        ("    CPU max MHz:", "3800.0000"),
        ("    CPU min MHz:", "800.0000"),
        ("Virtualization features:", ""),
        ("  Virtualization:", "VT-x"),
        ("Caches (sum of all):", ""),
        ("  L1d:", f"{cores * 48 / 1024:.1f} MiB ({cores} instances)"),
        ("  L1i:", f"{cores * 32 / 1024:.1f} MiB ({cores} instances)"),
        ("  L2:", f"{cores * 2} MiB ({cores} instances)"),
        ("  L3:", f"{105 * sockets} MiB ({sockets} instances)"),
        ("NUMA:", ""),
        ("  NUMA node(s):", str(sockets)),
    ]
    for n in range(sockets):
        lines.append((f"  NUMA node{n} CPU(s):", _ranges(_numa_cpus(node, n))))
    w = 37
    return "\n".join(f"{k:<{w}}{v}".rstrip() for k, v in lines) + "\n"


def _hardware(node):
    _cpus, sockets, *_ = _topo(node)
    out = [f"available: {sockets} nodes (0-{sockets - 1})"]
    for n in range(sockets):
        size = _mem_mb(node, n)
        out.append(f"node {n} cpus: " + " ".join(str(c) for c in _numa_cpus(node, n)))
        out.append(f"node {n} size: {size} MB")
        out.append(f"node {n} free: {int(size * 0.93)} MB")
    out.append("node distances:")
    out.append("node   " + "".join(f"{n:>4}" for n in range(sockets)))
    for a in range(sockets):
        out.append(f"  {a}:  " + "".join(f"{10 if a == b else 21:>4}" for b in range(sockets)))
    return "\n".join(out) + "\n"


def _show(ctx):
    node = ctx.node
    cpus, bound = _bound_cpus(ctx)
    numas = sorted({_numa_of_cpu(node, c) for c in cpus})
    _all, sockets, *_ = _topo(node)
    return (
        "policy: default\n"
        "preferred node: current\n"
        f"physcpubind: {' '.join(str(c) for c in sorted(cpus))} \n"
        f"cpubind: {' '.join(str(n) for n in numas)} \n"
        f"nodebind: {' '.join(str(n) for n in numas)} \n"
        f"membind: {' '.join(str(n) for n in range(sockets))} \n"
    )


@command("numactl", help="numactl -H | --show | --cpunodebind=N --membind=N <cmd> — NUMA topology and binding")
def numactl(ctx, args):
    if not args:
        raise CommandError(
            "usage: numactl [--all | -a] [--interleave= | -i <nodes>] [--preferred= | -p <node>]\n"
            "               [--physcpubind= | -C <cpus>] [--cpunodebind= | -N <nodes>]\n"
            "               [--membind= | -m <nodes>] [--localalloc | -l] command args ...\n"
            "       numactl [--show | -s]\n"
            "       numactl [--hardware | -H]",
            1,
        )
    if args[0] in ("-H", "--hardware"):
        return _hardware(ctx.node)
    if args[0] in ("-s", "--show"):
        return _show(ctx)
    # policy options followed by a command: run the command (binding is not modelled further)
    i = 0
    node_opts = {"-N": "--cpunodebind", "-m": "--membind", "-i": "--interleave", "-p": "--preferred"}
    while i < len(args) and args[i].startswith("-"):
        a = args[i]
        key, _, val = a.partition("=")
        if a in node_opts or a == "-C":
            if i + 1 >= len(args):
                raise CommandError(f"numactl: option requires an argument -- '{a[1:]}'", 1)
            key, val = node_opts.get(a, "--physcpubind"), args[i + 1]
            i += 1
        if key in ("--cpunodebind", "--membind", "--preferred", "--interleave"):
            allowed = {str(n) for n in range(_topo(ctx.node)[1])} | {"all"}
            if any(v not in allowed for v in val.replace("-", ",").split(",") if v):
                raise CommandError(f"numa_parse_nodestring: invalid node {val}\n(use numactl -H to list nodes)", 1)
        elif key == "--physcpubind":
            cpus = _topo(ctx.node)[0] * _topo(ctx.node)[1]
            if any(not v.isdigit() or int(v) >= cpus for v in val.replace("-", ",").split(",") if v):
                raise CommandError(f"numa_parse_cpustring: invalid cpu {val}\n(use numactl -H to list cpus)", 1)
        i += 1
    rest = args[i:]
    if not rest:
        raise CommandError("numactl: no command given (try numactl --show)", 1)
    from . import dispatch

    return dispatch(ctx, rest)


@command("taskset", help="taskset -cp <pid> | taskset -c <cpus> <cmd> — CPU affinity of a process")
def taskset(ctx, args):
    if not args:
        raise CommandError("usage: taskset [options] [mask | cpu-list] [pid|cmd [args...]]", 1)
    flags = "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--"))
    flags += "".join({"--pid": "p", "--cpu-list": "c"}.get(a, "") for a in args)
    pos = [a for a in args if not a.startswith("-")]
    if "p" in flags:
        pid = pos[-1] if pos else "$$"
        cpus, _ = _bound_cpus(ctx)
        return f"pid {pid}'s current affinity list: {_ranges(cpus)}\n"
    if len(pos) >= 2:
        from . import dispatch

        return dispatch(ctx, pos[1:])
    raise CommandError("taskset: bad usage", 1)
