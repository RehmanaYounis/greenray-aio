"""Queue and partition views: sinfo, squeue and sprio."""

from __future__ import annotations

from .. import command
from .common import (
    _FMT_RX,
    SLURM_VERSION,
    STATE_CODES,
    _banner_time,
    _err,
    _find_job,
    _has,
    _opt,
    _render,
    _ts,
    compress_hostlist,
    expand_hostlist,
    fmt_elapsed,
    fmt_limit,
    fmt_limit_upper,
    parse_gres_spec,
    parse_time,
)
from .conf import _param
from .fairshare import _fs_factor
from .nodes import _node_label
from .scheduler import _begin, _need_ctl, _part_nodes

SINFO_HDR = {
    "P": "PARTITION",
    "R": "PARTITION",
    "a": "AVAIL",
    "l": "TIMELIMIT",
    "D": "NODES",
    "t": "STATE",
    "T": "STATE",
    "N": "NODELIST",
    "n": "HOSTNAMES",
    "G": "GRES",
    "C": "CPUS(A/I/O/T)",
    "c": "CPUS",
    "m": "MEMORY",
    "E": "REASON",
    "H": "TIMESTAMP",
    "u": "USER",
    "f": "AVAIL_FEATURES",
    "b": "ACTIVE_FEATURES",
    "z": "S:C:T",
    "w": "WEIGHT",
    "d": "TMP_DISK",
    "O": "CPU_LOAD",
    "e": "FREE_MEM",
    "F": "NODES(A/I/O/T)",
    "s": "JOB_SIZE",
    "h": "OVERSUBSCRIBE",
    "g": "GROUPS",
    "r": "ROOT",
    "i": "RESERVATION",
    "o": "NODE_ADDR",
    "V": "CLUSTER",
    "Z": "GRES_USED",
}


def _sinfo_vals(state, p, sn):
    pname = p.name + ("*" if p.default else "")
    alloc = sn.alloc_cpus
    other = sn.cpus if (sn.drain or sn.down) else 0
    idle = 0 if other else sn.cpus - alloc
    gtype, gcnt = parse_gres_spec(sn.gres)
    gres = f"{sn.gres}(S:0-1)" if gcnt else "(null)"
    return {
        "P": pname,
        "R": p.name,
        "a": "up" if p.state == "UP" else p.state.lower(),
        "l": fmt_limit(parse_time(p.max_time)),
        "t": _node_label(sn),
        "T": _node_label(sn, True),
        "N": sn.name,
        "n": sn.name,
        "o": sn.name,
        "G": gres,
        "Z": sn.get("gres_used", ""),
        "C": f"{alloc if not other else 0}/{idle}/{other}/{sn.cpus}",
        "c": str(sn.cpus),
        "m": str(sn.real_memory),
        "E": sn.reason or "none",
        "H": sn.reason_time if sn.reason else "Unknown",
        "u": sn.reason_user if sn.reason else "Unknown",
        "f": sn.features or "(null)",
        "b": sn.features or "(null)",
        "z": "2:56:2",
        "w": str(sn.get("weight", 1)),
        "d": "0",
        "O": "0.42" if sn.alloc_cpus else "0.01",
        "e": "2011234",
        "s": "1-infinite",
        "h": "NO",
        "g": "all",
        "r": "no",
        "i": "",
        "V": "superpod",
    }


@command("sinfo", help="Slurm partition/node state: sinfo [-N] [-l] [-R] [-s] [-p P] [-n NODES] [-t STATE] [-o FMT]")
def sinfo(ctx, args):
    if _has(args, "-V", "--version"):
        return f"slurm {SLURM_VERSION}\n"
    _need_ctl(ctx, "slurm_load_partitions: ")
    _begin(ctx)
    state = ctx.state
    s = state.slurm
    node_mode = _has(args, "-N", "--Node")
    long = _has(args, "-l", "--long")
    noheader = _has(args, "-h", "--noheader")
    part_f = _opt(args, "-p", "--partition")
    node_f = _opt(args, "-n", "--nodes")
    state_f = _opt(args, "-t", "--states")
    fmt = _opt(args, "-o", "--format")
    if fmt and "%" not in fmt:
        raise _err(f"sinfo: error: Invalid node format specification: {fmt}")
    reasons = _has(args, "-R", "--list-reasons")
    summ = _has(args, "-s", "--summarize")
    width = max([len(p) + (1 if s.partitions[p].default else 0) for p in s.partitions] + [9])
    if reasons:
        fmt = fmt or ("%20E %12U %19H %N" if long else "%20E %9u %19H %N")
        fmt = fmt.replace("%12U", "%12u")
    elif summ:
        fmt = fmt or f"%{width}P %.5a %.10l %.15F %N"
    elif node_mode and long:
        fmt = fmt or f"%10N %.6D %.{width}P %.11T %.4c %.8z %.8m %.8d %.6w %.8f %20E"
    elif node_mode:
        fmt = fmt or f"%10N %.6D %{width}P %.5t"
    elif long:
        fmt = fmt or f"%{width}P %.5a %.10l %.10s %.4r %.8h %.10g %.6D %.11T %N"
    else:
        fmt = fmt or f"%{width}P %.5a %.10l %.6D %.6t %N"
    wanted_nodes = set(expand_hostlist(node_f)) if node_f else None
    wanted_states = {x.strip().lower() for x in state_f.split(",")} if state_f else None
    letters = {m[2] for m in _FMT_RX.findall(fmt)}
    rows = []
    seen_reason_nodes = set()
    for p in s.partitions.values():
        if part_f and p.name not in part_f.split(","):
            continue
        groups = {}
        order = []
        for n in _part_nodes(state, p.name):
            sn = s.nodes[n]
            if wanted_nodes is not None and n not in wanted_nodes:
                continue
            if wanted_states:
                lab = _node_label(sn).rstrip("*")
                longlab = _node_label(sn, True).rstrip("*")
                aliases = {lab, longlab, sn.state.lower()}
                if sn.drain:
                    aliases.add("drain")
                if not aliases & wanted_states:
                    continue
            if reasons:
                if not (sn.drain or sn.down or sn.not_responding) or n in seen_reason_nodes:
                    continue
                seen_reason_nodes.add(n)
            vals = _sinfo_vals(state, p, sn)
            if node_mode or reasons:
                key = (
                    (n,)
                    if node_mode
                    else tuple(vals.get(l, "") for l in sorted(letters - {"N", "n", "o", "D", "C", "F"}))
                )
            elif summ:
                key = ("summary",)
            else:
                key = tuple(vals.get(l, "") for l in sorted(letters - {"N", "n", "o", "D", "C", "F"}))
            if key not in groups:
                groups[key] = {"vals": vals, "nodes": [], "cpu": [0, 0, 0, 0], "f": [0, 0, 0, 0]}
                order.append(key)
            gr = groups[key]
            gr["nodes"].append(n)
            a, i, o, t = (int(x) for x in vals["C"].split("/"))
            gr["cpu"] = [gr["cpu"][0] + a, gr["cpu"][1] + i, gr["cpu"][2] + o, gr["cpu"][3] + t]
            cat = 2 if (sn.drain or sn.down) else (0 if sn.state in ("MIXED", "ALLOCATED") else 1)
            gr["f"][cat] += 1
            gr["f"][3] += 1
        if node_mode and not part_f and not reasons:
            pass
        for key in order:
            gr = groups[key]
            v = dict(gr["vals"])
            v["N"] = compress_hostlist(gr["nodes"])
            v["n"] = v["o"] = v["N"]
            v["D"] = str(len(gr["nodes"]))
            v["C"] = "/".join(str(x) for x in gr["cpu"])
            v["F"] = "/".join(str(x) for x in gr["f"])
            rows.append(v)
    out = _render(fmt, rows, SINFO_HDR, noheader)
    if long and not reasons:
        out = _banner_time(state) + "\n" + out
    return out


SQUEUE_HDR = {
    "i": "JOBID",
    "A": "JOBID",
    "P": "PARTITION",
    "j": "NAME",
    "u": "USER",
    "t": "ST",
    "T": "STATE",
    "M": "TIME",
    "l": "TIME_LIMIT",
    "L": "TIME_LEFT",
    "D": "NODES",
    "R": "NODELIST(REASON)",
    "r": "REASON",
    "N": "NODELIST",
    "b": "TRES_PER_NODE",
    "q": "QOS",
    "C": "CPUS",
    "a": "ACCOUNT",
    "Q": "PRIORITY",
    "p": "PRIORITY",
    "S": "START_TIME",
    "V": "SUBMIT_TIME",
    "Z": "WORK_DIR",
    "e": "END_TIME",
    "k": "COMMENT",
    "o": "COMMAND",
    "E": "DEPENDENCY",
    "Y": "SCHEDNODES",
    "y": "NICE",
    "h": "OVER_SUBSCRIBE",
    "x": "EXC_NODES",
    "n": "REQ_NODES",
}


def _job_tres_per_node(j):
    if not j.gpus_per_node:
        return "N/A"
    return f"gres/gpu:{j.gpu_type + ':' if j.gpu_type else ''}{j.gpus_per_node}"


def _job_vals(state, j):
    clock = state.slurm.clock
    used = (
        (clock - j.start_time)
        if j.state == "RUNNING" and j.start_time is not None
        else ((j.end_time - j.start_time) if j.start_time is not None and j.end_time is not None else 0)
    )
    reason = j.reason if j.reason not in ("", None) else "None"
    if j.state == "PENDING":
        nodelist_reason = f"({reason})"
    else:
        nodelist_reason = compress_hostlist(j.nodelist)
    tl = (
        j.time_limit
        if j.time_limit is not None
        else parse_time(state.slurm.partitions.get(j.partition, {}).get("max_time"))
    )
    return {
        "i": str(j.id),
        "A": str(j.id),
        "P": j.partition,
        "j": j.name,
        "u": j.user,
        "t": STATE_CODES.get(j.state, j.state[:2]),
        "T": j.state,
        "M": fmt_elapsed(used),
        "l": fmt_limit_upper(tl) if tl is not None else "UNLIMITED",
        "L": fmt_limit_upper(max(0, tl - used // 60)) if tl is not None else "UNLIMITED",
        "D": str(j.num_nodes),
        "R": nodelist_reason,
        "r": reason,
        "N": compress_hostlist(j.nodelist),
        "b": _job_tres_per_node(j),
        "q": j.qos,
        "a": j.account,
        "C": str(j.num_nodes * j.ntasks_per_node * j.cpus_per_task),
        "Q": str(j.priority),
        "p": str(j.priority),
        "S": _ts(state, j.start_time) if j.start_time is not None else "N/A",
        "V": _ts(state, j.submit_time),
        "e": _ts(state, j.end_time) if j.end_time is not None else "N/A",
        "Z": j.work_dir,
        "k": "(null)",
        "o": j.command or "(null)",
        "E": j.dependency or "(null)",
        "Y": "(null)",
        "y": str(j.nice),
        "h": "NO" if not j.exclusive else "USER",
        "x": compress_hostlist(j.exc_nodes) or "",
        "n": compress_hostlist(j.req_nodes) or "",
    }


@command("squeue", help="job queue: squeue [-u USER] [-p PART] [-j IDS] [-t STATES] [-l] [--start] [-o FMT]")
def squeue(ctx, args):
    if _has(args, "-V", "--version"):
        return f"slurm {SLURM_VERSION}\n"
    _need_ctl(ctx, "slurm_load_jobs error: ")
    _begin(ctx)
    state = ctx.state
    user = _opt(args, "-u", "--user")
    if _has(args, "--me"):
        user = "root"
    part = _opt(args, "-p", "--partition")
    ids = _opt(args, "-j", "--jobs")
    states = _opt(args, "-t", "--states")
    fmt = _opt(args, "-o", "--format")
    noheader = _has(args, "-h", "--noheader")
    long = _has(args, "-l", "--long")
    start = _has(args, "--start")
    nodes = _opt(args, "-w", "--nodelist")
    if start:
        fmt = fmt or "%.18i %.9P %.8j %.8u %.2t %.19S %.6D %20Y %R"
    elif long:
        fmt = fmt or "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"
    else:
        fmt = fmt or "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
    if states:
        want = set()
        for st in states.upper().split(","):
            if st == "ALL":
                want = None
                break
            long_name = next((k for k, v in STATE_CODES.items() if v == st), st)
            want.add(long_name)
    else:
        want = {"PENDING", "RUNNING", "COMPLETING", "SUSPENDED"}
    if ids:
        idl = [x.strip() for x in ids.split(",")]
        if not any(_find_job(state, i) for i in idl):
            raise _err("slurm_load_jobs error: Invalid job id specified")
    rows = []
    jobs = list(state.slurm.jobs)
    if start:
        jobs = [j for j in jobs if j.state == "PENDING"]
    jobs.sort(key=lambda j: (0 if j.state == "PENDING" else 1, -j.priority if j.state == "PENDING" else 0, j.id))
    for j in jobs:
        if want is not None and j.state not in want and not ids:
            continue
        if ids and str(j.id) not in [x.strip() for x in ids.split(",")]:
            continue
        if user and j.user not in user.split(","):
            continue
        if part and j.partition not in part.split(","):
            continue
        if nodes and not set(expand_hostlist(nodes)) & set(j.nodelist):
            continue
        rows.append(_job_vals(state, j))
    out = _render(fmt, rows, SQUEUE_HDR, noheader)
    if long:
        out = _banner_time(state) + "\n" + out
    return out


@command("sprio", help="pending job priority factors: sprio [-l] [-j ID]")
def sprio(ctx, args):
    state = ctx.state
    _need_ctl(ctx, "sprio: error: ")
    _begin(ctx)
    s = state.slurm
    w_age = int(_param(state, "PriorityWeightAge", 0) or 0)
    w_fs = int(_param(state, "PriorityWeightFairshare", 0) or 0)
    w_qos = int(_param(state, "PriorityWeightQOS", 0) or 0)
    ids = _opt(args, "-j", "--jobs")
    long = _has(args, "-l", "--long")
    rows = []
    maxq = max([int(q.get("priority", 0)) for q in s.qos.values()] + [0])
    for j in s.jobs:
        if j.state != "PENDING":
            continue
        if ids and str(j.id) not in ids.split(","):
            continue
        age = int(w_age * min(1.0, max(0, s.clock - j.submit_time) / (7 * 86400)))
        fs = int(w_fs * _fs_factor(state, j.user, j.account))
        qf = int(w_qos * (int(s.qos.get(j.qos, {}).get("priority", 0)) / maxq)) if maxq else 0
        if long:
            rows.append(
                f"{j.id:>15} {j.partition:<9} {j.user:>8} {j.priority:>10} {0:>10} {age:>10} {fs:>10} "
                f"{0:>10} {0:>10} {qf:>10}"
            )
        else:
            rows.append(f"{j.id:>15} {j.partition:<9} {j.priority:>10} {0:>10} {age:>10} {fs:>10} {qf:>10}")
    if long:
        hdr = "          JOBID PARTITION     USER   PRIORITY       SITE        AGE  FAIRSHARE    JOBSIZE  PARTITION        QOS"
    else:
        hdr = "          JOBID PARTITION   PRIORITY       SITE        AGE  FAIRSHARE        QOS"
    if not rows and not ids:
        return hdr + "\n"
    return "\n".join([hdr] + rows) + "\n"
