"""Accounting views: sacct job records and the sshare fair-share tree."""

from __future__ import annotations

import re

from .. import command
from .common import (
    _USER,
    HEAD,
    STATE_CODES,
    TERMINAL,
    _err,
    _find_job,
    _has,
    _opt,
    _ts,
    compress_hostlist,
    fmt_elapsed,
    fmt_limit_upper,
    job_id_text,
)
from .fairshare import _acct_usage, _assoc_shares, _fs_factor, _level_total, _norm_acct, _user_usage
from .sacctmgr import _assoc_walk
from .scheduler import (
    _alloc_cpu_count,
    _alloc_mem_mb,
    _begin,
    _ensure,
    _job_alloc_mem_mb,
    _job_gpus,
    _need_ctl,
    _req_mem_mb,
)

SACCT_W = {
    "jobid": 12,
    "jobname": 10,
    "partition": 10,
    "account": 10,
    "alloccpus": 10,
    "state": 10,
    "exitcode": 8,
    "elapsed": 10,
    "nodelist": 15,
    "alloctres": 40,
    "reqtres": 40,
    "start": 19,
    "end": 19,
    "submit": 19,
    "user": 9,
    "qos": 10,
    "reason": 20,
    "timelimit": 10,
    "nnodes": 8,
    "ntasks": 8,
    "jobidraw": 12,
    "workdir": 20,
    "comment": 15,
    "priority": 10,
    "cluster": 10,
    "maxrss": 10,
    "maxvmsize": 10,
    "reqcpus": 8,
    "allocnodes": 10,
    "submitline": 40,
    "derivedexitcode": 10,
    "ncpus": 10,
    "reqmem": 10,
    "reqnodes": 8,
    "uid": 10,
    "group": 9,
    "layout": 10,
    "nodes": 15,
}
SACCT_HDR = {
    "jobid": "JobID",
    "jobname": "JobName",
    "partition": "Partition",
    "account": "Account",
    "alloccpus": "AllocCPUS",
    "state": "State",
    "exitcode": "ExitCode",
    "elapsed": "Elapsed",
    "nodelist": "NodeList",
    "alloctres": "AllocTRES",
    "reqtres": "ReqTRES",
    "start": "Start",
    "end": "End",
    "submit": "Submit",
    "user": "User",
    "qos": "QOS",
    "reason": "Reason",
    "timelimit": "Timelimit",
    "nnodes": "NNodes",
    "ntasks": "NTasks",
    "jobidraw": "JobIDRaw",
    "workdir": "WorkDir",
    "comment": "Comment",
    "priority": "Priority",
    "cluster": "Cluster",
    "maxrss": "MaxRSS",
    "maxvmsize": "MaxVMSize",
    "reqcpus": "ReqCPUS",
    "allocnodes": "AllocNodes",
    "submitline": "SubmitLine",
    "derivedexitcode": "DerivedExitCode",
    "ncpus": "NCPUS",
    "reqmem": "ReqMem",
    "reqnodes": "ReqNodes",
    "uid": "UID",
    "group": "Group",
    "layout": "Layout",
    "nodes": "Nodes",
}


def _units(amount: int | None, start: int) -> str:
    """A memory figure as sacct prints it by default; nothing when none is recorded.

    sacct moves a value up a unit only while that loses nothing (its test is a multiple of half
    the next unit), so 204800 MB reads 200G and 203464 MB stays 203464M. Zero is a bare 0.
    ``start`` is the unit the figure is in: 0 for bytes, 2 for megabytes.
    """
    if amount is None:
        return ""
    if not amount:
        return "0"
    value, unit = float(amount), start
    while value >= 1024 and int(value) % 512 == 0 and unit < 5:
        value, unit = value / 1024, unit + 1
    suffix = " KMGTP"[unit].strip()
    return f"{int(value)}{suffix}" if value == int(value) else f"{value:.2f}{suffix}"


def _bytes(kb) -> int | None:
    return None if kb is None else int(kb) * 1024


def _sacct_rows(state, j, steps=True):
    clock = state.slurm.clock
    elapsed = 0
    if j.start_time is not None:
        elapsed = (j.end_time if j.end_time is not None else clock) - j.start_time
    ncpu = j.num_nodes * j.ntasks_per_node * j.cpus_per_task
    acpu = _alloc_cpu_count(state, j)
    gtres = f"gres/gpu={_job_gpus(j)}," if j.gpus_per_node else ""
    # asked for and given are separate: a job that names no memory asks for 0 and is still allocated some
    req_mb = _req_mem_mb(state, j)
    alloc_mb = _job_alloc_mem_mb(state, j)

    def with_mem(megabytes: int) -> str:
        return gtres + f"mem={_units(megabytes, 2)},node={j.num_nodes}"

    tres = f"billing={ncpu},cpu={ncpu}," + with_mem(req_mb * j.num_nodes)
    atres = f"billing={acpu},cpu={acpu}," + with_mem(alloc_mb)
    st = j.state
    if st == "CANCELLED" and j.start_time is not None:
        st = "CANCELLED by 0"
    base = {
        "jobid": job_id_text(j),
        "jobidraw": str(j.id),
        "jobname": j.name,
        "partition": j.partition,
        "account": j.account,
        "alloccpus": str(acpu if j.start_time is not None else 0),
        "state": st,
        "exitcode": j.exit_code,
        "elapsed": fmt_elapsed(elapsed, long=True),
        "nodelist": compress_hostlist(j.nodelist) or "None assigned",
        "alloctres": atres if j.start_time is not None else "",
        "reqtres": tres,
        "start": _ts(state, j.start_time) if j.start_time is not None else "Unknown",
        "end": _ts(state, j.end_time) if j.end_time is not None else "Unknown",
        "submit": _ts(state, j.submit_time),
        "user": j.user,
        "qos": j.qos,
        "reason": j.reason,
        "timelimit": fmt_limit_upper(j.time_limit),
        "nnodes": str(j.num_nodes),
        "ntasks": "",
        "workdir": j.work_dir,
        "comment": "",
        "priority": str(j.priority),
        "cluster": "superpod",
        "maxrss": "",  # usage is gathered per step; the allocation line has none
        "maxvmsize": "",
        "reqcpus": str(ncpu),
        "allocnodes": str(j.num_nodes),
        "submitline": j.command,
        "derivedexitcode": "0:0",
        "ncpus": str(acpu if j.start_time is not None else ncpu),
        "reqmem": _units(req_mb, 2),
        "reqnodes": str(j.num_nodes),
        "uid": "0" if j.user == "root" else str(1000 + sum(map(ord, j.user)) % 500),
        "group": j.user,
        "layout": "Unknown",
        "nodes": compress_hostlist(j.nodelist) or "None assigned",
    }
    rows = [base]
    if not steps or j.start_time is None:
        return rows

    def step(sid, name, state_, code, nodes=None):
        # what the accounting plugin sampled for this step, when the job's record carries it
        used = (j.get("usage") or {}).get(str(sid)) or {}
        r = dict(base)
        r.update(
            {
                "jobid": f"{j.id}.{sid}",
                "jobname": name,
                "partition": "",
                "user": "",
                "qos": "",
                "state": state_,
                "exitcode": code,
                "nodelist": nodes or base["nodelist"],
                "reason": "",
                "timelimit": "",
                "ntasks": "1",
                # a step has no request of its own: what was asked for is on the allocation line
                "reqmem": "",
                "reqtres": "",
                "maxrss": _units(_bytes(used.get("max_rss_kb")), 0),
                "maxvmsize": _units(_bytes(used.get("max_vmsize_kb")), 0),
            }
        )
        if sid == "batch" and r["alloctres"]:
            # the batch script runs on the first node only, and billing is charged to the job
            first = j.nodelist[0]
            cpus = acpu // max(1, len(j.nodelist))
            gpus = f"gres/gpu={j.gpus_per_node}," if j.gpus_per_node else ""
            r["alloctres"] = f"cpu={cpus},{gpus}mem={_units(_alloc_mem_mb(state, j, first), 2)},node=1"
        elif sid != "extern" and r["alloctres"]:
            r["alloctres"] = r["alloctres"].removeprefix(f"billing={acpu},")
        return r

    term = j.state if j.state in TERMINAL else "RUNNING"
    if j.state == "CANCELLED":
        term = "CANCELLED"
    if j.batch:
        rows.append(step("batch", "batch", term, j.exit_code, j.nodelist[0] if j.nodelist else None))
    rows.append(step("extern", "extern", "COMPLETED" if j.state in TERMINAL else "RUNNING", "0:0"))
    for s_ in j.steps:
        rows.append(step(s_["id"], s_["name"], s_["state"], s_["exit_code"], s_.get("nodes")))
    return rows


@command("sacct", help="accounting: sacct [-j IDS] [-X] [-u USER] [-a] [--format=JobID,JobName,State,ExitCode,...]")
def sacct(ctx, args):
    state = ctx.state
    _ensure(state)
    head = state.nodes.get(HEAD)
    if head is not None and head.services.get("slurmdbd", "active") != "active":
        raise _err("sacct: error: Problem talking to the database: Connection refused")
    _begin(ctx)
    if args and args[-1] in ("-j", "--jobs"):  # getopt's complaint when the value is missing
        raise _err(
            "sacct: option requires an argument -- 'j'"
            if args[-1] == "-j"
            else "sacct: option '--jobs' requires an argument"
        )
    ids = _opt(args, "-j", "--jobs")
    if ids and not all(re.fullmatch(r"\d+(\.\w+)?", x) for x in ids.split(",")):
        raise _err(f"sacct: error: Invalid job id: {ids}")
    fmt = _opt(args, "-o", "--format")
    user = _opt(args, "-u", "--user")
    states = _opt(args, "-s", "--state")
    noheader = _has(args, "-n", "--noheader")
    parsable = _has(args, "-p", "--parsable", "-P", "--parsable2")
    allocs = _has(args, "-X", "--allocations")
    # root's accounting view covers every user (the help page says so); others see their own jobs
    if not user and not ids and not _has(args, "-a", "--allusers") and _USER[-1] != "root":
        user = _USER[-1]
    brief, long_ = _has(args, "-b", "--brief"), _has(args, "-l", "--long")
    since, until = _opt(args, "-S", "--starttime"), _opt(args, "-E", "--endtime")
    for flag, val in (("-S", since), ("-E", until)):
        if val is not None and not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?|now(-\d+(seconds|minutes|hours|days|weeks)?)?|today|yesterday|midnight|\d{2}:\d{2}(:\d{2})?",
            val,
        ):
            raise _err(f"sacct: error: Invalid time specified for {flag}: {val}")
    default_cols = "JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode"
    if brief:
        default_cols = "JobID,State,ExitCode"
    elif long_:
        default_cols = "JobID,JobName,Partition,Account,AllocCPUS,NCPUS,NNodes,NodeList,State,ExitCode,Elapsed,Timelimit,Start,End,ReqTRES,AllocTRES"
    cols = [c.strip() for c in (fmt or default_cols).split(",") if c.strip()]
    spec = []
    for c in cols:
        name, _, w = c.partition("%")
        key = name.lower()
        if key not in SACCT_HDR:
            raise _err(f'sacct: error: Invalid field requested: "{name}"')
        spec.append((key, int(w) if w else SACCT_W.get(key, 10)))
    rows = []
    for j in state.slurm.jobs:
        if j.get("array_tasks") is not None and j.state == "COMPLETED":
            continue  # the tasks carry the accounting; a finished master has nothing of its own
        if ids and not any(
            _find_job(state, x) is j or (str(j.get("array_job_id")) == x.split(".")[0].split("_")[0] and "_" not in x)
            for x in ids.split(",")
        ):
            continue
        if user and j.user != user:
            continue
        if states:
            want = {s_.upper() for s_ in states.split(",")}
            if j.state not in want and STATE_CODES.get(j.state) not in want:
                continue
        if (
            since
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", since)
            and (j.end_time is not None and _ts(state, j.end_time) < since)
        ):
            continue
        rows.extend(_sacct_rows(state, j, steps=not allocs and not brief))
    out = []
    if parsable:
        if not noheader:
            out.append("|".join(SACCT_HDR[k] for k, _ in spec) + ("|" if _has(args, "-p", "--parsable") else ""))
        for r in rows:
            out.append("|".join(r[k] for k, _ in spec) + ("|" if _has(args, "-p", "--parsable") else ""))
        return "\n".join(out) + "\n"
    if not noheader:
        out.append(" ".join(SACCT_HDR[k][:w].rjust(w) for k, w in spec))
        out.append(" ".join("-" * w for _, w in spec))
    for r in rows:
        cells = []
        for k, w in spec:
            v = str(r[k])
            if len(v) > w:
                v = v[: w - 1] + "+"
            cells.append(v.ljust(w) if k in ("jobid",) else v.rjust(w))
        out.append(" ".join(cells))
    return "\n".join(out) + "\n"


@command("sshare", help="fair-share tree: sshare [-a] [-l] [-A accounts] [-u users]")
def sshare(ctx, args):
    state = ctx.state
    _need_ctl(ctx, "sshare: error: ")
    _begin(ctx)
    s = state.slurm
    show_all = _has(args, "-a", "--all")
    longf = _has(args, "-l", "--long")
    accts = [a for a in (_opt(args, "-A", "--accounts") or "").split(",") if a]
    users = [u for u in (_opt(args, "-u", "--users") or "").split(",") if u]
    me = _USER[-1]
    total = max(1, _acct_usage(state, "root"))
    hdr = (
        "             Account       User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare"
        + ("    LevelFS" if longf else "")
        + "\n-------------------- ---------- ---------- ----------- ----------- ------------- ----------"
        + (" ----------" if longf else "")
    )
    rows = []
    for depth, acct, user in _assoc_walk(s):
        if accts and acct not in accts:
            continue
        if user is not None:
            if users and user not in users:
                continue
            if not show_all and not users and user != me:
                continue
            raw = _assoc_shares(s, acct, user)
            norm = raw / _level_total(s, acct)
            usage = _user_usage(state, acct, user)
            eff = usage / total
            fs = _fs_factor(state, user, acct)
            lvl = (norm / eff) if eff else float("inf")
            name = " " * depth + acct
            line = f"{name:<20} {user:>10} {raw:>10} {norm:>11.6f} {usage:>11} {eff:>13.6f} {fs:>10.6f}"
        else:
            usage = _acct_usage(state, acct)
            eff = usage / total
            if acct == "root" and depth == 0:
                line = f"{'root':<20} {'':>10} {'':>10} {0.0:>11.6f} {usage:>11} {1.0:>13.6f} {'':>10}"
                rows.append(line + (" " * 11 if longf else ""))
                continue
            raw = int(s.accounts[acct].get("fairshare", 1))
            norm = _norm_acct(s, acct)
            lvl = (norm / eff) if eff else float("inf")
            line = f"{' ' * depth + acct:<20} {'':>10} {raw:>10} {norm:>11.6f} {usage:>11} {eff:>13.6f} {'':>10}"
        if longf:
            line += f" {('inf' if lvl == float('inf') else f'{lvl:.6f}'):>10}"
        rows.append(line)
    return hdr + "\n" + "\n".join(rows) + "\n"
