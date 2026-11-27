"""Job submission: sbatch, srun, salloc and scancel, with the option parsing and submit-time validation they share."""

from __future__ import annotations

import math
import re
import shlex

from ...core import audit
from ...core.errors import CommandError
from .. import command
from ..common import int_arg
from .common import (
    _USER,
    BATCH_DEFAULT_DURATION,
    SLURM_VERSION,
    STATE_CODES,
    TERMINAL,
    _err,
    _find_job,
    _has,
    _opt,
    _ts,
    compress_hostlist,
    expand_hostlist,
    is_admin,
    parse_gres_spec,
    parse_time,
    time_arg,
)
from .conf import _enforce, _gpu_types_enabled, _param, _pyxis_client_ok
from .nodes import _conf_gpus, _finish
from .scheduler import (
    _begin,
    _default_mem,
    _default_partition,
    _ensure,
    _job_defaults,
    _knows_memory,
    _mem_for_node,
    _need_ctl,
    _part_nodes,
    schedule,
)
from .steps import _CURRENT, _expand_pattern, _run_step

OPTS_ARG = {
    "-N": "nodes",
    "--nodes": "nodes",
    "-n": "ntasks",
    "--ntasks": "ntasks",
    "--ntasks-per-node": "ntasks_per_node",
    "-c": "cpus_per_task",
    "--cpus-per-task": "cpus_per_task",
    "-p": "partition",
    "--partition": "partition",
    "-w": "nodelist",
    "--nodelist": "nodelist",
    "-x": "exclude",
    "--exclude": "exclude",
    "-t": "time",
    "--time": "time",
    "-J": "name",
    "--job-name": "name",
    "-q": "qos",
    "--qos": "qos",
    "-A": "account",
    "--account": "account",
    "-o": "output",
    "--output": "output",
    "-e": "error",
    "--error": "error",
    "-D": "chdir",
    "--chdir": "chdir",
    "--gres": "gres",
    "-G": "gpus",
    "--gpus": "gpus",
    "--gpus-per-node": "gpus_per_node",
    "--gpus-per-task": "gpus_per_task",
    "--mem": "mem",
    "--mem-per-cpu": "mem_per_cpu",
    "--mail-type": "mail_type",
    "--mail-user": "mail_user",
    "--tasks-per-node": "ntasks_per_node",
    "--ntasks-per-core": "ntasks_per_core",
    "--threads-per-core": "threads_per_core",
    "--gpu-bind": "gpu_bind",
    "--open-mode": "open_mode",
    "--priority": "priority",
    "-d": "dependency",
    "--dependency": "dependency",
    "--wrap": "wrap",
    "-C": "constraint",
    "--constraint": "constraint",
    "--mpi": "mpi",
    "--export": "export",
    "--reservation": "reservation",
    "--comment": "comment",
    "--begin": "begin",
    "--mem-per-gpu": "mem_per_gpu",
    "--cpus-per-gpu": "cpus_per_gpu",
    "--container-image": "container_image",
    "--container-mounts": "container_mounts",
    "--container-name": "container_name",
    "--container-workdir": "container_workdir",
    "--container-env": "container_env",
    "--nice": "nice",
    "-a": "array",
    "--array": "array",
    "--signal": "signal",
    "--time-min": "time_min",
    "--distribution": "distribution",
    "-m": "distribution",
    "--gres-flags": "gres_flags",
    "--cpu-bind": "cpu_bind",
    "--cpu_bind": "cpu_bind",
    "--mem-bind": "mem_bind",
    "--hint": "hint",
    "--ntasks-per-gpu": "ntasks_per_gpu",
    "--sockets-per-node": "sockets_per_node",
}
OPTS_FLAG = {
    "--exclusive",
    "--wait",
    "-W",
    "--oversubscribe",
    "-s",
    "--spread-job",
    "--kill-on-invalid-dep",
    "--pty",
    "-l",
    "--label",
    "--parsable",
    "-H",
    "--hold",
    "--container-remap-root",
    "--no-container-remap-root",
    "--no-container-mount-home",
    "--container-mount-home",
    "-Q",
    "--quiet",
    "--requeue",
    "--no-requeue",
    "-v",
    "--verbose",
    "--overlap",
    "-K",
    "--kill-on-bad-exit",
    "--test-only",
    "--no-kill",
    "-k",
    "--container-writable",
    "--container-readonly",
    "-I",
    "--immediate",
    "-u",
    "--unbuffered",
    "--no-shell",
}
SHELLS = ("bash", "/bin/bash", "sh", "/bin/sh")
PYXIS_OPTS = {
    "--container-image",
    "--container-mounts",
    "--container-name",
    "--container-workdir",
    "--container-env",
    "--container-remap-root",
    "--no-container-remap-root",
    "--no-container-mount-home",
    "--container-mount-home",
    "--container-writable",
    "--container-readonly",
}


def parse_job_opts(prog, tokens, stop_at_positional=True, pyxis=True):
    """Parse sbatch/srun/salloc options. Returns (opts, positional_rest)."""
    opts = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("-") or t == "-":
            if stop_at_positional:
                return opts, tokens[i:]
            i += 1
            continue
        if t == "--":
            return opts, tokens[i + 1 :]
        name, val = t, None
        if t.startswith("--") and "=" in t:
            name, val = t.split("=", 1)
        elif not t.startswith("--") and len(t) > 2 and t[:2] in OPTS_ARG:
            name, val = t[:2], t[2:]
        if (name in PYXIS_OPTS) and not pyxis:
            raise _err(f"{prog}: unrecognized option '{t}'\nTry \"{prog} --help\" for more information")
        if name in OPTS_ARG:
            if val is None:
                if i + 1 >= len(tokens):
                    raise _err(f"{prog}: option '{name}' requires an argument")
                val = tokens[i + 1]
                i += 1
            opts[OPTS_ARG[name]] = val
        elif name in OPTS_FLAG:
            opts[name.lstrip("-")] = True
        else:
            raise _err(f"{prog}: unrecognized option '{t}'\nTry \"{prog} --help\" for more information")
        i += 1
    return opts, []


def _build_job(ctx, prog, opts, extra_check=True):
    """Turn parsed options into a job dict and validate it like slurmctld does at submit time."""
    state = ctx.state
    s = state.slurm
    fail = (
        (lambda m: _err(f"{prog}: error: Batch job submission failed: {m}"))
        if prog == "sbatch"
        else (lambda m: _err(f"{prog}: error: Unable to allocate resources: {m}"))
    )
    part = opts.get("partition") or _default_partition(state)
    if not part:
        raise fail("No partition specified or system default partition")
    if part not in s.partitions:
        raise _err(
            f"{prog}: error: invalid partition specified: {part}\n{prog}: error: "
            f"{'Batch job submission failed' if prog == 'sbatch' else 'Unable to allocate resources'}: Invalid partition name specified"
        )
    p = s.partitions[part]
    if p.state in ("INACTIVE", "DRAIN"):
        raise fail("Required partition not available (inactive or drain)")
    pnodes = _part_nodes(state, part)
    max_gpn = max([_conf_gpus(s.nodes[n]) for n in pnodes] + [0])
    for key, flag in (("ntasks", "-n"), ("ntasks_per_node", "--ntasks-per-node"), ("cpus_per_task", "-c")):
        if key in opts and not str(opts[key]).isdigit():
            raise _err(f'{prog}: error: Invalid numeric value "{opts[key]}" for {flag}.')
    if "nodes" in opts:
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", str(opts["nodes"]))
        if not m or int(m.group(1)) == 0 or (m.group(2) and int(m.group(2)) < int(m.group(1))):
            raise _err(f'{prog}: error: Invalid numeric value "{opts["nodes"]}" for -N.')
    if "nice" in opts and not re.fullmatch(r"-?\d+", str(opts["nice"])):
        raise _err(f"{prog}: error: Invalid --nice value: {opts['nice']}")
    num_nodes = int(str(opts["nodes"]).split("-")[0]) if "nodes" in opts else None
    ntasks = int(opts["ntasks"]) if "ntasks" in opts else None
    tpn = int(opts["ntasks_per_node"]) if "ntasks_per_node" in opts else None
    gpn, gtype = 0, None
    if "gres" in opts:
        gtype, gpn = parse_gres_spec(opts["gres"])
        if gpn == 0 and opts["gres"] not in ("", "none"):
            raise fail("Invalid generic resource (gres) specification")
    if "gpus_per_node" in opts:
        gtype, gpn = parse_gres_spec("gpu:" + opts["gpus_per_node"])
    if "gpus_per_task" in opts:
        gtype, per = parse_gres_spec("gpu:" + opts["gpus_per_task"])
        gpn = per * (tpn or 1)
    if "gpus" in opts:
        gtype, total = parse_gres_spec("gpu:" + opts["gpus"])
        if num_nodes is None:
            num_nodes = max(1, math.ceil(total / max_gpn)) if max_gpn else 1
        gpn = math.ceil(total / num_nodes)
    if num_nodes is None:
        num_nodes = 1
        if ntasks and tpn:
            num_nodes = max(1, math.ceil(ntasks / tpn))
    if tpn is None:
        tpn = max(1, math.ceil(ntasks / num_nodes)) if ntasks else 1
    if gpn and not _gpu_types_enabled(state):
        raise fail("Invalid generic resource (gres) specification")
    user = _USER[-1]
    qos = opts.get("qos") or s.users.get(user, {}).get("default_qos") or "normal"
    if qos not in s.qos:
        raise fail("Invalid qos specification")
    if p.allow_qos not in ("ALL", "", None) and qos not in str(p.allow_qos).split(","):
        raise fail("Invalid qos for job request")
    enf = _enforce(state)
    acct = opts.get("account") or s.users.get(user, {}).get("default_account", "root")
    if "associations" in enf:
        u = s.users.get(user)
        if u is None or acct not in u.get("accounts", [u.get("default_account")]):
            raise fail("Invalid account or account/partition combination specified")
        if acct not in s.accounts:
            raise fail("Invalid account or account/partition combination specified")
        if "qos" in enf and qos not in u.get("qos", ["normal"]):
            raise fail("Invalid qos specification")
    if num_nodes > len(pnodes):
        raise fail("Node count specification invalid")
    if gpn and gpn > max_gpn:
        raise fail("Requested node configuration is not available")
    if gtype:
        types = {parse_gres_spec(s.nodes[n].gres)[0] for n in pnodes}
        if gtype not in types:
            raise fail("Requested node configuration is not available")
    req = expand_hostlist(opts.get("nodelist")) if opts.get("nodelist") else []
    for n in req:
        if n not in pnodes:
            raise fail("Requested node configuration is not available")
    if len(req) > num_nodes:
        num_nodes = len(req)
    tl = time_arg(opts["time"], f"{prog}: error: Invalid --time specification") if "time" in opts else None
    if tl is None and p.get("default_time") not in (None, "NONE", ""):
        tl = parse_time(p.default_time)
    epl = str(_param(state, "EnforcePartLimits", "NO")).upper()
    if epl in ("YES", "ALL", "ANY"):
        mt = parse_time(p.max_time)
        if mt is not None and tl is not None and tl > mt:
            raise fail("Requested time limit is invalid (missing or exceeds some limit)")
        if str(p.max_nodes).isdigit() and num_nodes > int(p.max_nodes):
            raise fail("Node count specification invalid")
    job = {
        "name": opts.get("name") or "",
        "user": user,
        "account": acct,
        "account_explicit": "account" in opts,  # -A/--account on the command line or in an #SBATCH line
        "partition": part,
        "qos": qos,
        "state": "PENDING",
        "reason": "None",
        "num_nodes": num_nodes,
        "ntasks_per_node": tpn,
        "cpus_per_task": int(opts.get("cpus_per_task", 1)),
        "gpus_per_node": gpn,
        "gpu_type": gtype,
        "time_limit": tl,
        "req_nodes": req,
        "exc_nodes": expand_hostlist(opts.get("exclude")) if opts.get("exclude") else [],
        "submit_time": s.clock,
        "work_dir": opts.get("chdir") or getattr(ctx, "cwd", None) or "/root",
        "container_image": opts.get("container_image", ""),
        "dependency": opts.get("dependency", ""),
        "held": bool(opts.get("hold") or opts.get("H")),
        "exclusive": bool(opts.get("exclusive")),
        "submit_host": ctx.host,
        "nice": int(opts.get("nice", 0) or 0),
        "gres_flags": opts.get("gres_flags", ""),
        "cpu_bind": opts.get("cpu_bind", ""),
    }
    ok_flags = {"enforce-binding", "disable-binding", "one-task-per-sharing", "multiple-tasks-per-sharing"}
    if any(f not in ok_flags for f in str(job["gres_flags"]).split(",") if f):
        raise _err(f"{prog}: error: Invalid --gres-flags specification")
    job.update(_specification(prog, opts, state, pnodes, tl, fail))
    # memory: one of --mem, --mem-per-cpu, --mem-per-gpu, checked against the biggest node in the partition
    per_node = job.get("ntasks_per_node") or 1
    if "mem_per_cpu" in opts:
        job["mem_per_cpu"] = _mem_mb(prog, "--mem-per-cpu", opts["mem_per_cpu"])
        job["min_memory_mb"] = job["mem_per_cpu"] * per_node * job["cpus_per_task"]
    elif "mem_per_gpu" in opts:
        job["mem_per_gpu"] = _mem_mb(prog, "--mem-per-gpu", opts["mem_per_gpu"])
        job["min_memory_mb"] = job["mem_per_gpu"] * (job["gpus_per_node"] or 0)
    biggest = max([int(s.nodes[n].get("real_memory", 0)) for n in pnodes] + [0])
    if job.get("min_memory_mb", 0) and biggest and job["min_memory_mb"] > biggest:
        raise fail("Requested node configuration is not available")
    # the same goes for memory that comes from slurm.conf's default: refused now, not parked forever
    sized = [n for n in pnodes if _knows_memory(s.nodes[n])]
    if sized and all(_mem_for_node(state, job, n) > int(s.nodes[n].real_memory) for n in sized):
        raise fail("Requested node configuration is not available")
    usable_after_exclude = [n for n in pnodes if n not in job["exc_nodes"]]
    if job["exc_nodes"] and (num_nodes or 1) > len(usable_after_exclude):
        raise fail("Requested node configuration is not available")
    for clause in [c for c in str(job["dependency"]).split(",") if c]:
        kind, _, ids = clause.partition(":")
        if kind == "singleton":
            continue
        if kind not in ("after", "afterok", "afterany", "afternotok", "afterburstbuffer", "aftercorr"):
            raise fail("Job dependency problem")
        for one in [x for x in ids.split(":") if x]:
            if one.endswith("+") or not one.split("_")[0].isdigit() or _find_job(state, one) is None:
                raise fail("Job dependency problem")
    if "begin" in opts:
        job["eligible_time"] = _begin_time(prog, state, opts["begin"])
    if "array" in opts:
        job["array_tasks"], job["array_limit"] = _parse_array(prog, opts["array"])
        job["array_task_count"] = len(job["array_tasks"])
        job["array_spec"] = opts["array"]
    for key in ("mail_type", "mail_user", "gpu_bind", "open_mode"):
        if key in opts:
            job[key] = opts[key]
    if "priority" in opts:
        if _USER[-1] != "root":
            raise _err(f"{prog}: error: Batch job submission failed: Access/permission denied")
        job["priority"] = int_arg(
            opts["priority"], f'{prog}: error: Invalid numeric value "{opts["priority"]}" for --priority.'
        )
        job["priority_fixed"] = True
    return job


def _begin_time(prog, state, spec: str) -> int:
    """``--begin`` as simulated clock seconds: now+<n><unit>, HH:MM, YYYY-MM-DD[THH:MM[:SS]], tomorrow, midnight."""
    import datetime as _dt

    from .common import BASE_TIME

    clock = state.slurm.get("clock", 0)
    now = BASE_TIME + _dt.timedelta(seconds=clock)
    s = spec.strip().lower()
    if s == "now":
        return clock
    m = re.fullmatch(r"now\+(\d+)\s*(second|seconds|minute|minutes|hour|hours|day|days)?", s)
    if m:
        unit = (m.group(2) or "seconds").rstrip("s")
        mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit]
        return clock + int(m.group(1)) * mult
    if s in ("midnight", "tomorrow"):
        day = now.replace(hour=0, minute=0, second=0) + _dt.timedelta(days=1)
        return int((day - BASE_TIME).total_seconds())
    if s == "noon":
        day = now.replace(hour=12, minute=0, second=0)
        day += _dt.timedelta(days=1) if day <= now else _dt.timedelta()
        return int((day - BASE_TIME).total_seconds())
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        t = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=int(m.group(3) or 0))
        if t <= now:
            t += _dt.timedelta(days=1)
        return int((t - BASE_TIME).total_seconds())
    for fmt in ("%Y-%m-%dt%H:%M:%S", "%Y-%m-%dt%H:%M", "%Y-%m-%d"):
        try:
            t = _dt.datetime.strptime(s, fmt)
            return int((t - BASE_TIME).total_seconds())
        except ValueError:
            continue
    raise _err(f"{prog}: error: Invalid --begin specification")


def _parse_array(prog, spec: str) -> tuple[list[int], int | None]:
    """``1-4%2`` -> ([1, 2, 3, 4], 2); also ``0,3,5`` and ``1-10:2``."""
    body, _, limit = spec.partition("%")
    tasks: list[int] = []
    for part in body.split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", part.strip())
        if not m:
            raise _err(f"{prog}: error: Invalid job array specification")
        lo, hi, step = int(m.group(1)), int(m.group(2) or m.group(1)), int(m.group(3) or 1)
        if hi < lo or step < 1:
            raise _err(f"{prog}: error: Invalid job array specification")
        tasks.extend(range(lo, hi + 1, step))
    if not tasks or len(tasks) > 1001:
        raise _err(f"{prog}: error: Batch job submission failed: Invalid job array specification")
    if limit and not limit.isdigit():
        raise _err(f"{prog}: error: Invalid job array specification")
    return sorted(set(tasks)), (int(limit) if limit else None)


HINTS = {"compute_bound", "memory_bound", "multithread", "nomultithread", "help"}
DISTRIBUTIONS = {"block", "cyclic", "plane", "arbitrary", "*"}
MEM_BINDS = {"none", "local", "map_mem", "mask_mem", "prefer", "quiet", "verbose", "nosort", "sort"}
MPI_TYPES = {"pmix", "pmi2", "none", "cray_shasta", "list"}


def _mem_mb(prog, flag, value):
    m = re.fullmatch(r"(\d+)([kKmMgGtT]?)", str(value))
    if not m:
        raise _err(f"{prog}: error: Invalid {flag} specification")
    scale = {"": 1, "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}[m.group(2).lower()]
    return int(int(m.group(1)) * scale)


def _specification(prog, opts, state, pnodes, time_limit, fail):
    """The parts of a submission slurmctld keeps on the job record, validated as it would."""
    s = state.slurm
    spec = {
        "comment": opts.get("comment", ""),
        "features": opts.get("constraint", ""),
        "reservation": opts.get("reservation", ""),
        "export": opts.get("export", "ALL"),
        "hint": opts.get("hint", ""),
        "distribution": opts.get("distribution", ""),
        "mem_bind": opts.get("mem_bind", ""),
        "mpi": opts.get("mpi", ""),
        "signal": opts.get("signal", ""),
        "requeue": not opts.get("no-requeue"),
        "stderr": opts.get("error", ""),
        "container_mounts": opts.get("container_mounts", ""),
        "container_name": opts.get("container_name", ""),
        "container_workdir": opts.get("container_workdir", ""),
        "container_env": opts.get("container_env", ""),
        "container_flags": sorted(k for k in opts if k.startswith(("container-", "no-container-")) and opts[k] is True),
    }
    if spec["features"]:
        known = {f for n in pnodes for f in str(s.nodes[n].get("features") or "").split(",") if f}
        wanted = {f for f in re.split(r"[&|,()]", spec["features"]) if f}
        if not wanted <= known:
            raise fail("Invalid feature specification")
    if spec["reservation"] and spec["reservation"] not in (s.get("reservations") or {}):
        raise fail("Requested reservation is invalid")
    if spec["hint"] and spec["hint"] not in HINTS:
        raise _err(f"{prog}: error: Invalid --hint specification: {spec['hint']}")
    if spec["distribution"] and spec["distribution"].split(":")[0] not in DISTRIBUTIONS:
        raise _err(f"{prog}: error: Invalid --distribution specification: {spec['distribution']}")
    if spec["mem_bind"] and spec["mem_bind"].split(":")[0] not in MEM_BINDS:
        raise _err(f"{prog}: error: Invalid --mem-bind specification: {spec['mem_bind']}")
    if spec["mpi"] and spec["mpi"] not in MPI_TYPES:
        raise _err(f"{prog}: error: Invalid MPI type '{spec['mpi']}', --mpi=list for available types")
    if spec["signal"] and not re.fullmatch(r"([RB]:)?[A-Za-z0-9]+(@\d+)?", spec["signal"]):
        raise _err(f"{prog}: error: Invalid signal specification: {spec['signal']}")
    for key, flag in (("mem", "--mem"), ("mem_per_gpu", "--mem-per-gpu")):
        if key in opts:
            spec["min_memory_mb"] = _mem_mb(prog, flag, opts[key])
    for key, flag in (("sockets_per_node", "--sockets-per-node"), ("cpus_per_gpu", "--cpus-per-gpu")):
        if key in opts:
            if not str(opts[key]).isdigit():
                raise _err(f'{prog}: error: Invalid numeric value "{opts[key]}" for {flag}.')
            spec[key] = int(opts[key])
    if "ntasks_per_gpu" in opts:
        if not str(opts["ntasks_per_gpu"]).isdigit():
            raise _err(f'{prog}: error: Invalid numeric value "{opts["ntasks_per_gpu"]}" for --ntasks-per-gpu.')
        spec["ntasks_per_gpu"] = int(opts["ntasks_per_gpu"])
    if "time_min" in opts:
        tmin = time_arg(opts["time_min"], f"{prog}: error: Invalid --time-min specification")
        if tmin is None or (time_limit is not None and tmin > time_limit):
            raise fail("Requested time limit is invalid (missing or exceeds some limit)")
        spec["time_min"] = tmin
    if "nice" in opts and not re.fullmatch(r"-?\d+", str(opts["nice"])):
        raise _err(f"{prog}: error: Invalid --nice value: {opts['nice']}")
    return spec


def _verbose_block(prog, opts):
    """What ``-v`` prints before the submission: the options that were set."""
    if not (opts.get("v") or opts.get("verbose")):
        return ""
    rows = [f"{prog}: defined options", f"{prog}: -------------------- --------------------"]
    for key, val in sorted(opts.items()):
        rows.append(f"{prog}: {key.replace('_', '-'):<20}: {1 if val is True else val}")
    rows += [f"{prog}: -------------------- --------------------", f"{prog}: end of defined options"]
    return "\n".join(rows) + "\n"


def _new_job(state, job):
    s = state.slurm
    job["id"] = s.next_job_id
    s.next_job_id += 1
    s.jobs.append(job)
    j = s.jobs[-1]
    _job_defaults(state, j)
    j.default_mem = list(_default_mem(state, j))
    return j


def _parse_script(text):
    toks = []
    lines = text.splitlines()
    for line in lines[1:]:
        st = line.strip()
        if not st:
            continue
        if st.startswith("#SBATCH"):
            body = st[len("#SBATCH") :].split(" #", 1)[0]
            try:
                toks += shlex.split(body)
            except ValueError:
                toks += body.split()
            continue
        if st.startswith("#"):
            continue
        break
    return toks


@command("sbatch", help="submit a batch script: sbatch [opts] script.sh | sbatch --wrap='cmd'")
def sbatch(ctx, args):
    if _has(args, "-V", "--version"):
        return f"slurm {SLURM_VERSION}\n"
    state = ctx.state
    _ensure(state)
    pyxis = _pyxis_client_ok(ctx.node)
    opts, rest = parse_job_opts("sbatch", args, pyxis=pyxis)
    if "wrap" in opts:
        script = "#!/bin/sh\n# This script was created by sbatch --wrap.\n\n" + opts["wrap"] + "\n"
        cmd_path = "(null)"
    elif rest:
        path = rest[0]
        if not path.startswith("/"):
            path = ctx.cwd.rstrip("/") + "/" + path
        script = ctx.node.files.get(path)
        if script is None:
            script = ctx.node.files.get(rest[0])
        if script is None:
            raise _err(f"sbatch: error: Unable to open file {rest[0]}")
        cmd_path = path
        if not script.startswith("#!"):
            raise _err(
                "sbatch: error: This does not look like a batch script.  The first\n"
                "sbatch: error: line must start with #! followed by the path to an interpreter.\n"
                "sbatch: error: For instance: #!/bin/sh"
            )
        sopts, _ = parse_job_opts("sbatch", _parse_script(script), stop_at_positional=False, pyxis=pyxis)
        sopts.update(opts)
        opts = sopts
    elif ctx.stdin.strip():  # sbatch <<EOF ... EOF, or a script piped in
        script = ctx.stdin if ctx.stdin.startswith("#!") else "#!/bin/bash\n" + ctx.stdin
        cmd_path = "(null)"
        sopts, _ = parse_job_opts("sbatch", _parse_script(script), stop_at_positional=False, pyxis=pyxis)
        sopts.update(opts)
        opts = sopts
    else:
        raise _err("sbatch: error: Batch script is empty!")
    _need_ctl(ctx, "sbatch: error: Batch job submission failed: ")
    _begin(ctx)
    job = _build_job(ctx, "sbatch", opts)
    if not job["name"]:
        job["name"] = "wrap" if "wrap" in opts else (cmd_path.rsplit("/", 1)[-1] if rest else "sbatch")
    job.update({"script": script, "command": cmd_path, "batch": True, "duration": _script_duration(script)})
    if opts.get("test-only"):
        return _test_only(ctx, "sbatch", job)
    j = _new_job(state, job)
    out_pat = opts.get("output") or ("slurm-%A_%a.out" if j.get("array_tasks") else "slurm-%j.out")
    j.stdout = _expand_pattern(out_pat, j)
    j.stdout_pattern = out_pat
    if not j.stdout.startswith("/"):
        j.stdout = j.work_dir.rstrip("/") + "/" + j.stdout
    schedule(state)
    if opts.get("wait") or opts.get("W"):  # sbatch --wait returns when the job has finished
        from .scheduler import tick

        for _ in range(400):
            if j.state in TERMINAL and not j.get("array_tasks"):
                break
            tick(ctx)
    if opts.get("parsable"):
        return _verbose_block("sbatch", opts) + f"{j.id}\n"
    return _verbose_block("sbatch", opts) + f"Submitted batch job {j.id}\n"


def _script_duration(script: str) -> int:
    """How long the batch script runs: a ``sleep N`` says so, anything else takes the default."""
    for line in script.splitlines():
        m = re.search(r"\bsleep\s+(\d+)([smh]?)\b", line.split("#", 1)[0])
        if m:
            return int(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]
    return BATCH_DEFAULT_DURATION


def _test_only(ctx, prog, job) -> str:
    """``--test-only``: report where and when the job could start, without submitting it."""
    from ...core.state import wrap
    from .scheduler import _job_defaults, _place, _static_reason

    state = ctx.state
    probe = wrap(dict(job))
    probe["id"] = state.slurm.next_job_id
    _job_defaults(state, probe)
    reason = _static_reason(state, probe)
    if reason:
        raise _err(
            f"{prog}: error: Batch job submission failed: {reason}" if prog == "sbatch" else f"{prog}: error: {reason}"
        )
    chosen = _place(state, probe)
    when = _ts(state, state.slurm.clock) if chosen else _ts(state, state.slurm.clock + 3600)
    nodes = (
        compress_hostlist(list(chosen))
        if chosen
        else compress_hostlist(_part_nodes(state, probe.partition)[: probe.num_nodes])
    )
    cpus = probe.num_nodes * probe.ntasks_per_node * probe.cpus_per_task
    return f"{prog}: Job {probe.id} to start at {when} using {cpus} processors on nodes {nodes} in partition {probe.partition}\n"


def _alloc_now(ctx, prog, opts):
    """Create a job and try to start it immediately (srun/salloc). Returns (job, waited_msg)."""
    state = ctx.state
    job = _build_job(ctx, prog, opts)
    job["batch"] = False
    j = _new_job(state, job)
    schedule(state)
    if j.state == "RUNNING":
        return j, ""
    reason = j.reason
    _finish(state, j, "CANCELLED", "0:0")
    if reason == "BadConstraints":
        raise _err(f"{prog}: error: Unable to allocate resources: Requested node configuration is not available")
    j.reason = reason
    raise _err(
        f"{prog}: job {j.id} queued and waiting for resources\n"
        f"{prog}: (pending reason: {reason}; srun would block here -- the simulator gave up)\n"
        f"^C{prog}: Job allocation {j.id} has been revoked."
    )


@command("srun", help="run a parallel job/step: srun [-N n] [--gres=gpu:N] [--container-image=reg#img:tag] cmd")
def srun(ctx, args):
    if _has(args, "-V", "--version"):
        return f"slurm {SLURM_VERSION}\n"
    state = ctx.state
    _ensure(state)
    pyxis = _pyxis_client_ok(ctx.node)
    opts, cmd = parse_job_opts("srun", args, pyxis=pyxis)
    if not cmd:
        if str(opts.get("mpi", "")).lower() == "list":
            return "MPI plugin types are...\n\tnone\n\tpmi2\n\tpmix\n\tcray_shasta\nspecific pmix plugin versions available: pmix_v4\n"
        raise _err("srun: fatal: No command given to execute.")
    _need_ctl(ctx, "srun: error: Unable to allocate resources: ")
    # inside an allocation (batch script, salloc, srun --pty) -> job step
    parent = _CURRENT[-1] if _CURRENT else None
    if parent is None and ctx.node.env.get("SLURM_JOB_ID"):
        parent = _find_job(state, ctx.node.env["SLURM_JOB_ID"])
        if parent is None or parent.state != "RUNNING":
            raise _err(
                f"srun: error: Unable to confirm allocation for job {ctx.node.env['SLURM_JOB_ID']}: "
                f"Invalid job id specified\nsrun: Check SLURM_JOB_ID environment variable. "
                f"Expired or invalid job {ctx.node.env['SLURM_JOB_ID']}"
            )
    if parent is not None:
        if "gres" in opts or "gpus_per_node" in opts:
            _, want = parse_gres_spec(opts.get("gres") or "gpu:" + opts.get("gpus_per_node", "0"))
            if want > parent.gpus_per_node:
                raise _err(
                    f"srun: error: Unable to create step for job {parent.id}: Invalid generic resource (gres) specification"
                )
        if opts.get("pty") and cmd[0] in SHELLS:
            # a shell step inside an allocation: move onto the job's node; exit returns here, the job stays
            host = parent.nodelist[0]
            ctx.host_stack.append(ctx.host)
            ctx.host = host
            state.nodes[host].env["SLURM_JOB_ID"] = str(parent.id)
            return f"(interactive shell on {host} inside job {parent.id}; type 'exit' to return to the allocation)\n"
        out, rc = _run_step(ctx, parent, opts, cmd)
        if rc:
            raise CommandError(out, rc)
        return out + ("\n" if out else "")
    _begin(ctx)
    j, _ = _alloc_now(ctx, "srun", opts)
    j.name = opts.get("name") or cmd[0].rsplit("/", 1)[-1]
    j.command = " ".join(cmd)
    if opts.get("container_image"):
        j.container_image = opts["container_image"]
    if opts.get("pty") and cmd[0] in SHELLS:
        host = j.nodelist[0]
        ctx.host_stack.append(ctx.host)
        ctx.host = host
        state.nodes[host].env["SLURM_JOB_ID"] = str(j.id)
        j.pty = True
        j.pty_depth = len(ctx.host_stack)
        return f"(interactive shell on {host} inside job {j.id}; type 'exit' to end the allocation)\n"
    out, rc = _run_step(ctx, j, opts, cmd)
    _finish(state, j, "FAILED" if rc else "COMPLETED", f"{rc}:0")
    schedule(state)
    if rc:
        raise CommandError(out, rc)
    if opts.get("output"):  # -o sends the tasks' stdout to a file instead of the terminal
        path = _expand_pattern(opts["output"], j)
        path = path if path.startswith("/") else j.work_dir.rstrip("/") + "/" + path
        ctx.node.files[path] = out + ("\n" if out else "")
        out = ""
    return _verbose_block("srun", opts) + out + ("\n" if out else "")


@command("salloc", help="get an interactive allocation: salloc -N1 --gres=gpu:8 (then srun ...; scancel to release)")
def salloc(ctx, args):
    state = ctx.state
    _ensure(state)
    opts, cmd = parse_job_opts("salloc", args, pyxis=_pyxis_client_ok(ctx.node))
    if opts.get("pty"):  # an srun option; salloc opens no terminal itself
        raise _err("salloc: unrecognized option '--pty'\nTry \"salloc --help\" for more information")
    _need_ctl(ctx, "salloc: error: Unable to allocate resources: ")
    _begin(ctx)
    if opts.get("test-only"):
        return _test_only(ctx, "salloc", _build_job(ctx, "salloc", opts))
    j, _ = _alloc_now(ctx, "salloc", opts)
    j.name = opts.get("name") or "interactive"
    ctx.node.env["SLURM_JOB_ID"] = str(j.id)
    ctx.node.env["SLURM_JOB_NODELIST"] = compress_hostlist(j.nodelist)
    ctx.node.env["SLURM_NNODES"] = str(j.num_nodes)
    granted = _verbose_block("salloc", opts) + (
        ""
        if opts.get("Q") or opts.get("quiet")
        else f"salloc: Granted job allocation {j.id}\nsalloc: Nodes {compress_hostlist(j.nodelist)} are ready for job\n"
    )
    if cmd:  # salloc ... <command>: run it inside the allocation, then release
        from ...core.shell import Shell

        sh = Shell(state, ctx.host)
        sh.ctx.executed = ctx.executed
        sh.ctx.outputs = ctx.outputs
        sh.ctx.results = ctx.results
        out = sh.run(" ".join(shlex.quote(a) for a in cmd))
        _release_allocation(ctx, j)
        return (
            granted
            + out
            + ("\n" if out and not out.endswith("\n") else "")
            + f"salloc: Relinquishing job allocation {j.id}\n"
        )
    # the allocation is a shell: `exit` leaves it and releases the job
    ctx.alloc_stack.append((j.id, len(ctx.host_stack)))
    return granted


def _release_allocation(ctx, j) -> None:
    state = ctx.state
    if j.state not in TERMINAL:
        _finish(state, j, "COMPLETED", "0:0")
    for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES"):
        ctx.node.env.pop(key, None)
    schedule(state)


def leave_allocation(ctx) -> str | None:
    """Called by ``exit``: when the learner is in an salloc shell at this ssh depth, release it."""
    if not ctx.alloc_stack or ctx.alloc_stack[-1][1] != len(ctx.host_stack):
        return None
    jid, _ = ctx.alloc_stack.pop()
    j = _find_job(ctx.state, jid)
    if j is not None:
        _release_allocation(ctx, j)
    return f"salloc: Relinquishing job allocation {jid}\n"


@command("scancel", help="cancel jobs: scancel ID... | -u USER | -p PART | -t STATE | -n NAME")
def scancel(ctx, args):
    state = ctx.state
    _need_ctl(ctx, "scancel: error: Kill job error: ")
    _begin(ctx)
    user = _opt(args, "-u", "--user")
    if _has(args, "--me"):
        user = _USER[-1]
    part = _opt(args, "-p", "--partition")
    st = _opt(args, "-t", "--state")
    name = _opt(args, "-n", "--name", "--jobname")
    signal = _opt(args, "-s", "--signal")
    if (
        signal is not None
        and signal.upper().removeprefix("SIG") not in ("KILL", "TERM", "INT", "HUP", "USR1", "USR2", "STOP", "CONT")
        and not signal.isdigit()
    ):
        raise _err(f"scancel: error: Invalid job signal: {signal}")
    skip = set()
    for i, a in enumerate(args):
        if a in ("-u", "--user", "-p", "--partition", "-t", "--state", "-n", "--name", "--jobname", "-s", "--signal"):
            skip.add(i + 1)
    ids = [a for i, a in enumerate(args) if not a.startswith("-") and i not in skip]
    out = []
    targets = []
    if ids:
        for i in ids:
            for one in i.split(","):
                j = _find_job(state, one)
                if j is None or j.state in TERMINAL:
                    out.append(
                        f"scancel: error: Kill job error on job id {one}: "
                        f"{'Invalid job id specified' if j is None else 'Job/step already completing or completed'}"
                    )
                    continue
                targets.append(j)
                if j.get("array_tasks") is not None and "_" not in one:
                    pass
    else:
        if not (user or part or st or name):
            raise _err("scancel: error: No job identification provided")
        for j in state.slurm.jobs:
            if j.state in TERMINAL:
                continue
            if user and j.user != user:
                continue
            if part and j.partition != part:
                continue
            if st and STATE_CODES.get(j.state) != st.upper() and j.state != st.upper():
                continue
            if name and j.name != name:
                continue
            targets.append(j)
    gentle = signal is not None and signal.upper().removeprefix("SIG") not in ("KILL", "9")
    denied = [j for j in targets if j.user != _USER[-1] and not is_admin()]
    if denied:
        raise _err(
            "\n".join(f"scancel: error: Kill job error on job id {j.id}: Access/permission denied" for j in denied)
        )
    expanded = []
    for j in targets:  # cancelling an array master cancels every task that has not finished
        expanded.append(j)
        if j.get("array_tasks") is not None:
            j.array_tasks = []
            expanded.extend(c for c in state.slurm.jobs if c.get("array_job_id") == j.id and c.state not in TERMINAL)
    targets = expanded
    for j in targets:
        if j.user != "root" and _USER[-1] == "root":
            audit.record(state, audit.CANCEL_USER_JOB, ctx.host, "scancel", f"{j.id} ({j.user})")
        if gentle and j.state == "RUNNING":  # a signal other than KILL is delivered to the steps, the job goes on
            j.setdefault("signals", []).append(signal.upper())
            continue
        _finish(state, j, "CANCELLED", "0:0")
    schedule(state)
    if out:
        raise CommandError("\n".join(out), 1)
    return ""
