"""The simulated slurmctld: lazy state init, job placement, QOS/partition limits and the 30 s clock tick."""

from __future__ import annotations

import math

from .affinity import _alloc_cores, _local_free_cores, _numa_local
from .common import (
    _GUARD,
    CONF,
    HEAD,
    NOT_AVAIL_TEXT,
    TERMINAL,
    TICK,
    _err,
    _find_job,
    compress_hostlist,
    expand_hostlist,
    parse_gres_spec,
    parse_time,
    parse_tres,
)
from .conf import _enforce, _node_gres_from_conf, _param, _partition_from_conf, parse_slurm_conf
from .fairshare import _fs_factor
from .munge import _munge_ok
from .nodes import (
    _conf_gpus,
    _derive_state,
    _do_slurm_reboot,
    _finish,
    _health_check,
    _node_defaults,
    _refresh,
    _register,
    _usable,
    _used_gpu_idx,
)
from .steps import _run_batch


def _ensure(state):
    """Lazily fill in everything this module needs."""
    s = state.slurm
    s.setdefault("clock", 0)
    if "config" not in s:
        conf = parse_slurm_conf(state.nodes[HEAD].files.get(CONF, "") if HEAD in state.nodes else "")
        s.config = conf
        for n, attrs in conf["nodes"].items():
            if n in s.nodes:
                s.nodes[n].gres = _node_gres_from_conf(state, attrs)
        for p in conf["part_order"]:
            if p not in s.partitions:
                s.partitions[p] = _partition_from_conf(p, conf["partitions"][p], conf["nodes"])
    for p in s.partitions.values():
        p.setdefault("qos", "")
        p.setdefault("max_nodes", "UNLIMITED")
        p.setdefault("default_time", "NONE")
        p.setdefault("priority_tier", 1)
        p.setdefault("allow_qos", "ALL")
        p.setdefault("allow_accounts", "ALL")
        p.setdefault("state", "UP")
        p.setdefault("default", False)
        p.setdefault("max_time", "UNLIMITED")
    for q in s.qos.values():
        for k, v in (
            ("priority", 0),
            ("max_tres_per_user", ""),
            ("max_tres_per_job", ""),
            ("max_jobs_per_user", None),
            ("max_wall", None),
            ("grp_tres", ""),
            ("flags", ""),
            ("preempt", ""),
            ("description", q.get("name", "")),
        ):
            q.setdefault(k, v)
    for u in s.users.values():
        u.setdefault("default_account", "root")
        u.setdefault("accounts", [u.default_account])
        u.setdefault("qos", ["normal"])
        u.setdefault("default_qos", "")
        u.setdefault("fairshare_factor", 0.5)
    for a in s.accounts.values():
        a.setdefault("users", [])
        a.setdefault("fairshare", 1)
        a.setdefault("parent", "root" if a.get("name") != "root" else "")
        a.setdefault("description", a.get("name", ""))
        a.setdefault("organization", a.get("name", ""))
    for sn in s.nodes.values():
        _node_defaults(state, sn)
    for j in s.jobs:
        _job_defaults(state, j)
    for name, sn in s.nodes.items():
        if sn.get("_reregister"):
            sn["_reregister"] = False
            node = state.nodes.get(name)
            if node is not None and node.get("up", True) and node.services.get("slurmd") == "active":
                _register(state, name, log=False)


def _default_partition(state):
    for p in state.slurm.partitions.values():
        if p.get("default"):
            return p.name
    return None


def _job_defaults(state, j):
    s = state.slurm
    if "id" not in j:
        j["id"] = s.next_job_id
        s.next_job_id += 1
    user = j.setdefault("user", "root")
    acct = s.users.get(user, {}).get("default_account", "root")
    for k, v in (
        ("name", "job"),
        ("account", acct),
        ("partition", _default_partition(state) or "defq"),
        ("qos", "normal"),
        ("state", "PENDING"),
        ("reason", "None"),
        ("num_nodes", 1),
        ("ntasks_per_node", 1),
        ("cpus_per_task", 1),
        ("gpus_per_node", 0),
        ("gpu_type", None),
        ("time_limit", None),
        ("req_nodes", []),
        ("exc_nodes", []),
        ("nodelist", []),
        ("alloc_gpus", {}),
        ("submit_time", s.get("clock", 0)),
        ("start_time", None),
        ("end_time", None),
        ("duration", None),
        ("command", ""),
        ("script", ""),
        ("work_dir", "/root"),
        ("stdout", ""),
        ("container_image", ""),
        ("exit_code", "0:0"),
        ("batch", True),
        ("held", False),
        ("dependency", ""),
        ("exclusive", False),
        ("restarts", 0),
        ("steps", []),
        ("priority_fixed", False),
        ("nice", 0),
        ("submit_host", HEAD),
        ("gres_flags", ""),
        ("cpu_bind", ""),
        ("alloc_cores", {}),
        ("numa_local", None),
    ):
        j.setdefault(k, v)
    if isinstance(j.nodelist, str):
        j.nodelist = expand_hostlist(j.nodelist)
    if isinstance(j.req_nodes, str):
        j.req_nodes = expand_hostlist(j.req_nodes)
    if isinstance(j.exc_nodes, str):
        j.exc_nodes = expand_hostlist(j.exc_nodes)
    j.state = str(j.state).upper()
    if isinstance(j.time_limit, str):
        j.time_limit = parse_time(j.time_limit)
    if j.state == "RUNNING":
        if j.start_time is None:
            j.start_time = max(0, s.get("clock", 0) - 3600)
        if j.nodelist and not j.alloc_gpus and j.gpus_per_node:
            alloc = {}
            for n in j.nodelist:
                used = _used_gpu_idx(state, n, exclude=j.id)
                alloc[n] = [i for i in range(8) if i not in used][: j.gpus_per_node]
            j.alloc_gpus = alloc
        if j.nodelist and not j.alloc_cores:
            j.alloc_cores = {
                n: _alloc_cores(state, j, n, (j.alloc_gpus or {}).get(n, [])) for n in j.nodelist if n in s.nodes
            }
    if "priority" not in j:
        j.priority = _priority(state, j)
    elif not j.get("priority_fixed") and j.get("_prio_auto", True) is False:
        pass


def _priority(state, j) -> int:
    if j.get("priority_fixed"):
        return int(j.get("priority", 0))
    s = state.slurm
    w_age = int(_param(state, "PriorityWeightAge", 0) or 0)
    w_fs = int(_param(state, "PriorityWeightFairshare", 0) or 0)
    w_qos = int(_param(state, "PriorityWeightQOS", 0) or 0)
    w_part = int(_param(state, "PriorityWeightPartition", 0) or 0)
    age = min(1.0, max(0, s.get("clock", 0) - j.get("submit_time", 0)) / (7 * 86400))
    fs = _fs_factor(state, j.get("user"), j.get("account"))
    maxq = max([int(q.get("priority", 0)) for q in s.qos.values()] + [0])
    qf = (int(s.qos.get(j.get("qos"), {}).get("priority", 0)) / maxq) if maxq else 0.0
    part = s.partitions.get(j.get("partition"), {})
    pf = 1.0 if part else 0.0
    return int(w_age * age + w_fs * fs + w_qos * qf + w_part * pf) - int(j.get("nice", 0))


def _alloc_cpu_count(state, j) -> int:
    """CPUs actually allocated: whole nodes for --exclusive jobs, otherwise tasks x cpus-per-task."""
    req = j.num_nodes * j.ntasks_per_node * j.cpus_per_task
    if j.get("exclusive") and j.get("start_time") is not None and j.nodelist:
        return sum(int(state.slurm.nodes[n].get("cpus", 224)) if n in state.slurm.nodes else 224 for n in j.nodelist)
    return req


def _job_gpus(j) -> int:
    return int(j.gpus_per_node) * int(j.num_nodes)


def _part_nodes(state, pname) -> list[str]:
    p = state.slurm.partitions.get(pname)
    if not p:
        return []
    return [n for n in expand_hostlist(p.nodes) if n in state.slurm.nodes]


def _effective_qos(state, j) -> list:
    s = state.slurm
    out = []
    p = s.partitions.get(j.partition)
    if p and p.get("qos") and p.qos in s.qos:
        out.append(s.qos[p.qos])
    if j.qos in s.qos and (not out or out[0].name != j.qos):
        out.append(s.qos[j.qos])
    return out


def _qos_reason(state, j):
    if "limits" not in _enforce(state):
        return None
    s = state.slurm
    for q in _effective_qos(state, j):

        def in_q(o, q=q):
            return q in _effective_qos(state, o) or q.name == o.qos

        running = [o for o in s.jobs if o.state == "RUNNING" and in_q(o)]
        per_job = parse_tres(q.get("max_tres_per_job"))
        if "gres/gpu" in per_job and _job_gpus(j) > per_job["gres/gpu"]:
            return "QOSMaxGRESPerJob"
        if "node" in per_job and j.num_nodes > per_job["node"]:
            return "QOSMaxNodePerJobLimit"
        per_user = parse_tres(q.get("max_tres_per_user"))
        mine = [o for o in running if o.user == j.user]
        if "gres/gpu" in per_user and sum(_job_gpus(o) for o in mine) + _job_gpus(j) > per_user["gres/gpu"]:
            return "QOSMaxGRESPerUser"
        if "node" in per_user and sum(o.num_nodes for o in mine) + j.num_nodes > per_user["node"]:
            return "QOSMaxNodePerUserLimit"
        mj = q.get("max_jobs_per_user")
        if mj not in (None, "", -1) and len(mine) >= int(mj):
            return "QOSMaxJobsPerUserLimit"
        mw = parse_time(q.get("max_wall")) if q.get("max_wall") not in (None, "") else None
        if mw is not None and (j.time_limit is None or j.time_limit > mw):
            return "QOSMaxWallDurationPerJobLimit"
        grp = parse_tres(q.get("grp_tres"))
        if "gres/gpu" in grp and sum(_job_gpus(o) for o in running) + _job_gpus(j) > grp["gres/gpu"]:
            return "QOSGrpGRES"
    return None


def _dependency_reason(state, j):
    dep = j.get("dependency") or ""
    if not dep:
        return None
    for clause in dep.split(","):
        if ":" not in clause:
            continue
        kind, ids = clause.split(":", 1)
        for i in ids.split(":"):
            other = _find_job(state, i)
            if other is None:
                continue
            if kind == "afterok":
                if other.state == "COMPLETED" and other.exit_code == "0:0":
                    continue
                if other.state in TERMINAL:
                    return "DependencyNeverSatisfied"
                return "Dependency"
            if kind == "afterany" and other.state not in TERMINAL:
                return "Dependency"
            if kind == "afternotok":
                if other.state in TERMINAL and other.state != "COMPLETED":
                    continue
                if other.state == "COMPLETED":
                    return "DependencyNeverSatisfied"
                return "Dependency"
            if kind == "after" and other.state == "PENDING":
                return "Dependency"
    return None


def _static_reason(state, j):
    s = state.slurm
    if j.held:
        return j.get("hold_reason") or "JobHeldAdmin"
    r = _dependency_reason(state, j)
    if r:
        return r
    p = s.partitions.get(j.partition)
    if p is None:
        return "PartitionDown"
    if p.state == "DOWN":
        return "PartitionDown"
    if p.state == "INACTIVE":
        return "PartitionInactive"
    maxt = parse_time(p.max_time)
    if maxt is not None and j.time_limit is not None and j.time_limit > maxt:
        return "PartitionTimeLimit"
    pnodes = _part_nodes(state, j.partition)
    maxn = p.get("max_nodes", "UNLIMITED")
    if j.num_nodes > len(pnodes) or (str(maxn).isdigit() and j.num_nodes > int(maxn)):
        return "PartitionNodeLimit"
    if j.gpus_per_node and j.gpus_per_node > max([_conf_gpus(s.nodes[n]) for n in pnodes] + [0]):
        return "BadConstraints"
    return _qos_reason(state, j)


def _fits(state, j, name) -> list | None:
    """GPU indices the job would get on this node, or None if it does not fit."""
    sn = state.slurm.nodes[name]
    running = [o for o in state.slurm.jobs if o.state == "RUNNING" and name in o.nodelist]
    if j.exclusive and running:
        return None
    if any(o.exclusive for o in running):
        return None
    ctype, ccount = parse_gres_spec(sn.gres)
    if j.gpu_type and ctype and j.gpu_type != ctype:
        return None
    used = _used_gpu_idx(state, name)
    free = [i for i in range(ccount) if i not in used]
    need = ccount if j.exclusive else j.gpus_per_node
    if len(free) < need or (j.gpus_per_node and j.gpus_per_node > len(free)):
        return None
    cpus = sn.cpus if j.exclusive else j.ntasks_per_node * j.cpus_per_task
    if sn.alloc_cpus + cpus > sn.cpus:
        return None
    if "enforce-binding" in str(j.get("gres_flags", "")) and j.gpus_per_node and not j.exclusive:
        if _local_free_cores(state, j, name, free[:need]) < math.ceil(cpus / 2):
            return None
    return free[:need]


def _place(state, j):
    cands = [n for n in _part_nodes(state, j.partition) if n not in j.exc_nodes]
    cands.sort(key=lambda n: (state.slurm.nodes[n].get("weight", 1), n))
    chosen = {}
    for n in j.req_nodes:
        if n not in cands or not _usable(state, n):
            return None
        g = _fits(state, j, n)
        if g is None:
            return None
        chosen[n] = g
    for n in cands:
        if len(chosen) >= j.num_nodes:
            break
        if n in chosen or not _usable(state, n):
            continue
        g = _fits(state, j, n)
        if g is not None:
            chosen[n] = g
    return chosen if len(chosen) >= j.num_nodes else None


def _not_avail_reason(state, j):
    bad_req = [n for n in j.req_nodes if not _usable(state, n)]
    if bad_req:
        return "ReqNodeNotAvail, UnavailableNodes:" + compress_hostlist(bad_req)
    usable = [n for n in _part_nodes(state, j.partition) if _usable(state, n) and n not in j.exc_nodes]
    if len(usable) < j.num_nodes:
        return NOT_AVAIL_TEXT
    return None


def _start(state, j, chosen, add_procs=True):
    j.state = "RUNNING"
    j.reason = "None"
    j.nodelist = list(chosen)
    j.alloc_gpus = {n: list(g) for n, g in chosen.items()}
    j.alloc_cores = {}
    for n, g in chosen.items():
        j.alloc_cores[n] = _alloc_cores(state, j, n, g)
    loc = [_numa_local(j, n) for n in chosen]
    j.numa_local = None if any(x is None for x in loc) or not loc else all(loc)
    j.start_time = state.slurm.clock
    if j.batch and add_procs:
        pname = _proc_name(j)
        for n, idx in chosen.items():
            node = state.nodes.get(n)
            if node is None:
                continue
            for i in idx:
                if i < len(node.gpus):
                    g = node.gpus[i]
                    g.processes.append(
                        {
                            "pid": 30000 + j.id % 10000 * 3 + i,
                            "name": pname,
                            "used_mib": 61440,
                            "user": j.user,
                            "slurm_job": j.id,
                        }
                    )
                    g.memory_used_mib = min(g.memory_total_mib, g.memory_used_mib + 61440)
                    g.utilization = 97
    for n in chosen:
        _derive_state(state, state.slurm.nodes[n])
    if j.batch and j.script and add_procs:
        _run_batch(state, j)


def _proc_name(j):
    for line in (j.script or j.command or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        for t in toks:
            if t.startswith(("python", "torchrun")):
                return t
        return toks[0]
    return "bash"


def schedule(state):
    s = state.slurm
    _refresh(state)
    pend = [j for j in s.jobs if j.state == "PENDING"]
    for j in pend:
        j.priority = _priority(state, j)
    pend.sort(key=lambda j: (-s.partitions.get(j.partition, {}).get("priority_tier", 1), -j.priority, j.id))
    blocked = set()
    for j in pend:
        r = _static_reason(state, j)
        if r:
            j.reason = r
            continue
        chosen = _place(state, j)
        if chosen:
            _start(state, j, chosen)
            continue
        r = _not_avail_reason(state, j)
        if r:
            j.reason = r
            continue
        j.reason = "Priority" if j.partition in blocked else "Resources"
        blocked.add(j.partition)
    _refresh(state)


def tick(ctx):
    state = ctx.state
    _ensure(state)
    if _GUARD["depth"]:
        return
    s = state.slurm
    s.clock += TICK
    for j in list(s.jobs):
        if j.state != "RUNNING":
            continue
        if j.duration is not None and j.start_time is not None and s.clock >= j.start_time + j.duration:
            _finish(state, j, "COMPLETED", j.get("final_exit", "0:0") or "0:0")
            if j.get("final_state"):
                j.state = j.final_state
        elif (
            j.time_limit is not None
            and j.start_time is not None
            and s.clock >= j.start_time + j.time_limit * 60
            and j.duration is None
            and j.get("enforce_timeout")
        ):
            _finish(state, j, "TIMEOUT", "0:0")
        elif j.get("pty") and len(ctx.host_stack) < j.get("pty_depth", 0):
            _finish(state, j, "COMPLETED", "0:0")
    for name, sn in s.nodes.items():
        if sn.get("reboot_pending") and not any(j.state == "RUNNING" and name in j.nodelist for j in s.jobs):
            _do_slurm_reboot(ctx, name)
    _health_check(state)
    schedule(state)


def _controller_ok(state) -> bool:
    head = state.nodes.get(HEAD)
    if head is None:
        return state.slurm.get("controller_up", True)
    return (
        state.slurm.get("controller_up", True)
        and head.services.get("slurmctld", "active") == "active"
        and head.get("up", True)
    )


def _need_ctl(ctx, msg):
    if not _controller_ok(ctx.state):
        raise _err(msg + "Unable to contact slurm controller (connect failure)")
    prog = (
        msg.split(":", 1)[0]
        if msg.split(":", 1)[0] in ("sbatch", "srun", "salloc", "scancel", "sprio", "sshare")
        else ""
    )
    lead = f"{prog}: error: " if prog else ""
    node = ctx.state.nodes.get(ctx.host)
    if node is not None and "munge" in node.services and node.services.get("munge") != "active":
        raise _err(
            f'{lead}Munge encode failed: Failed to access "/run/munge/munge.socket.2": No such file or directory\n'
            + msg
            + "Protocol authentication error"
        )
    if not _munge_ok(ctx.state, ctx.host):
        raise _err(msg + "Protocol authentication error")


def _begin(ctx):
    _ensure(ctx.state)
    tick(ctx)
