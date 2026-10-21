"""scontrol (show/update/reconfigure/reboot/hold) and the slurmd diagnostic flags."""

from __future__ import annotations

from .. import command
from .affinity import _gpu_core_map
from .common import (
    GRES_CONF,
    HEAD,
    PLUGSTACK,
    SLURM_VERSION,
    TERMINAL,
    _err,
    _find_job,
    _ranges,
    _ts,
    compress_hostlist,
    expand_hostlist,
    fmt_elapsed,
    fmt_limit_upper,
    parse_gres_spec,
    parse_time,
)
from .conf import _parse_gres_conf, _partition_from_conf
from .nodes import (
    _conf_gpus,
    _derive_state,
    _do_slurm_reboot,
    _finish,
    _refresh,
    _scontrol_node_state,
    _set_drain,
    _used_gpu_idx,
    apply_config,
)
from .scheduler import _alloc_cpu_count, _begin, _controller_ok, _ensure, _job_gpus, _need_ctl, _part_nodes, schedule
from .status import _job_tres_per_node


def _show_node(state, sn, details=False):
    node = state.nodes.get(sn.name)
    gres = sn.gres if sn.gres != "(null)" else "(null)"
    if gres != "(null)":
        gres += "(S:0-1)"
    gt, gc = parse_gres_spec(sn.gres)
    used_idx = sorted(_used_gpu_idx(state, sn.name))
    cfg_tres = f"cpu={sn.cpus},mem={sn.real_memory}M,billing={sn.cpus}" + (f",gres/gpu={gc}" if gc else "")
    used_g = len(used_idx)
    alloc_tres = (f"cpu={sn.alloc_cpus}" + (f",gres/gpu={used_g}" if used_g else "")) if sn.alloc_cpus else ""
    parts = [p.name for p in state.slurm.partitions.values() if sn.name in expand_hostlist(p.nodes)]
    lines = [
        f"NodeName={sn.name} Arch=x86_64 CoresPerSocket=56 ",
        f"   CPUAlloc={sn.alloc_cpus} CPUEfctv={sn.cpus} CPUTot={sn.cpus} CPULoad={'0.42' if sn.alloc_cpus else '0.01'}",
        f"   AvailableFeatures={sn.features or '(null)'}",
        f"   ActiveFeatures={sn.features or '(null)'}",
        f"   Gres={gres}",
    ]
    if details:
        idx = ",".join(_ranges(used_idx)) if used_idx else "N/A"
        lines.append(f"   GresUsed={'gpu:' + (gt + ':' if gt else '') + str(used_g)}(IDX:{idx})")
    lines += [
        f"   NodeAddr={sn.name} NodeHostName={sn.name} Version={SLURM_VERSION}",
        f"   OS=Linux {node.kernel if node else '5.15.0'} #43-Ubuntu SMP ",
        f"   RealMemory={sn.real_memory} AllocMem=0 FreeMem=2011234 Sockets=2 Boards=1",
        f"   State={_scontrol_node_state(sn)} ThreadsPerCore=2 TmpDisk=0 Weight={sn.get('weight', 1)} Owner=N/A MCS_label=N/A",
        f"   Partitions={','.join(parts)} ",
        f"   BootTime={sn.boot_time} SlurmdStartTime={sn.boot_time}",
        f"   LastBusyTime={_ts(state)} ResumeAfterTime=None",
        f"   CfgTRES={cfg_tres}",
        f"   AllocTRES={alloc_tres}",
        "   CapWatts=n/a",
        "   CurrentWatts=0 AveWatts=0",
        "   ExtSensorsJoules=n/a ExtSensorsWatts=0 ExtSensorsTemp=n/a",
    ]
    if sn.reason:
        lines.append(f"   Reason={sn.reason} [{sn.reason_user}@{sn.reason_time}]")
    return "\n".join(lines) + "\n"


def _show_partition(state, p):
    nodes = _part_nodes(state, p.name)
    cpus = sum(state.slurm.nodes[n].cpus for n in nodes)
    mem = sum(state.slurm.nodes[n].real_memory for n in nodes)
    gpus = sum(_conf_gpus(state.slurm.nodes[n]) for n in nodes)
    mt = parse_time(p.max_time)
    tres = f"cpu={cpus},mem={mem}M,node={len(nodes)},billing={cpus}" + (f",gres/gpu={gpus}" if gpus else "")
    return (
        "\n".join(
            [
                f"PartitionName={p.name}",
                f"   AllowGroups=ALL AllowAccounts={p.allow_accounts} AllowQos={p.allow_qos}",
                f"   AllocNodes=ALL Default={'YES' if p.default else 'NO'} QoS={p.qos or 'N/A'}",
                f"   DefaultTime={p.default_time} DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO",
                f"   MaxNodes={p.max_nodes} MaxTime={fmt_limit_upper(mt)} MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED MaxCPUsPerSocket=UNLIMITED",
                f"   Nodes={p.nodes}",
                f"   PriorityJobFactor=1 PriorityTier={p.priority_tier} RootOnly=NO ReqResv=NO OverSubscribe=NO",
                "   OverTimeLimit=NONE PreemptMode=OFF",
                f"   State={p.state} TotalCPUs={cpus} TotalNodes={len(nodes)} SelectTypeParameters=NONE",
                "   JobDefaults=(null)",
                "   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED",
                f"   TRES={tres}",
            ]
        )
        + "\n"
    )


def _show_job(state, j, details=False):
    clock = state.slurm.clock
    run = (
        (clock - j.start_time)
        if j.state == "RUNNING"
        else ((j.end_time - j.start_time) if j.start_time is not None and j.end_time is not None else 0)
    )
    tl = (
        j.time_limit
        if j.time_limit is not None
        else parse_time(state.slurm.partitions.get(j.partition, {}).get("max_time"))
    )
    ncpu = _alloc_cpu_count(state, j) if j.state == "RUNNING" else j.num_nodes * j.ntasks_per_node * j.cpus_per_task
    rcpu = j.num_nodes * j.ntasks_per_node * j.cpus_per_task
    gp = f",gres/gpu={_job_gpus(j)}" if j.gpus_per_node else ""
    req_tres = f"cpu={rcpu},mem=0,node={j.num_nodes},billing={rcpu}{gp}"  # scontrol order (cpu,mem,node,billing,gres)
    alloc_tres = f"cpu={ncpu},mem=0,node={j.num_nodes},billing={ncpu}{gp}"
    out = f"StdOut={j.stdout}" if j.stdout else "StdOut=/dev/null"
    lines = [
        f"JobId={j.id} JobName={j.name}",
        f"   UserId={j.user}({0 if j.user == 'root' else 1000 + sum(map(ord, j.user)) % 500}) GroupId={j.user}({0 if j.user == 'root' else 1000}) MCS_label=N/A",
        f"   Priority={j.priority} Nice={j.nice} Account={j.account} QOS={j.qos}",
        f"   JobState={j.state} Reason={j.reason if j.state == 'PENDING' else 'None'} Dependency={j.dependency or '(null)'}",
        f"   Requeue=1 Restarts={j.restarts} BatchFlag={1 if j.batch else 0} Reboot=0 ExitCode={j.exit_code}",
        f"   RunTime={fmt_elapsed(run, long=True)} TimeLimit={fmt_limit_upper(tl)} TimeMin=N/A",
        f"   SubmitTime={_ts(state, j.submit_time)} EligibleTime={_ts(state, j.submit_time)}",
        f"   StartTime={_ts(state, j.start_time) if j.start_time is not None else 'Unknown'} "
        f"EndTime={_ts(state, j.end_time) if j.end_time is not None else 'Unknown'} Deadline=N/A",
        f"   Partition={j.partition} AllocNode:Sid={j.submit_host}:{2000 + j.id % 1000}",
        f"   ReqNodeList={compress_hostlist(j.req_nodes) or '(null)'} ExcNodeList={compress_hostlist(j.exc_nodes) or '(null)'}",
        f"   NodeList={compress_hostlist(j.nodelist) or '(null)'}",
        f"   NumNodes={j.num_nodes} NumCPUs={ncpu} NumTasks={j.num_nodes * j.ntasks_per_node} CPUs/Task={j.cpus_per_task} ReqB:S:C:T=0:0:*:*",
        f"   ReqTRES={req_tres}",
        f"   AllocTRES={alloc_tres if j.state == 'RUNNING' else '(null)'}",
    ]
    if j.gpus_per_node:
        lines.append(f"   TresPerNode={_job_tres_per_node(j)}")
    if j.get("gres_flags"):
        lines.append(f"   GresEnforceBind={'Yes' if 'enforce-binding' in j.gres_flags else 'No'}")
    if details and j.state == "RUNNING":
        for n in j.nodelist:
            cores = (j.get("alloc_cores") or {}).get(n, [])
            # scontrol -d prints Slurm's abstract CPU ids (core-major), not Linux CPU numbers
            cpu_ids = ",".join(_ranges([x for c in cores for x in (2 * c, 2 * c + 1)])) if cores else "0-1"
            g = (j.get("alloc_gpus") or {}).get(n, [])
            gt = j.gpu_type or parse_gres_spec(state.slurm.nodes[n].gres)[0] if n in state.slurm.nodes else None
            gres = f" GRES=gpu{':' + gt if gt else ''}:{len(g)}(IDX:{','.join(_ranges(g))})" if g else ""
            lines.append(f"     Nodes={n} CPU_IDs={cpu_ids} Mem=0{gres}")
    if j.container_image:
        lines.append(f"   Comment=pyxis container-image={j.container_image}")
    lines += [
        f"   Command={j.command or '(null)'}",
        f"   WorkDir={j.work_dir}",
        f"   StdErr={j.stdout or '/dev/null'}",
        "   StdIn=/dev/null",
        f"   {out}",
    ]
    return "\n".join(lines) + "\n"


CONFIG_DEFAULTS = {
    "AccountingStorageEnforce": "none",
    "EnforcePartLimits": "NO",
    "ReturnToService": "0",
    "HealthCheckProgram": "(null)",
    "HealthCheckInterval": "0",
    "SchedulerType": "sched/backfill",
    "SlurmctldPort": "6817",
    "SlurmdPort": "6818",
    "SlurmdTimeout": "300 sec",
    "PlugStackConfig": PLUGSTACK,
    "JobRequeue": "1",
    "PrologFlags": "(null)",
    "Prolog": "(null)",
    "Epilog": "(null)",
}


def _show_config(state):
    cfg = state.slurm.config
    items = {}
    for k, v in CONFIG_DEFAULTS.items():
        items[k] = v
    for lk, v in cfg["params"].items():
        items[cfg["names"].get(lk, lk)] = v
    items["SLURM_VERSION"] = SLURM_VERSION
    lines = [f"Configuration data as of {_ts(state)}"]
    for k in sorted(items, key=str.lower):
        lines.append(f"{k:<31}= {items[k]}")
    lines += ["", f"Slurmctld(primary) at {HEAD} is UP"]
    return "\n".join(lines) + "\n"


def _parse_kv_args(args):
    out = []
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            out.append((k.lower(), v))
        else:
            out.append((a.lower(), None))
    return out


def _update_node(ctx, kv):
    state = ctx.state
    s = state.slurm
    names = expand_hostlist(dict(kv).get("nodename") or dict(kv).get("node"))
    st = (dict(kv).get("state") or "").upper()
    reason = dict(kv).get("reason")
    for n in names:
        if n not in s.nodes:
            raise _err("slurm_update error: Invalid node name specified")
    if st in ("DRAIN", "DOWN", "FAIL") and not reason:
        raise _err("You must specify a reason when DOWNING or DRAINING a node. Request denied")
    for n in names:
        sn = s.nodes[n]
        if st == "DRAIN":
            _set_drain(state, sn, reason)
        elif st in ("DOWN", "FAIL"):
            sn.down = True
            sn.reason = reason
            sn.reason_user = "root"
            sn.reason_time = _ts(state)
            sn.drain_log.append({"time": _ts(state), "reason": reason, "user": "root", "state": st})
            if st == "DOWN":
                for j in s.jobs:
                    if j.state == "RUNNING" and n in j.nodelist:
                        _finish(state, j, "NODE_FAIL", "0:0")
        elif st in ("RESUME", "IDLE", "UNDRAIN"):
            if not (sn.drain or sn.down or sn.invalid_reg):
                raise _err("slurm_update error: Invalid node state specified")
            diag = state.get("dcgm", {}).get("diag_results", {}).get(n)
            sn.last_resume_diag = diag.get("result") if diag else None
            sn.last_resume_diag_level = diag.get("level") if diag else None
            sn.drain = False
            if st != "UNDRAIN":
                sn.down = False
            if sn.invalid_reg:
                # the controller still holds the bad registration from slurmd
                _, cc = parse_gres_spec(sn.gres)
                rep = sn.gres_reported if sn.gres_reported is not None else cc
                if rep < cc:
                    _set_drain(
                        state,
                        sn,
                        f"gres/gpu count reported lower than configured ({rep} < {cc})",
                        user="slurm",
                        log=False,
                    )
                    _derive_state(state, sn)
                    continue
                sn.invalid_reg = False
            if not sn.drain and not sn.down:
                sn.reason = ""
        elif st:
            raise _err("slurm_update error: Invalid node state specified")
        elif reason is not None:
            sn.reason = reason
        for k, v in kv:
            if k == "weight":
                sn.weight = int(v)
            elif k in ("features", "availablefeatures", "activefeatures"):
                sn.features = v
        _derive_state(state, sn)
    schedule(state)
    return ""


def _update_partition(ctx, kv, create=False):
    state = ctx.state
    s = state.slurm
    d = dict(kv)
    name = d.get("partitionname") or d.get("partition")
    if create:
        if name in s.partitions:
            raise _err("Error creating the partition: Duplicate partition name")
        s.partitions[name] = _partition_from_conf(
            name, {k: v for k, v in kv if v is not None}, state.slurm.config["nodes"]
        )
        _refresh(state)
        return ""
    if name not in s.partitions:
        raise _err("slurm_update error: Invalid partition name specified")
    p = s.partitions[name]
    for k, v in kv:
        if v is None:
            continue
        if k == "state":
            if v.upper() not in ("UP", "DOWN", "DRAIN", "INACTIVE"):
                raise _err("Invalid input: State=" + v)
            p.state = v.upper()
        elif k == "nodes":
            p.nodes = v
        elif k == "maxtime":
            p.max_time = v
        elif k == "qos":
            if v and v not in s.qos:
                raise _err("slurm_update error: Invalid qos specification")
            p.qos = v
        elif k == "allowqos":
            p.allow_qos = v
        elif k == "allowaccounts":
            p.allow_accounts = v
        elif k == "default":
            on = v.upper() == "YES"
            if on:
                for o in s.partitions.values():
                    o.default = False
            p.default = on
        elif k == "maxnodes":
            p.max_nodes = v
        elif k == "prioritytier":
            p.priority_tier = int(v)
        elif k == "defaulttime":
            p.default_time = v
    schedule(state)
    return ""


def _update_job(ctx, kv):
    state = ctx.state
    s = state.slurm
    d = dict(kv)
    jid = d.get("jobid") or d.get("job")
    j = _find_job(state, jid) if jid else None
    if j is None:
        raise _err("slurm_update error: Invalid job id specified")
    for k, v in kv:
        if k in ("jobid", "job") or v is None:
            continue
        if j.state in TERMINAL:
            raise _err("slurm_update error: Job/step already completing or completed")
        pending_only = k not in ("timelimit", "time", "comment", "priority", "nice", "name", "jobname")
        if pending_only and j.state != "PENDING":
            raise _err("slurm_update error: Job is no longer pending execution")
        if k in ("timelimit", "time"):
            j.time_limit = parse_time(v)
        elif k == "partition":
            if v not in s.partitions:
                raise _err("slurm_update error: Invalid partition name specified")
            j.partition = v
        elif k in ("numnodes", "nodes", "minnodes"):
            j.num_nodes = int(v.split("-")[0])
        elif k in ("gres", "tresperNode".lower(), "trespernode"):
            gt, gc = parse_gres_spec(v)
            j.gpus_per_node = gc
            j.gpu_type = gt
        elif k in ("reqnodelist", "nodelist", "reqnodes"):
            j.req_nodes = expand_hostlist(v)
        elif k in ("excnodelist", "exclude"):
            j.exc_nodes = expand_hostlist(v)
        elif k == "qos":
            if v not in s.qos:
                raise _err("slurm_update error: Invalid qos specification")
            j.qos = v
        elif k == "priority":
            j.priority = int(v)
            j.priority_fixed = True
        elif k == "nice":
            j.nice = int(v)
        elif k in ("name", "jobname"):
            j.name = v
        elif k == "account":
            j.account = v
        elif k == "dependency":
            j.dependency = v
        elif k == "numtasks":
            j.ntasks_per_node = max(1, int(v) // max(1, j.num_nodes))
        elif k == "ntaskspernode":
            j.ntasks_per_node = int(v)
        else:
            raise _err(f"Update of this parameter is not supported: {k}={v}")
    if j.state == "PENDING" and j.gpus_per_node:
        pn = _part_nodes(state, j.partition)
        if j.gpus_per_node > max([_conf_gpus(s.nodes[n]) for n in pn] + [0]):
            raise _err("slurm_update error: Requested node configuration is not available")
    schedule(state)
    return ""


def _scontrol_reboot(ctx, args):
    state = ctx.state
    s = state.slurm
    asap = False
    nxt = ""
    reason = ""
    nodes = []
    for a in args:
        if a.upper() == "ASAP":
            asap = True
        elif a.lower().startswith("nextstate="):
            nxt = a.split("=", 1)[1].upper()
        elif a.lower().startswith("reason="):
            reason = a.split("=", 1)[1]
        elif a.upper() == "ALL":
            nodes = list(s.nodes)
        else:
            nodes = expand_hostlist(a)
    if not nodes:
        raise _err("scontrol: error: Missing node list. Specify ALL|<NodeList>")
    for n in nodes:
        if n not in s.nodes:
            raise _err("scontrol: error: Invalid node name specified")
        sn = s.nodes[n]
        sn.reboot_next = nxt
        busy = any(j.state == "RUNNING" and n in j.nodelist for j in s.jobs)
        if busy:
            sn.reboot_pending = True
            if asap:
                sn.reboot_was_drained = sn.drain
                sn.reboot_drained = True
                _set_drain(state, sn, reason or "Reboot ASAP", user="root")
        else:
            _do_slurm_reboot(ctx, n)
    _refresh(state)
    return ""


SCONTROL_USAGE = """\
scontrol [<OPTION>] [<COMMAND>]
    Valid <OPTION> values are:
     -d or --details  Show details for show commands
     -o or --oneliner Equivalent to "oneliner" command
    Valid <COMMAND> values (subset supported by this simulator):
     show <ENTITY> [<ID>]   ENTITY: config, job, node, partition, hostnames
     update <SPECIFICATIONS> e.g. NodeName=dgx-01 State=DRAIN Reason="..."
     create|delete partition, reconfigure, reboot [ASAP] [nextstate=..] <nodes>
     hold|release|requeue <job_list>, ping, version
"""


@command(
    "scontrol",
    help="scontrol show node|partition|job|config, update NodeName=.. State=.., reconfigure, reboot, hold/release/requeue",
)
def scontrol(ctx, args):
    if not args:
        raise _err("scontrol: error: no command given")
    if args[0] in ("-V", "--version", "version"):
        return f"slurm {SLURM_VERSION}\n"
    if args[0] in ("-h", "--help", "help", "--usage"):
        return SCONTROL_USAGE
    detail = False
    while args and args[0].startswith("-"):
        if args[0] in ("-d", "--details"):
            detail = True
        elif args[0] not in ("-o", "--oneliner", "-a", "--all", "-Q", "--quiet", "-v", "--verbose"):
            raise _err(f"scontrol: unrecognized option '{args[0]}'\nTry \"scontrol --help\" for more information")
        args = args[1:]
    if not args:
        raise _err("scontrol: error: no command given")
    sub = args[0].lower()
    rest = args[1:]
    state = ctx.state
    if sub == "show" and rest and rest[0].lower() in ("hostnames", "hostname"):
        return "".join(h + "\n" for h in expand_hostlist(rest[1] if len(rest) > 1 else ""))
    if sub == "show" and rest and rest[0].lower() == "hostlist":
        return compress_hostlist(expand_hostlist(rest[1] if len(rest) > 1 else "")) + "\n"
    if sub == "ping":
        up = _controller_ok(state)
        return f"Slurmctld(primary) at {HEAD} is {'UP' if up else 'DOWN'}\n"
    _need_ctl(ctx, "slurm_load_ctl_conf error: " if sub == "show" else "slurm_update error: ")
    _begin(ctx)
    s = state.slurm
    if sub == "show":
        if not rest:
            raise _err("scontrol: error: too few arguments for keyword:show")
        what = rest[0].lower()
        ent = rest[1] if len(rest) > 1 else None
        if "=" in what:
            what, ent = what.split("=", 1)
        if what in ("node", "nodes"):
            names = expand_hostlist(ent) if ent else list(s.nodes)
            out = []
            for n in names:
                if n not in s.nodes:
                    raise _err(f"Node {n} not found")
                out.append(_show_node(state, s.nodes[n], details=detail))
            return "\n".join(out)
        if what in ("partition", "partitions", "part"):
            names = [ent] if ent else list(s.partitions)
            out = []
            for p in names:
                if p not in s.partitions:
                    raise _err(f"Partition {p} not found")
                out.append(_show_partition(state, s.partitions[p]))
            return "\n".join(out)
        if what in ("job", "jobs", "jobid"):
            if ent:
                j = _find_job(state, ent)
                if j is None:
                    raise _err("slurm_load_jobs error: Invalid job id specified")
                return _show_job(state, j, details=detail)
            js = [j for j in s.jobs if j.state not in TERMINAL or s.clock - (j.end_time or 0) < 300]
            if not js:
                return "No jobs in the system\n"
            return "\n".join(_show_job(state, j, details=detail) for j in js)
        if what in ("config", "conf"):
            return _show_config(state)
        if what in ("qos",):
            raise _err("scontrol: error: use 'sacctmgr show qos'")
        raise _err(f"scontrol: error: Invalid entity: {rest[0]} for keyword: show")
    if sub in ("update", "create"):
        kv = _parse_kv_args(rest)
        keys = [k for k, _ in kv]
        if sub == "create" and rest and rest[0].lower() in ("partition", "partitionname"):
            kv = _parse_kv_args(rest[1:]) if "=" not in rest[0] else kv
            return _update_partition(ctx, kv, create=True)
        if "nodename" in keys or "node" in keys:
            if sub == "create":
                raise _err("scontrol: error: create node is not supported in this simulator; edit slurm.conf")
            return _update_node(ctx, kv)
        if "jobid" in keys or "job" in keys:
            return _update_job(ctx, kv)
        if "partitionname" in keys:
            return _update_partition(ctx, kv, create=(sub == "create"))
        raise _err("scontrol: error: Invalid update request. Specify NodeName=, PartitionName= or JobId=")
    if sub == "delete":
        kv = dict(_parse_kv_args(rest))
        name = kv.get("partitionname") or kv.get("partition")
        if name in s.partitions:
            del s.partitions[name]
            return ""
        raise _err("slurm_delete_partition error: Invalid partition name specified")
    if sub in ("reconfigure", "reconfig"):
        apply_config(state)
        schedule(state)
        return ""
    if sub == "reboot" or sub == "reboot_nodes":
        return _scontrol_reboot(ctx, rest)
    if sub in ("hold", "uhold", "release", "requeue", "requeuehold"):
        for jid in rest:
            for one in jid.split(","):
                j = _find_job(state, one)
                if j is None:
                    raise _err(f"Invalid job id specified for job {one}")
                if sub in ("hold", "uhold"):
                    if j.state != "PENDING":
                        raise _err(f"Job is no longer pending execution for job {one}")
                    j.held = True
                    j.hold_reason = "JobHeldUser" if sub == "uhold" else "JobHeldAdmin"
                elif sub == "release":
                    j.held = False
                else:
                    if j.state != "RUNNING" and j.state not in TERMINAL:
                        raise _err(f"Job is pending execution for job {one}")
                    if j.state == "RUNNING":
                        _finish(state, j, "PENDING", "0:0")
                    j.state = "PENDING"
                    j.reason = "BeginTime"
                    j.nodelist = []
                    j.alloc_gpus = {}
                    j.start_time = None
                    j.end_time = None
                    j.restarts += 1
                    if sub == "requeuehold":
                        j.held = True
                        j.hold_reason = "JobHeldAdmin"
                    j["_requeued"] = True
        # requeued jobs are eligible on the next scheduling pass (BeginTime delay)
        _refresh(state)
        return ""
    if sub == "listpids":
        return "PID      JOBID    STEPID   LOCALID GLOBALID\n"
    raise _err(f"invalid keyword: {args[0]}")


@command("slurmd", help="slurmd -C (print node hardware config line) | -V")
def slurmd_cmd(ctx, args):
    node = ctx.node
    if "-V" in args or "--version" in args:
        return f"slurm {SLURM_VERSION}\n"
    if "-C" in args:
        gp = len([g for g in node.gpus if not g.fallen_off_bus])
        g = f" Gres=gpu:h100:{gp}" if gp else ""
        return (
            f"NodeName={ctx.host} CPUs={node.cpus} Boards=1 SocketsPerBoard=2 CoresPerSocket={node.cpus // 4} "
            f"ThreadsPerCore=2 RealMemory={node.real_memory_mb}{g}\nUpTime=12-03:11:40\n"
        )
    if "-G" in args:
        _ensure(ctx.state)
        parsed = _parse_gres_conf(node.files.get(GRES_CONF, ""), ctx.host)
        det = [x for x in node.gpus if not x.fallen_off_bus and not x.reset_required]
        cmap, cerr = _gpu_core_map(ctx.state, ctx.host)
        lines = [f"slurmd: gpu/nvml: _get_system_gpu_list_nvml: {len(det)} GPU system device(s) detected"]
        if cerr:
            lines.append(f"slurmd: error: {cerr}")
        for x in det:
            if parsed["lines"] and not any(
                f"/dev/nvidia{x.index}" in ln["files"] for ln in parsed["lines"] if ln["files"]
            ):
                continue
            cores = cmap.get(x.index)
            cspec = f"Cores={','.join(_ranges(cores))} CoreCnt={node.cpus // 2} " if cores else "CoreCnt=112 "
            lines.append(
                f"slurmd: Gres Name=gpu Type=h100 Count=1 Index={x.index} ID=7696487 File=/dev/nvidia{x.index} "
                f"{cspec}Links=-1,0,0,0,0,0,0,0 Flags=HAS_FILE,HAS_TYPE,ENV_NVML"
            )
        return "\n".join(lines) + "\n"
    raise _err("slurmd is managed by systemd here: use 'systemctl restart slurmd' (slurmd -C / -G for diagnostics)")
