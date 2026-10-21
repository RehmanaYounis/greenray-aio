"""Node state: slurmd registration, gres/health checks, reboots, and the slurm.conf reload that drives them."""

from __future__ import annotations

import re

from .. import HOOKS_REBOOT
from .affinity import _set_core_map
from .common import CONF, GRES_CONF, HEAD, SLURM_VERSION, TICK, _ts, parse_gres_spec
from .conf import (
    _gpu_types_enabled,
    _new_snode,
    _node_gres_from_conf,
    _param,
    _parse_gres_conf,
    _partition_from_conf,
    parse_slurm_conf,
)
from .munge import _munge_log, _munge_ok


def _node_defaults(state, sn):
    st = str(sn.get("state") or "IDLE").upper()
    if "_seen_state" not in sn or st != sn.get("_seen_state"):
        sn.drain = "DRAIN" in st or st.startswith("INVAL") or "INVALID_REG" in st
        sn.down = st.startswith("DOWN") or st.startswith("FAIL")
        sn.invalid_reg = st.startswith("INVAL") or "INVALID_REG" in st
        sn.not_responding = st.endswith("*") or "NOT_RESPONDING" in st
        sn["_seen_state"] = st
    sn.setdefault("reason", "")
    sn.setdefault("reason_user", "root")
    sn.setdefault("reason_time", _ts(state, 0))
    sn.setdefault("gres_reported", None)
    sn.setdefault("drain_log", [])
    sn.setdefault("boot_time", "2026-10-07T07:31:02")
    sn.setdefault("cpus", 224)
    sn.setdefault("real_memory", 2063000)
    sn.setdefault("weight", 1)
    sn.setdefault("features", "")
    sn.setdefault("gres", "(null)")
    if "gpu_cores" not in sn and sn.get("name") in state.nodes:
        _set_core_map(state, sn)


def _used_gpu_idx(state, node, exclude=None):
    used = set()
    for j in state.slurm.jobs:
        if j.get("state") == "RUNNING" and j.get("id") != exclude:
            used.update((j.get("alloc_gpus") or {}).get(node, []))
    return used


def _conf_gpus(sn) -> int:
    return parse_gres_spec(sn.get("gres"))[1]


def _derive_state(state, sn):
    name = sn.name
    running = [j for j in state.slurm.jobs if j.state == "RUNNING" and name in j.nodelist]
    used_gpus = sum(len((j.alloc_gpus or {}).get(name, [])) for j in running)
    alloc_cpus = 0
    for j in running:
        alloc_cpus += sn.cpus if j.exclusive else j.ntasks_per_node * j.cpus_per_task
    sn.alloc_cpus = min(alloc_cpus, sn.cpus)
    gtype, gcount = parse_gres_spec(sn.gres)
    sn.gres_used = f"gpu:{gtype}:{used_gpus}" if gtype else f"gpu:{used_gpus}"
    busy = bool(running)
    if sn.down:
        st = "DOWN"
    elif sn.invalid_reg and sn.drain:
        st = "INVAL"
    elif sn.drain:
        st = "DRAINING" if busy else "DRAIN"
    elif busy:
        st = "ALLOCATED" if sn.alloc_cpus >= sn.cpus else "MIXED"
    else:
        st = "IDLE"
    sn.state = st
    sn["_seen_state"] = st


def _refresh(state):
    s = state.slurm
    rts = str(_param(state, "ReturnToService", "0"))
    for name, sn in s.nodes.items():
        _node_defaults(state, sn)
        node = state.nodes.get(name)
        if node is not None:
            alive = node.get("up", True) and node.get("power", "on") == "on"
            slurmd_ok = node.services.get("slurmd", "active") == "active"
            if not alive:
                if not sn.down:
                    sn.down = True
                    sn.reason = "Not responding"
                    sn.reason_user = "slurm"
                    sn.reason_time = _ts(state)
                    for j in s.jobs:
                        if j.state == "RUNNING" and name in j.nodelist:
                            _finish(state, j, "NODE_FAIL", "0:0")
                sn.not_responding = True
            else:
                auth_ok = _munge_ok(state, name)
                sn.not_responding = not slurmd_ok or not auth_ok
                if slurmd_ok and not auth_ok and not sn.down:
                    sn.down = True
                    sn.reason = "Not responding"
                    sn.reason_user = "slurm"
                    sn.reason_time = _ts(state)
                    for j in s.jobs:
                        if j.state == "RUNNING" and name in j.nodelist:
                            _finish(state, j, "NODE_FAIL", "0:0")
                if sn.down and sn.reason == "Not responding" and slurmd_ok and auth_ok and rts in ("1", "2"):
                    sn.down = False
                    sn.reason = ""
        _derive_state(state, sn)


def _usable(state, name) -> bool:
    sn = state.slurm.nodes.get(name)
    if sn is None:
        return False
    return not (sn.down or sn.drain or sn.not_responding)


def _node_label(sn, long=False) -> str:
    """sinfo state string."""
    st = sn.state
    if long:
        m = {
            "IDLE": "idle",
            "MIXED": "mixed",
            "ALLOCATED": "allocated",
            "DRAIN": "drained",
            "DRAINING": "draining",
            "DOWN": "down",
            "INVAL": "inval",
        }
    else:
        m = {
            "IDLE": "idle",
            "MIXED": "mix",
            "ALLOCATED": "alloc",
            "DRAIN": "drain",
            "DRAINING": "drng",
            "DOWN": "down",
            "INVAL": "inval",
        }
    out = m.get(st, st.lower())
    if sn.get("not_responding"):
        out += "*"
    return out


def _scontrol_node_state(sn) -> str:
    st = sn.state
    base = {"DRAIN": "IDLE", "DRAINING": "MIXED", "INVAL": "IDLE"}.get(st, st)
    if st == "DRAINING" and sn.alloc_cpus >= sn.cpus:
        base = "ALLOCATED"
    flags = []
    if sn.drain or st == "INVAL":
        flags.append("DRAIN")
    if sn.invalid_reg:
        flags.append("INVALID_REG")
    if sn.not_responding:
        flags.append("NOT_RESPONDING")
    if sn.get("reboot_pending"):
        flags.append("REBOOT_REQUESTED")
    return "+".join([base] + flags)


def _set_drain(state, sn, reason, user="root", log=True):
    sn.drain = True
    sn.reason = reason
    sn.reason_user = user
    sn.reason_time = _ts(state)
    if log:
        sn.drain_log.append({"time": _ts(state), "reason": reason, "user": user})


def _finish(state, j, st="COMPLETED", exit_code="0:0"):
    j.state = st
    j.exit_code = exit_code
    j.end_time = state.slurm.clock
    for n in j.nodelist:
        node = state.nodes.get(n)
        if node is None:
            continue
        for g in node.gpus:
            keep = [p for p in g.processes if p.get("slurm_job") != j.id]
            if len(keep) != len(g.processes):
                freed = sum(p.used_mib for p in g.processes if p.get("slurm_job") == j.id)
                g.processes = keep
                g.memory_used_mib = max(0, g.memory_used_mib - freed)
                if not keep:
                    g.utilization = 0
        if j.get("pty") and node.env.get("SLURM_JOB_ID") == str(j.id):
            node.env.pop("SLURM_JOB_ID", None)
    for n in j.nodelist:
        if n in state.slurm.nodes:
            _derive_state(state, state.slurm.nodes[n])


def _register(state, host, log=True, after_reboot=False) -> bool:
    """slurmd (re)registration: detect GPUs, compare with the controller's config.

    ``after_reboot``: called from a reboot hook, possibly before the nvidia hook has
    cleared transient GPU faults, so judge the GPUs as a fresh boot will see them.
    """
    s = state.slurm
    sn = s.nodes.get(host)
    node = state.nodes.get(host)
    if sn is None or node is None:
        return True
    journal = node.journal.setdefault("slurmd", [])
    head_j = state.nodes[HEAD].journal.setdefault("slurmctld", []) if HEAD in state.nodes else []
    ts = _ts(state).replace("T", " ")
    pid = 4100 + (s.clock // TICK) % 800
    if not _munge_ok(state, host):
        if log:
            _munge_log(state, host)
        sn.not_responding = True
        return False
    parsed = _parse_gres_conf(node.files.get(GRES_CONF, ""), host)
    _set_core_map(state, sn, log=log)
    if after_reboot:
        detected = [g.index for g in node.gpus]
    else:
        detected = (
            [g.index for g in node.gpus if not g.fallen_off_bus and not g.reset_required]
            if node.get("driver_loaded", True)
            else []
        )
    for ln in parsed["lines"]:
        for f in ln["files"]:
            m = re.match(r"/dev/nvidia(\d+)$", f)
            if m and int(m.group(1)) >= len(node.gpus):
                msg = f"fatal: can't stat gres.conf file {f}: No such file or directory"
                journal.append(f"{ts} {host} slurmd[{pid}]: {msg}")
                journal.append(
                    f"{ts} {host} systemd[1]: slurmd.service: Main process exited, code=exited, status=1/FAILURE"
                )
                node.services["slurmd"] = "failed"
                sn.not_responding = True
                return False
    rep_type = None
    if parsed["lines"]:
        reported = 0
        for ln in parsed["lines"]:
            rep_type = ln["type"] or rep_type
            if ln["files"]:
                idx = [int(m.group(1)) for f in ln["files"] if (m := re.match(r"/dev/nvidia(\d+)$", f))]
                reported += len([i for i in idx if i in detected])
            else:
                reported += min(ln["count"], len(detected))
    elif parsed["autodetect"]:
        reported = len(detected)
        rep_type = "h100" if node.gpus else None
    else:
        reported = 0
    sn.gres_reported = reported
    if log:
        if parsed["autodetect"] or parsed["lines"]:
            journal.append(
                f"{ts} {host} slurmd[{pid}]: slurmd: gpu/nvml: _get_system_gpu_list_nvml: "
                f"{len(detected)} GPU system device(s) detected"
            )
        for g in node.gpus:
            if g.fallen_off_bus:
                journal.append(
                    f"{ts} {host} slurmd[{pid}]: slurmd: error: _nvml_get_mem_freqs: "
                    f"Failed to get handle for GPU {g.bus_id}: Unknown Error"
                )
        journal.append(f"{ts} {host} slurmd[{pid}]: slurmd: Gres Name=gpu Type={rep_type or '(null)'} Count={reported}")
        journal.append(f"{ts} {host} slurmd[{pid}]: slurmd: slurmd version {SLURM_VERSION} started")
        journal.append(
            f"{ts} {host} slurmd[{pid}]: slurmd: CPUs=224 Boards=1 Sockets=2 Cores=56 Threads=2 "
            f"Memory=2063000 TmpDisk=0 Uptime=1043 CPUSpecList=(null) FeaturesAvail=(null) FeaturesActive=(null)"
        )
    conf_type, conf_count = parse_gres_spec(sn.gres)
    sn.not_responding = False
    bad = None
    if conf_count:
        eff = reported if (conf_type is None or rep_type is None or conf_type == rep_type) else 0
        if eff < conf_count:
            label = f"gres/gpu:{conf_type}" if (conf_type and rep_type and conf_type != rep_type) else "gres/gpu"
            bad = f"{label} count reported lower than configured ({eff} < {conf_count})"
    if bad:
        sn.invalid_reg = True
        if not sn.drain or sn.reason != bad:
            _set_drain(state, sn, bad, user="slurm", log=False)
        head_j.append(f"{ts} {HEAD} slurmctld[2211]: error: _slurm_rpc_node_registration node={host}: Invalid argument")
        head_j.append(f"{ts} {HEAD} slurmctld[2211]: error: Setting node {host} state to INVAL with reason:{bad}")
        journal.append(f"{ts} {host} slurmd[{pid}]: slurmd: error: Unable to register: Invalid argument")
        _derive_state(state, sn)
        return False
    if sn.invalid_reg:
        sn.invalid_reg = False
        head_j.append(
            f"{ts} {HEAD} slurmctld[2211]: Node {host} now responding (registration valid; node remains DRAIN until resumed)"
        )
    _derive_state(state, sn)
    return True


def apply_config(state):
    """slurmctld re-reads slurm.conf (restart or `scontrol reconfigure`)."""
    s = state.slurm
    conf = parse_slurm_conf(state.nodes[HEAD].files.get(CONF, ""))
    s.config = conf
    for n, attrs in conf["nodes"].items():
        if n not in s.nodes:
            s.nodes[n] = _new_snode(n, attrs)
            _node_defaults(state, s.nodes[n])
        sn = s.nodes[n]
        sn.gres = _node_gres_from_conf(state, attrs)
        sn.cpus = int(attrs.get("cpus", sn.cpus))
        sn.real_memory = int(attrs.get("realmemory", sn.real_memory))
        sn.weight = int(attrs.get("weight", sn.get("weight", 1)))
        sn.features = attrs.get("feature", attrs.get("features", ""))
    for n in list(s.nodes):
        if n not in conf["nodes"] and not any(j.state == "RUNNING" and n in j.nodelist for j in s.jobs):
            del s.nodes[n]
    parts = {}
    for p in conf["part_order"]:
        parts[p] = _partition_from_conf(p, conf["partitions"][p], conf["nodes"])
    s.partitions = parts
    ts = _ts(state).replace("T", " ")
    lg = state.nodes[HEAD].journal.setdefault("slurmctld", [])
    if not _gpu_types_enabled(state) and any(a.get("gres") for a in conf["nodes"].values()):
        lg.append(
            f"{ts} {HEAD} slurmctld[2211]: error: Gres=gpu configured for nodes but 'gpu' is not in GresTypes; ignoring"
        )
    lg.append(
        f"{ts} {HEAD} slurmctld[2211]: Reconfigure: read {CONF}, {len(conf['nodes'])} nodes, {len(parts)} partitions"
    )
    for n in s.nodes:
        node = state.nodes.get(n)
        if node is not None and node.get("up", True) and node.services.get("slurmd") == "active":
            _register(state, n)
    _refresh(state)


def _health_check(state):
    """Simulated NHC (only when HealthCheckProgram is configured)."""
    if not _param(state, "HealthCheckProgram", ""):
        return
    for name, sn in state.slurm.nodes.items():
        node = state.nodes.get(name)
        if node is None or sn.drain or sn.down or not node.get("up", True):
            continue
        bad = [g for g in node.gpus if g.fallen_off_bus or g.reset_required]
        if bad:
            g = bad[0]
            why = "GPU fell off the bus (Xid 79)" if g.fallen_off_bus else "GPU requires reset"
            _set_drain(state, sn, f"NHC: check_nv_gpu: GPU {g.bus_id} {why}", user="root", log=False)


def _do_slurm_reboot(ctx, name):
    state = ctx.state
    sn = state.slurm.nodes[name]
    node = state.nodes.get(name)
    nxt = sn.get("reboot_next", "")
    sn.reboot_pending = False
    if node is not None:
        sn["_expected_reboot"] = True
        for fn in HOOKS_REBOOT:
            fn(ctx, node)
        sn["_expected_reboot"] = False
    if sn.get("reboot_drained"):
        sn.drain = sn.get("reboot_was_drained", False)
        if not sn.drain:
            sn.reason = ""
    if nxt == "RESUME":
        sn.drain = False
        sn.down = False
        sn.reason = ""
    elif nxt == "DOWN":
        sn.down = True
        sn.reason = sn.reason or "Reboot complete"
    _derive_state(state, sn)
