"""Slurm emulation: sinfo, squeue, scontrol, sbatch, srun, salloc, scancel, sacct,
sacctmgr, sprio, sshare, slurmd -C, plus a tiny scheduler.

State lives in ``state.slurm`` (see aiolab/state.py). Extra fields this module adds
lazily (scenario setups may set them directly):

  slurm.config        parsed *running* slurm.conf: {"params": {lowercase key: value},
                      "names": {lowercase: OriginalKey}, "nodes": {...}, "partitions": {...}}
                      Re-read only on `scontrol reconfigure` or slurmctld restart.
  slurm.clock         simulated seconds since 2026-10-19 09:00:00; every Slurm command
                      advances it by 30 s, which is how batch jobs progress and finish.
  slurm.nodes[n]      + drain, down, invalid_reg, not_responding (flags; `state` is derived
                      from them: IDLE / MIXED / ALLOCATED / DRAIN / DRAINING / INVAL / DOWN),
                      reason_user, reason_time, gres_reported, drain_log, last_resume_diag
  slurm.partitions[p] + qos, max_nodes, default_time
  slurm.qos[q]        + max_tres_per_job, max_jobs_per_user, max_wall, grp_tres, flags
  slurm.users[u]      + qos (allowed list), accounts

Job fields (``slurm.jobs`` entries; setups may give only some of them):
  id, name, user, account, partition, qos, state (PENDING/RUNNING/COMPLETED/FAILED/
  CANCELLED/NODE_FAIL/TIMEOUT), reason, num_nodes, ntasks_per_node, cpus_per_task,
  gpus_per_node, gpu_type, time_limit (minutes, None = partition limit), req_nodes,
  exc_nodes, nodelist (list), alloc_gpus {node: [idx]}, submit_time, start_time,
  end_time, duration (seconds; None = runs until cancelled), command, script, work_dir,
  stdout, container_image, exit_code ("0:0"), priority, batch, held, dependency,
  exclusive, restarts, steps
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fnmatch
import math
import re
import shlex

from ..core.errors import CommandError
from . import HOOKS_REBOOT, HOOKS_SERVICE, command

SLURM_VERSION = "23.11.6"
BASE_TIME = _dt.datetime(2026, 10, 19, 9, 0, 0)
TICK = 30
CONF = "/etc/slurm/slurm.conf"
GRES_CONF = "/etc/slurm/gres.conf"
PLUGSTACK = "/etc/slurm/plugstack.conf"
HEAD = "bcm-head"
BATCH_DEFAULT_DURATION = 90
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "NODE_FAIL", "TIMEOUT", "OUT_OF_MEMORY")
STATE_CODES = {
    "PENDING": "PD",
    "RUNNING": "R",
    "COMPLETED": "CD",
    "FAILED": "F",
    "CANCELLED": "CA",
    "NODE_FAIL": "NF",
    "TIMEOUT": "TO",
    "COMPLETING": "CG",
    "OUT_OF_MEMORY": "OOM",
    "SUSPENDED": "S",
    "REQUEUED": "RQ",
}
NOT_AVAIL_TEXT = "Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions"

_GUARD = {"depth": 0}
# user identity of the current command (``su - alice -c "sbatch ..."``); root by default
_USER: list = ["root"]
# local task id of the job step task being executed (for per-task CPU binding)
_TASK: dict = {"local": None, "per_node": 1}
MUNGE_KEY = "/etc/munge/munge.key"
CORES_PER_SOCKET = 56


# ====================================================================== helpers


def _split_commas(expr: str) -> list[str]:
    out, buf, depth = [], "", 0
    for ch in expr:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        out.append(buf)
    return [x.strip() for x in out if x.strip()]


def expand_hostlist(expr) -> list[str]:
    """dgx-[01-03,05] -> [dgx-01, dgx-02, dgx-03, dgx-05]"""
    if expr is None:
        return []
    if isinstance(expr, (list, tuple)):
        return [h for e in expr for h in expand_hostlist(e)]
    expr = str(expr).strip()
    if not expr or expr in ("(null)", "None"):
        return []
    out = []
    for part in _split_commas(expr):
        m = re.match(r"^(.*?)\[([^\]]+)\](.*)$", part)
        if not m:
            out.append(part)
            continue
        pre, body, post = m.groups()
        for rng in body.split(","):
            if "-" in rng:
                a, b = rng.split("-", 1)
                for i in range(int(a), int(b) + 1):
                    out.extend(expand_hostlist(f"{pre}{i:0{len(a)}d}{post}"))
            else:
                out.extend(expand_hostlist(pre + rng + post))
    return out


def compress_hostlist(names) -> str:
    names = list(dict.fromkeys(names))
    if not names:
        return ""
    groups, order, singles = {}, [], []
    for n in names:
        m = re.match(r"^(.*?)(\d+)$", n)
        if not m:
            singles.append(n)
            continue
        key = (m.group(1), len(m.group(2)))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(int(m.group(2)))
    parts = []
    for pre, w in order:
        nums = sorted(set(groups[(pre, w)]))
        if len(nums) == 1:
            parts.append(f"{pre}{nums[0]:0{w}d}")
            continue
        ranges, start, prev = [], nums[0], nums[0]
        for x in nums[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            ranges.append(f"{start:0{w}d}" if start == prev else f"{start:0{w}d}-{prev:0{w}d}")
            if x is not None:
                start = prev = x
        parts.append(f"{pre}[{','.join(ranges)}]")
    return ",".join(parts + singles)


def parse_time(spec) -> int | None:
    """Slurm time spec -> minutes (None = unlimited)."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return int(spec)
    s = str(spec).strip()
    if s.upper() in ("UNLIMITED", "INFINITE", "", "NONE"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
        parts = [int(x) for x in s.split(":")]
        h, m, sec = (parts + [0, 0, 0])[:3]
        return days * 1440 + h * 60 + m + (1 if sec else 0)
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] + (1 if parts[1] else 0)
    return parts[0] * 60 + parts[1] + (1 if parts[2] else 0)


def fmt_limit(minutes) -> str:
    if minutes is None:
        return "infinite"
    d, rem = divmod(int(minutes), 1440)
    h, m = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:00" if d else f"{h}:{m:02d}:00"


def fmt_limit_upper(minutes) -> str:
    if minutes is None:
        return "UNLIMITED"
    d, rem = divmod(int(minutes), 1440)
    h, m = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:00" if d else f"{h:02d}:{m:02d}:00"


def fmt_elapsed(sec: int, long=False) -> str:
    sec = max(0, int(sec))
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if long:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"
    if d:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _ts(state, sec=None) -> str:
    if sec is None:
        sec = state.slurm.get("clock", 0)
    return (BASE_TIME + _dt.timedelta(seconds=int(sec))).strftime("%Y-%m-%dT%H:%M:%S")


def _banner_time(state) -> str:
    return (BASE_TIME + _dt.timedelta(seconds=int(state.slurm.get("clock", 0)))).strftime("%a %b %d %H:%M:%S %Y")


def parse_gres_spec(spec) -> tuple[str | None, int]:
    """'gpu:h100:8' -> ('h100', 8); 'gpu:8' -> (None, 8); '(null)' -> (None, 0)."""
    if not spec or str(spec) in ("(null)", "None"):
        return None, 0
    for tok in _split_commas(str(spec)):
        tok = re.sub(r"\(.*\)$", "", tok.strip())
        tok = tok.removeprefix("gres/").removeprefix("gres:")
        parts = tok.split(":")
        if parts[0] != "gpu":
            continue
        if len(parts) == 1:
            return None, 1
        if len(parts) == 2:
            return (None, int(parts[1])) if parts[1].isdigit() else (parts[1], 1)
        return parts[1], int(parts[2]) if parts[2].isdigit() else 1
    return None, 0


def parse_tres(spec) -> dict:
    """'gres/gpu=8,cpu=64' -> {'gres/gpu': 8, 'cpu': 64}"""
    out = {}
    for tok in str(spec or "").split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = k.strip().lower()
            if k == "gpu":
                k = "gres/gpu"
            with contextlib.suppress(ValueError):
                out[k] = int(v)
    return out


def _kv(tokens) -> list[tuple[str, str]]:
    out = []
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            out.append((k, v))
        else:
            out.append((t, ""))
    return out


def _err(msg, code=1):
    return CommandError(msg, code)


# ====================================================================== config


def parse_slurm_conf(text: str) -> dict:
    conf = {"params": {}, "names": {}, "nodes": {}, "partitions": {}, "part_order": []}
    node_default, part_default = {}, {}
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            toks = shlex.split(line)
        except ValueError:
            toks = line.split()
        kv = _kv(toks)
        k0, v0 = kv[0]
        attrs = {k.lower(): v for k, v in kv[1:]}
        if k0.lower() == "nodename":
            if v0.upper() == "DEFAULT":
                node_default.update(attrs)
                continue
            for n in expand_hostlist(v0):
                d = dict(node_default)
                d.update(attrs)
                conf["nodes"][n] = d
        elif k0.lower() == "partitionname":
            if v0.upper() == "DEFAULT":
                part_default.update(attrs)
                continue
            d = dict(part_default)
            d.update(attrs)
            conf["partitions"][v0] = d
            if v0 not in conf["part_order"]:
                conf["part_order"].append(v0)
        else:
            for k, v in kv:
                conf["params"][k.lower()] = v
                conf["names"][k.lower()] = k
    return conf


def _param(state, key, default=""):
    cfg = state.slurm.get("config") or {}
    return (cfg.get("params") or {}).get(key.lower(), default)


def _enforce(state) -> set:
    raw = _param(state, "AccountingStorageEnforce", "none").lower()
    flags = {x.strip() for x in raw.split(",") if x.strip() and x.strip() != "none"}
    if flags & {"limits", "qos", "safe"}:
        flags.add("associations")
    if "safe" in flags:
        flags.add("limits")
    return flags


def _gpu_types_enabled(state) -> bool:
    return "gpu" in [x.strip() for x in _param(state, "GresTypes", "").split(",")]


def _partition_from_conf(name, attrs, conf_nodes):
    nodes = attrs.get("nodes", "")
    if nodes.upper() == "ALL":
        nodes = compress_hostlist(list(conf_nodes))
    return {
        "name": name,
        "nodes": nodes,
        "default": attrs.get("default", "NO").upper() == "YES",
        "state": attrs.get("state", "UP").upper(),
        "max_time": attrs.get("maxtime", "UNLIMITED").upper()
        if attrs.get("maxtime", "UNLIMITED").upper() in ("INFINITE", "UNLIMITED")
        else attrs.get("maxtime"),
        "default_time": attrs.get("defaulttime", "NONE"),
        "allow_qos": attrs.get("allowqos", "ALL"),
        "allow_accounts": attrs.get("allowaccounts", "ALL"),
        "priority_tier": int(attrs.get("prioritytier", 1)),
        "qos": attrs.get("qos", ""),
        "max_nodes": attrs.get("maxnodes", "UNLIMITED"),
    }


def _node_gres_from_conf(state, attrs):
    g = attrs.get("gres", "")
    if not g or not _gpu_types_enabled(state):
        return "(null)"
    return g


def _new_snode(name, attrs):
    return {
        "name": name,
        "state": "IDLE",
        "reason": "",
        "gres": "(null)",
        "gres_used": "gpu:0",
        "cpus": int(attrs.get("cpus", 1)),
        "alloc_cpus": 0,
        "real_memory": int(attrs.get("realmemory", 1)),
        "features": attrs.get("feature", ""),
        "weight": int(attrs.get("weight", 1)),
    }


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


def _used_gpu_idx(state, node, exclude=None):
    used = set()
    for j in state.slurm.jobs:
        if j.get("state") == "RUNNING" and j.get("id") != exclude:
            used.update((j.get("alloc_gpus") or {}).get(node, []))
    return used


# ====================================================================== node state


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


# ====================================================================== slurmd registration


def _parse_gres_conf(text, host):
    """Returns dict(autodetect, lines=[{type, files, count}], error)."""
    res = {"autodetect": False, "lines": [], "error": None}
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        attrs = {}
        for k, v in _kv(line.split()):
            attrs[k.lower()] = v
        if "nodename" in attrs and host not in expand_hostlist(attrs["nodename"]):
            continue
        if "autodetect" in attrs:
            res["autodetect"] = attrs["autodetect"].lower() in ("nvml", "nvidia", "rsmi")
        if attrs.get("name") == "gpu":
            files = expand_hostlist(attrs["file"]) if "file" in attrs else []
            count = int(attrs["count"]) if attrs.get("count", "").isdigit() else (len(files) or 1)
            res["lines"].append(
                {"type": attrs.get("type"), "files": files, "count": count, "cores": attrs.get("cores")}
            )
        elif "name" in attrs and attrs["name"] != "gpu":
            pass
    return res


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


# ====================================================================== priority / scheduler


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


def _find_job(state, jid):
    jid = str(jid).split(".")[0].split("_")[0]
    for j in state.slurm.jobs:
        if str(j.get("id")) == jid:
            return j
    return None


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


# ====================================================================== format engine

_FMT_RX = re.compile(r"%(\.?)(\d*)([a-zA-Z])")


def _render(fmt, rows, header_map, noheader=False):
    specs = []
    pos = 0
    literals = []
    for m in _FMT_RX.finditer(fmt):
        literals.append(fmt[pos : m.start()])
        specs.append((m.group(1) == ".", int(m.group(2)) if m.group(2) else None, m.group(3)))
        pos = m.end()
    tail = fmt[pos:]

    def cell(val, right, width):
        val = str(val)
        if width is None:
            return val
        val = val[:width]
        return val.rjust(width) if right else val.ljust(width)

    out = []
    if not noheader:
        line = ""
        for lit, (right, width, letter) in zip(literals, specs):
            line += lit + cell(header_map.get(letter, letter.upper()), right, width)
        out.append((line + tail).rstrip())
    for r in rows:
        line = ""
        for lit, (right, width, letter) in zip(literals, specs):
            line += lit + cell(r.get(letter, ""), right, width)
        out.append((line + tail).rstrip())
    return "\n".join(out) + "\n"


def _opt(args, *names, default=None):
    """Value of -x VAL / -xVAL / --long=VAL / --long VAL."""
    for i, a in enumerate(args):
        for n in names:
            if a == n and i + 1 < len(args):
                return args[i + 1]
            if n.startswith("--") and a.startswith(n + "="):
                return a.split("=", 1)[1]
            if not n.startswith("--") and len(n) == 2 and a.startswith(n) and len(a) > 2 and not a.startswith("--"):
                return a[2:]
    return default


def _has(args, *flags):
    return any(a in flags for a in args)


# ====================================================================== sinfo

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


# ====================================================================== squeue

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


# ====================================================================== scontrol


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


def _ranges(nums):
    nums = sorted(nums)
    out, start, prev = [], nums[0], nums[0]
    for x in nums[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if x is not None:
            start = prev = x
    return out


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


@command(
    "scontrol",
    help="scontrol show node|partition|job|config, update NodeName=.. State=.., reconfigure, reboot, hold/release/requeue",
)
def scontrol(ctx, args):
    if not args:
        raise _err("scontrol: error: no command given")
    if args[0] in ("-V", "--version", "version"):
        return f"slurm {SLURM_VERSION}\n"
    detail = False
    while args and args[0].startswith("-"):
        if args[0] in ("-d", "--details"):
            detail = True
        args = args[1:]
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


# ====================================================================== job request parsing

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
    "--mem-per-gpu": "mem",
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


def _plugstack_has_pyxis(files, path=PLUGSTACK, depth=0):
    text = files.get(path)
    if text is None or depth > 3:
        return None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        toks = line.split()
        if toks[0] == "include" and len(toks) > 1:
            for f in sorted(files):
                if fnmatch.fnmatch(f, toks[1]) and _plugstack_has_pyxis(files, f, depth + 1):
                    return True
        elif toks[0] in ("required", "optional") and len(toks) > 1 and "spank_pyxis" in toks[1]:
            return True
    return False


def _pyxis_client_ok(node) -> bool:
    has = _plugstack_has_pyxis(node.files)
    if has is None:  # no plugstack.conf on this host: assume the shared (configless) one
        return True
    return bool(has) and node.get("pyxis_installed", True) is not False


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
    tl = parse_time(opts["time"]) if "time" in opts else None
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
    return job


def _new_job(state, job):
    s = state.slurm
    job["id"] = s.next_job_id
    s.next_job_id += 1
    s.jobs.append(job)
    j = s.jobs[-1]
    _job_defaults(state, j)
    return j


# ====================================================================== sbatch


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
    else:
        raise _err("sbatch: error: Batch script is empty!")
    _need_ctl(ctx, "sbatch: error: Batch job submission failed: ")
    _begin(ctx)
    job = _build_job(ctx, "sbatch", opts)
    if not job["name"]:
        job["name"] = "wrap" if "wrap" in opts else (cmd_path.rsplit("/", 1)[-1] if rest else "sbatch")
    job.update({"script": script, "command": cmd_path, "batch": True, "duration": BATCH_DEFAULT_DURATION})
    j = _new_job(state, job)
    out_pat = opts.get("output") or "slurm-%j.out"
    j.stdout = _expand_pattern(out_pat, j)
    if not j.stdout.startswith("/"):
        j.stdout = j.work_dir.rstrip("/") + "/" + j.stdout
    schedule(state)
    if opts.get("parsable"):
        return f"{j.id}\n"
    return f"Submitted batch job {j.id}\n"


def _expand_pattern(pat, j):
    return (
        pat.replace("%j", str(j.id))
        .replace("%x", j.name)
        .replace("%u", j.user)
        .replace("%N", j.nodelist[0] if j.nodelist else "node")
        .replace("%A", str(j.id))
    )


def _job_env(j, node_name):
    return {
        "SLURM_JOB_ID": str(j.id),
        "SLURM_JOBID": str(j.id),
        "SLURM_JOB_NAME": j.name,
        "SLURM_JOB_NODELIST": compress_hostlist(j.nodelist),
        "SLURM_NNODES": str(j.num_nodes),
        "SLURM_JOB_NUM_NODES": str(j.num_nodes),
        "SLURM_NTASKS": str(j.num_nodes * j.ntasks_per_node),
        "SLURM_NTASKS_PER_NODE": str(j.ntasks_per_node),
        "SLURM_JOB_PARTITION": j.partition,
        "SLURM_SUBMIT_DIR": j.work_dir,
        "SLURMD_NODENAME": node_name,
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(len(j.alloc_gpus.get(node_name, [])))),
        "SLURM_JOB_GPUS": ",".join(str(i) for i in j.alloc_gpus.get(node_name, [])),
        "SLURM_JOB_ACCOUNT": j.account,
        "SLURM_JOB_USER": j.user,
        "SLURM_CPUS_ON_NODE": str(2 * len((j.get("alloc_cores") or {}).get(node_name, [])) or j.ntasks_per_node),
    }


def _run_batch(state, j):
    """Execute the batch script on the first allocated node and write its output file."""
    from ..core.shell import Shell

    host = j.nodelist[0]
    node = state.nodes.get(host)
    if node is None:
        return
    env = _job_env(j, host)
    saved = dict(node.env)
    node.env.update(env)
    out = []
    _GUARD["depth"] += 1
    failed = False
    try:
        from .storage import path_available

        if not path_available(state, host, j.work_dir):
            out.append(
                f"slurmstepd: error: couldn't chdir to `{j.work_dir}': No such file or directory: going to /tmp instead"
            )
        sh = Shell(state, host)
        sh.ctx.cwd = j.work_dir
        _CURRENT.append(j)
        for line in j.script.splitlines()[1:]:
            st = line.strip()
            if not st or st.startswith("#"):
                continue
            if st.startswith(("set ", "module ", "source ", "cd ")):
                continue
            simple = not any(c in st for c in "|;&><$`") and not st.startswith(("srun", "export"))
            if simple:
                try:
                    argv = shlex.split(st)
                except ValueError:
                    argv = []
                res, code = _exec_task(_FakeCtx(state, host), j, host, argv, bool(j.container_image))
                if code:
                    failed = True
                    j.final_exit = f"{code}:0"
                if argv and argv[0] == "exit":
                    if res:
                        out.append(res)
                    break
            else:
                res = sh.run(st)
                if "Exited with exit code" in res or "command not found" in res:
                    failed = True
            if res:
                out.append(res)
    except Exception as e:  # never let a script crash the simulator
        out.append(f"slurmstepd: error: {e}")
        failed = True
    finally:
        _GUARD["depth"] -= 1
        if _CURRENT and _CURRENT[-1] is j:
            _CURRENT.pop()
        node.env.clear()
        node.env.update(saved)
    if failed:
        j.final_exit = j.get("final_exit") or "1:0"
        j.final_state = "FAILED"
    text = "\n".join(out) + ("\n" if out else "")
    sub = state.nodes.get(j.submit_host) or state.nodes.get(HEAD)
    sub.files[j.stdout] = text
    if host != j.submit_host:
        node.files[j.stdout] = text


_CURRENT: list = []


class _FakeCtx:
    def __init__(self, state, host):
        self.state = state
        self.host = host
        self.host_stack = []


# ====================================================================== srun / salloc


def _container_python(code, j, gpus_here):
    out = []
    for m in re.finditer(r"print\((.*?)\)(?:;|$)", code):
        expr = m.group(1).strip()
        vals = []
        for part in [x.strip() for x in expr.split(",")]:
            if part == "torch.__version__":
                vals.append("2.4.0a0+07cecf4168.nv24.05")
            elif part == "torch.cuda.device_count()":
                vals.append(str(gpus_here))
            elif part == "torch.cuda.is_available()":
                vals.append(str(gpus_here > 0))
            elif part == "torch.version.cuda":
                vals.append("12.4")
            elif part.startswith("torch.cuda.get_device_name"):
                vals.append("NVIDIA H100 80GB HBM3")
            elif part.startswith(("'", '"')):
                vals.append(part.strip("'\""))
            else:
                vals.append(part)
        out.append(" ".join(vals))
    return "\n".join(out)


def _exec_task(ctx, j, host, argv, container):
    """Run one task's command on ``host``. Returns (output, exit_code)."""
    from ..core.shell import Context
    from . import dispatch

    state = ctx.state
    node = state.nodes[host]
    gidx = (j.alloc_gpus or {}).get(host, [])
    constrain = (
        "ConstrainDevices=yes" in state.nodes[HEAD].files.get("/etc/slurm/cgroup.conf", "")
        if HEAD in state.nodes
        else True
    )
    if not argv:
        return "", 0
    prog = argv[0].rsplit("/", 1)[-1]
    if prog in ("true", ":", "sleep", "wait"):
        return "", 0
    if prog == "false":
        return "", 1
    if prog == "exit":
        return "", int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 0
    if prog == "nvidia-smi" and argv[1:] in (["-L"], ["--list-gpus"]):
        if not node.get("driver_loaded", True):
            return (
                "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
                "Make sure that the latest NVIDIA driver is installed and running."
            ), 9
        gpus = [node.gpus[i] for i in gidx if i < len(node.gpus)] if constrain else list(node.gpus)
        if not gpus and constrain:
            return "No devices found.", 6
        lines = []
        for k, g in enumerate(gpus):
            if g.fallen_off_bus:
                lines.append(f"Unable to determine the device handle for GPU{g.bus_id}: Unknown Error")
                continue
            lines.append(f"GPU {k if constrain else g.index}: {g.name} (UUID: {g.uuid})")
        return "\n".join(lines), 0
    if prog in ("python", "python3") and "-c" in argv:
        code = argv[argv.index("-c") + 1] if argv.index("-c") + 1 < len(argv) else ""
        if "torch" in code and not container:
            return (
                'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\n'
                "ModuleNotFoundError: No module named 'torch'"
            ), 1
        return _container_python(code, j, len(gidx)), 0
    if container and prog == "cat" and argv[1:] == ["/etc/os-release"]:
        return 'PRETTY_NAME="Ubuntu 22.04.4 LTS"\nNAME="Ubuntu"\nVERSION_ID="22.04"', 0
    if prog in ("bash", "sh") and "-c" in argv:
        from ..core.shell import Shell

        sh = Shell(state, host)
        return sh.run(argv[argv.index("-c") + 1]), 0
    if prog == "env" and not argv[1:]:
        env = dict(node.env)
        env.update(_job_env(j, host))
        return "\n".join(f"{k}={v}" for k, v in sorted(env.items())), 0
    if prog == "printenv" and argv[1:]:
        env = _job_env(j, host)
        env.update({k: v for k, v in node.env.items()})
        return "\n".join(env.get(a, "") for a in argv[1:]), 0
    sub = Context(state=state, host=host)
    sub.cwd = j.work_dir
    saved = dict(node.env)
    node.env.update(_job_env(j, host))
    try:
        out = dispatch(sub, argv)
        code = 0
    except CommandError as e:
        out, code = str(e), e.code
        if "command not found" in out:
            out = f"slurmstepd: error: execve(): {argv[0]}: No such file or directory"
            code = 2
    finally:
        node.env.clear()
        node.env.update(saved)
    return (out or "").rstrip("\n"), code


def _pyxis_import(state, host, image):
    m = re.match(r"^(?:(?P<user>[^@#/]+)@)?(?:(?P<reg>[^#]+)#)?(?P<img>[^:#]+)(?::(?P<tag>[^#]+))?$", image or "")
    lines = [f"pyxis: importing docker image: {image}"]
    if image.endswith(".sqsh") or image.startswith("/"):
        node = state.nodes[host]
        if image in node.files or image in node.get("enroot_images", []):
            return True, []
        return False, [
            f"slurmstepd: error: pyxis: unable to open squashfs image {image}: No such file or directory",
            "slurmstepd: error: pyxis: couldn't start container",
            "slurmstepd: error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1",
            "slurmstepd: error: Failed to invoke spank plugin stack",
        ]
    if not m:
        return False, lines + ["slurmstepd: error: pyxis: invalid image name"]
    reg = m.group("reg") or "registry-1.docker.io"
    img = m.group("img")
    tag = m.group("tag") or "latest"
    catalog = state.get("ngc", {}).get("catalog", {})
    ok = True
    err = None
    if reg == "nvcr.io":
        if img not in catalog:
            ok = False
            err = (
                f"URL https://nvcr.io/proxy_auth?scope=repository:{img}:pull returned error code: 401 Unauthorized"
                if not img.startswith("nvidia/")
                else f"URL https://nvcr.io/v2/{img}/manifests/{tag} returned error code: 404 Not Found"
            )
        elif tag not in catalog[img]:
            ok = False
            err = f"URL https://nvcr.io/v2/{img}/manifests/{tag} returned error code: 404 Not Found"
    elif reg == "registry-1.docker.io":
        first = img.split("/", 1)[0]
        if "." in first:  # e.g. nvcr.io/nvidia/pytorch: registry written with '/' instead of '#'
            ok = False
            err = f"URL https://registry-1.docker.io/v2/{img}/manifests/{tag} returned error code: 401 Unauthorized"
    if ok:
        node = state.nodes[host]
        node.setdefault("enroot_images", [])
        if image not in node.enroot_images:
            node.enroot_images.append(image)
        return True, [lines[0], f"pyxis: imported docker image: {image}"]
    return False, lines + [
        f"slurmstepd: error: pyxis: child {23000 + len(image)} failed with error code: 1",
        "slurmstepd: error: pyxis: failed to import docker image",
        "slurmstepd: error: pyxis: printing enroot log file:",
        "slurmstepd: error: pyxis:     [INFO] Querying registry for permission grant",
        "slurmstepd: error: pyxis:     [INFO] Authenticating with user: <anonymous>",
        "slurmstepd: error: pyxis:     [INFO] Authentication succeeded",
        "slurmstepd: error: pyxis:     [INFO] Fetching image manifest list",
        f"slurmstepd: error: pyxis:     [ERROR] {err}",
        "slurmstepd: error: pyxis: couldn't start container",
        "slurmstepd: error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1",
        "slurmstepd: error: Failed to invoke spank plugin stack",
    ]


def _run_step(ctx, j, opts, argv):
    """Run a job step on the job's nodes. Returns (output, rc)."""
    state = ctx.state
    label = opts.get("label") or opts.get("l")
    nodes = list(j.nodelist)
    if "nodes" in opts:
        nodes = nodes[: int(str(opts["nodes"]).split("-")[0])]
    tpn = int(opts["ntasks_per_node"]) if "ntasks_per_node" in opts else None
    if "ntasks" in opts:
        total = int(opts["ntasks"])
    else:
        total = len(nodes) * (tpn or j.ntasks_per_node)
    per = tpn or max(1, math.ceil(total / max(1, len(nodes))))
    image = opts.get("container_image") or (j.container_image if not j.batch else "")
    out = []
    rc = 0
    step_id = len(j.steps)
    task = 0
    for host in nodes:
        node = state.nodes[host]
        if image:
            if not _plugstack_has_pyxis(node.files) or node.get("pyxis_installed", True) is False:
                out.append(
                    f'slurmstepd: error: spank: plugin "pyxis" is not loaded on {host} '
                    f"(check {PLUGSTACK}); option --container-image cannot be honoured"
                )
                out.append(f"srun: error: {host}: task {task}: Exited with exit code 1")
                rc = 1
                task += per
                continue
            ok, msgs = _pyxis_import(state, host, image)
            out.extend(msgs)
            if not ok:
                for t in range(per):
                    out.append(f"srun: error: {host}: task {task + t}: Exited with exit code 1")
                rc = 1
                task += per
                continue
        for k in range(per):
            if task >= total:
                break
            _TASK.update({"local": k, "per_node": per})
            try:
                text, code = _exec_task(ctx, j, host, argv, bool(image))
            finally:
                _TASK.update({"local": None, "per_node": 1})
            if text:
                out.extend((f"{task}: {l}" if label else l) for l in text.split("\n"))
            if code:
                out.append(f"srun: error: {host}: task {task}: Exited with exit code {code}")
                rc = code
            task += 1
    j.steps.append(
        {
            "id": str(step_id),
            "name": argv[0] if argv else "",
            "state": "FAILED" if rc else "COMPLETED",
            "exit_code": f"{rc}:0",
            "nodes": compress_hostlist(nodes),
            "container_image": image,
        }
    )
    if image and not j.container_image:
        j.container_image = image
    return "\n".join(out), rc


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
    if opts.get("pty") and cmd[0] in ("bash", "/bin/bash", "sh", "/bin/sh"):
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
    return out + ("\n" if out else "")


@command("salloc", help="get an interactive allocation: salloc -N1 --gres=gpu:8 (then srun ...; scancel to release)")
def salloc(ctx, args):
    state = ctx.state
    _ensure(state)
    opts, cmd = parse_job_opts("salloc", args, pyxis=_pyxis_client_ok(ctx.node))
    _need_ctl(ctx, "salloc: error: Unable to allocate resources: ")
    _begin(ctx)
    j, _ = _alloc_now(ctx, "salloc", opts)
    j.name = opts.get("name") or "interactive"
    ctx.node.env["SLURM_JOB_ID"] = str(j.id)
    ctx.node.env["SLURM_JOB_NODELIST"] = compress_hostlist(j.nodelist)
    return (
        f"salloc: Granted job allocation {j.id}\n"
        f"salloc: Nodes {compress_hostlist(j.nodelist)} are ready for job\n"
        f"(SLURM_JOB_ID={j.id} exported; run 'srun ...' for steps, 'scancel {j.id}' to release)\n"
    )


# ====================================================================== scancel


@command("scancel", help="cancel jobs: scancel ID... | -u USER | -p PART | -t STATE | -n NAME")
def scancel(ctx, args):
    state = ctx.state
    _need_ctl(ctx, "scancel: error: Kill job error: ")
    _begin(ctx)
    user = _opt(args, "-u", "--user")
    part = _opt(args, "-p", "--partition")
    st = _opt(args, "-t", "--state")
    name = _opt(args, "-n", "--name", "--jobname")
    skip = set()
    for i, a in enumerate(args):
        if a in ("-u", "--user", "-p", "--partition", "-t", "--state", "-n", "--name", "--jobname"):
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
    for j in targets:
        _finish(state, j, "CANCELLED", "0:0")
    schedule(state)
    if out:
        raise CommandError("\n".join(out), 1)
    return ""


# ====================================================================== sacct

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
    "reqcpus": 8,
    "allocnodes": 10,
    "submitline": 40,
    "derivedexitcode": 10,
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
    "reqcpus": "ReqCPUS",
    "allocnodes": "AllocNodes",
    "submitline": "SubmitLine",
    "derivedexitcode": "DerivedExitCode",
}


def _sacct_rows(state, j, steps=True):
    clock = state.slurm.clock
    elapsed = 0
    if j.start_time is not None:
        elapsed = (j.end_time if j.end_time is not None else clock) - j.start_time
    ncpu = j.num_nodes * j.ntasks_per_node * j.cpus_per_task
    acpu = _alloc_cpu_count(state, j)
    gtres = f"gres/gpu={_job_gpus(j)}," if j.gpus_per_node else ""
    tres = f"billing={ncpu},cpu={ncpu}," + gtres + f"mem=0,node={j.num_nodes}"
    atres = f"billing={acpu},cpu={acpu}," + gtres + f"mem=0,node={j.num_nodes}"
    st = j.state
    if st == "CANCELLED" and j.start_time is not None:
        st = "CANCELLED by 0"
    base = {
        "jobid": str(j.id),
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
        "maxrss": "",
        "reqcpus": str(ncpu),
        "allocnodes": str(j.num_nodes),
        "submitline": j.command,
        "derivedexitcode": "0:0",
    }
    rows = [base]
    if not steps or j.start_time is None:
        return rows

    def step(sid, name, state_, code, nodes=None):
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
            }
        )
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
    ids = _opt(args, "-j", "--jobs")
    fmt = _opt(args, "-o", "--format")
    user = _opt(args, "-u", "--user")
    states = _opt(args, "-s", "--state")
    noheader = _has(args, "-n", "--noheader")
    parsable = _has(args, "-p", "--parsable", "-P", "--parsable2")
    allocs = _has(args, "-X", "--allocations")
    cols = [
        c.strip() for c in (fmt or "JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode").split(",") if c.strip()
    ]
    spec = []
    for c in cols:
        name, _, w = c.partition("%")
        key = name.lower()
        if key not in SACCT_HDR:
            raise _err(f'sacct: error: Invalid field requested: "{name}"')
        spec.append((key, int(w) if w else SACCT_W.get(key, 10)))
    rows = []
    for j in state.slurm.jobs:
        if ids and str(j.id) not in [x.split(".")[0] for x in ids.split(",")]:
            continue
        if user and j.user != user:
            continue
        if states:
            want = {s_.upper() for s_ in states.split(",")}
            if j.state not in want and STATE_CODES.get(j.state) not in want:
                continue
        rows.extend(_sacct_rows(state, j, steps=not allocs))
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


# ====================================================================== sacctmgr

QOS_KEYS = {
    "maxtresperuser": "max_tres_per_user",
    "maxtrespu": "max_tres_per_user",
    "maxtresperjob": "max_tres_per_job",
    "maxtres": "max_tres_per_job",
    "maxjobsperuser": "max_jobs_per_user",
    "maxjobspu": "max_jobs_per_user",
    "maxjobs": "max_jobs_per_user",
    "maxwall": "max_wall",
    "maxwalldurationperjob": "max_wall",
    "priority": "priority",
    "grptres": "grp_tres",
    "flags": "flags",
    "preempt": "preempt",
    "description": "description",
    "maxsubmitjobsperuser": "max_submit_pu",
    "maxsubmitpu": "max_submit_pu",
    "usagefactor": "usage_factor",
    "gracetime": "grace_time",
    "preemptmode": "preempt_mode",
}
QOS_COLS = {
    "name": ("Name", 10, "name"),
    "priority": ("Priority", 10, "priority"),
    "gracetime": ("GraceTime", 10, "grace_time"),
    "preempt": ("Preempt", 10, "preempt"),
    "preemptmode": ("PreemptMode", 11, "preempt_mode"),
    "flags": ("Flags", 20, "flags"),
    "usagefactor": ("UsageFactor", 11, "usage_factor"),
    "grptres": ("GrpTRES", 13, "grp_tres"),
    "maxtres": ("MaxTRES", 13, "max_tres_per_job"),
    "maxtresperjob": ("MaxTRES", 13, "max_tres_per_job"),
    "maxwall": ("MaxWall", 11, "max_wall"),
    "maxtrespu": ("MaxTRESPU", 13, "max_tres_per_user"),
    "maxtresperuser": ("MaxTRESPU", 13, "max_tres_per_user"),
    "maxjobspu": ("MaxJobsPU", 9, "max_jobs_per_user"),
    "maxjobsperuser": ("MaxJobsPU", 9, "max_jobs_per_user"),
    "maxsubmitpu": ("MaxSubmitPU", 11, "max_submit_pu"),
    "description": ("Descr", 11, "description"),
}
QOS_DEFAULT_COLS = [
    "name",
    "priority",
    "gracetime",
    "preempt",
    "preemptmode",
    "flags",
    "usagefactor",
    "grptres",
    "maxtres",
    "maxwall",
    "maxtrespu",
    "maxjobspu",
    "maxsubmitpu",
]


def _fmt_table(cols, rows, parsable=False, noheader=False, left=()):
    out = []
    if parsable:
        if not noheader:
            out.append("|".join(h for h, _ in cols) + "|")
        for r in rows:
            out.append("|".join(str(v) for v in r) + "|")
        return "\n".join(out) + "\n"
    if not noheader:
        out.append(" ".join(h[:w].rjust(w) for h, w in cols))
        out.append(" ".join("-" * w for _, w in cols))
    for r in rows:
        out.append(
            " ".join(
                (str(v)[:w].ljust(w) if i in left else str(v)[:w].rjust(w))
                for i, ((_, w), v) in enumerate(zip(cols, r))
            ).rstrip()
        )
    return "\n".join(out) + "\n"


def _merge_tres(old, new):
    cur = {}
    for tok in str(old or "").split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            cur[k] = v
    for tok in str(new).split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = "gres/gpu" if k == "gpu" else k
            if v in ("-1", ""):
                cur.pop(k, None)
            else:
                cur[k] = v
    order = ["cpu", "mem", "node", "billing"]
    keys = sorted(cur, key=lambda k: (order.index(k) if k in order else 10, k))
    return ",".join(f"{k}={cur[k]}" for k in keys)


def _set_qos_field(q, key, val):
    lk = key.lower()
    if lk not in QOS_KEYS:
        raise _err(f" Unknown option: {key}={val}\n Use keyword 'where' to modify condition")
    field = QOS_KEYS[lk]
    if field in ("max_tres_per_user", "max_tres_per_job", "grp_tres"):
        q[field] = _merge_tres(q.get(field), val)
    elif field in ("priority",):
        q[field] = int(val)
    elif field in ("max_jobs_per_user", "max_submit_pu"):
        q[field] = None if val in ("-1", "") else int(val)
    elif field == "max_wall":
        q[field] = None if val in ("-1", "") else val
    elif field == "flags":
        q[field] = "" if val in ("-1", "") else val
    else:
        q[field] = val


def _sacctmgr_split(args):
    """-> (entity, names, where{}, set{})"""
    entity = args[0].lower() if args else ""
    names, where, sets = [], {}, {}
    mode = "where"
    for a in args[1:]:
        la = a.lower()
        if la == "where":
            mode = "where"
            continue
        if la == "set":
            mode = "set"
            continue
        if "=" in a:
            k, v = a.split("=", 1)
            op = "="
            if k.endswith("+") or k.endswith("-"):
                op = k[-1] + "="
                k = k[:-1]
            (sets if mode == "set" else where)[k.lower()] = (v, op)
        else:
            names.extend(x for x in a.split(",") if x)
    if "name" in where:
        names += where.pop("name")[0].split(",")
    if "names" in where:
        names += where.pop("names")[0].split(",")
    return entity, names, where, sets


def _confirm(immediate):
    return (
        ""
        if immediate
        else "Would you like to commit changes? (You have 30 seconds to decide)\n(N/y): y  [simulator auto-confirmed; use -i to skip this prompt]\n"
    )


@command("sacctmgr", help="accounting manager: sacctmgr [-i] add|modify|show|delete qos|account|user|assoc ...")
def sacctmgr(ctx, args):
    state = ctx.state
    _ensure(state)
    immediate = False
    parsable = False
    noheader = False
    while args and args[0].startswith("-"):
        if args[0] in ("-i", "--immediate"):
            immediate = True
        elif args[0] in ("-p", "-P", "--parsable", "--parsable2"):
            parsable = True
        elif args[0] in ("-n", "--noheader"):
            noheader = True
        elif args[0] in ("-V", "--version"):
            return f"slurm {SLURM_VERSION}\n"
        args = args[1:]
    if not args:
        raise _err("sacctmgr: no command given")
    head = state.nodes.get(HEAD)
    if head is not None and head.services.get("slurmdbd", "active") != "active":
        raise _err(
            "sacctmgr: error: Sending PersistInit msg: Connection refused\n"
            "sacctmgr: error: Problem talking to the database: Connection refused"
        )
    verb = args[0].lower()
    s = state.slurm
    if verb in ("add", "create"):
        entity, names, where, sets = _sacctmgr_split(args[1:])
        allkv = {**where, **sets}
        if entity == "qos":
            if not names:
                raise _err(" Need name of qos to add.")
            out = [" Adding QOS(s)"]
            for n in names:
                if n in s.qos:
                    return f" Nothing new added.\nsacctmgr: error: QOS {n} already exists\n"
                q = {
                    "name": n,
                    "priority": 0,
                    "max_tres_per_user": "",
                    "preempt": "",
                    "max_tres_per_job": "",
                    "max_jobs_per_user": None,
                    "max_wall": None,
                    "grp_tres": "",
                    "flags": "",
                    "description": n,
                }
                for k, (v, _) in allkv.items():
                    _set_qos_field(q, k, v)
                s.qos[n] = q
                out.append(f"  {n}")
            out.append(" Settings")
            out.append(f"  Description    = {names[0]}")
            for k, (v, _) in allkv.items():
                out.append(f"  {k:<15}= {v}")
            return "\n".join(out) + "\n" + _confirm(immediate)
        if entity == "account":
            if not names:
                raise _err(" Need name of account to add.")
            parent = allkv.get("parent", ("root", "="))[0] or "root"
            if parent not in s.accounts:
                raise _err(f" Parent account {parent} doesn't exist.\nsacctmgr: error: Problem adding accounts")
            new = [n for n in names if n not in s.accounts]
            if not new:
                return " Nothing new added.\n"
            try:
                shares = int(allkv.get("fairshare", ("1", "="))[0])
            except ValueError:
                raise _err(" Bad FairShare value. Use a number, or 'parent'.") from None
            out = [" Adding Account(s)"]
            for n in new:
                s.accounts[n] = {
                    "name": n,
                    "fairshare": shares,
                    "users": [],
                    "parent": parent,
                    "description": allkv.get("description", (n, "="))[0],
                    "organization": allkv.get("organization", (n, "="))[0],
                }
                out.append(f"  {n}")
            out += [
                " Settings",
                f"  Description     = {s.accounts[new[0]].description}",
                f"  Organization    = {s.accounts[new[0]].organization}",
                " Associations",
            ]
            out += [f"  A = {n:<10} C = superpod  " for n in new]
            out += [" Settings", f"  Parent        = {parent}"]
            if "fairshare" in allkv:
                out.append(f"  Fairshare     = {shares}")
            return "\n".join(out) + "\n" + _confirm(immediate)
        if entity == "user":
            if not names:
                raise _err(" Need name of user to add.")
            accts = [a for a in allkv.get("account", allkv.get("accounts", ("", "=")))[0].split(",") if a]
            dflt = allkv.get("defaultaccount", ("", "="))[0]
            if not accts and dflt:
                accts = [dflt]
            if not accts:
                raise _err(" Need name of account to add user to.")
            for acct in accts + ([dflt] if dflt else []):
                if acct not in s.accounts:
                    raise _err(
                        f" There is no uncommitted association for account {acct}\n"
                        f"sacctmgr: error: Account {acct} does not exist"
                    )
            out = [" Adding User(s)"]
            assocs = []
            for n in names:
                u = s.users.get(n)
                if u is None:
                    s.users[n] = {
                        "name": n,
                        "default_account": dflt or accts[0],
                        "accounts": [],
                        "qos": ["normal"],
                        "default_qos": "",
                        "fairshare_factor": 0.5,
                        "shares": {},
                    }
                    u = s.users[n]
                elif dflt:
                    u.default_account = dflt
                u.setdefault("shares", {})
                for acct in accts:
                    if acct not in u.accounts:
                        u.accounts.append(acct)
                        assocs.append((n, acct))
                    if n not in s.accounts[acct].users:
                        s.accounts[acct].users.append(n)
                    if "fairshare" in allkv:
                        u.shares[acct] = int(allkv["fairshare"][0])
                if "qos" in allkv:
                    u.qos = allkv["qos"][0].split(",")
                if "defaultqos" in allkv:
                    u.default_qos = allkv["defaultqos"][0]
                out.append(f"  {n}")
            if not assocs:
                return " Nothing new added.\n"
            out += [" Settings", f"  Default Account = {s.users[names[0]].default_account}", " Associations"]
            out += [f"  U = {n:<9} A = {a:<10} C = superpod  " for n, a in assocs]
            return "\n".join(out) + "\n" + _confirm(immediate)
        raise _err(f" Unknown entity: {entity}")
    if verb in ("modify", "update"):
        entity, names, where, sets = _sacctmgr_split(args[1:])
        if not sets:
            raise _err(" You didn't give me anything to set")
        if entity == "qos":
            targets = [n for n in names if n in s.qos]
            if not targets:
                return " Nothing modified\n"
            for n in targets:
                for k, (v, _) in sets.items():
                    _set_qos_field(s.qos[n], k, v)
            out = [" Modified qos..."] + [f"  {n}" for n in targets]
            schedule(state)
            return "\n".join(out) + "\n" + _confirm(immediate)
        if entity == "user":
            targets = [n for n in names if n in s.users]
            if not targets:
                return " Nothing modified\n"
            for n in targets:
                u = s.users[n]
                for k, (v, op) in sets.items():
                    if k == "qos":
                        vals = v.split(",")
                        if op == "+=":
                            u.qos = list(dict.fromkeys(list(u.qos) + vals))
                        elif op == "-=":
                            u.qos = [x for x in u.qos if x not in vals]
                        else:
                            u.qos = vals
                    elif k == "defaultqos":
                        u.default_qos = v
                    elif k == "defaultaccount":
                        if v not in u.get("accounts", []):
                            raise _err(
                                f" Can't modify because these users aren't associated with new default account '{v}'...\n"
                                f"  U = {n}\n"
                                f"sacctmgr: error: Add the association first: sacctmgr add user {n} account={v}"
                            )
                        u.default_account = v
                    elif k == "fairshare":
                        u.setdefault("shares", {})
                        for acct in where.get("account", (",".join(u.get("accounts", [])), "="))[0].split(","):
                            if acct in u.get("accounts", []):
                                u.shares[acct] = int(v)
                    else:
                        raise _err(f" Unknown option: {k}={v}")
            out = [" Modified user associations..."] + [
                f"  C = superpod    A = {s.users[n].default_account:<10} U = {n}" for n in targets
            ]
            schedule(state)
            return "\n".join(out) + "\n" + _confirm(immediate)
        if entity == "account":
            targets = [n for n in names if n in s.accounts]
            for n in targets:
                for k, (v, _) in sets.items():
                    if k == "fairshare":
                        s.accounts[n].fairshare = int(v)
                    elif k == "parent":
                        if v not in s.accounts or v == n:
                            raise _err(f" Parent account {v} doesn't exist.")
                        s.accounts[n].parent = v
                    elif k in ("description", "organization"):
                        s.accounts[n][k] = v
                    else:
                        raise _err(f" Unknown option: {k}={v}")
            if targets:
                schedule(state)
            return (
                (" Modified account associations...\n" + "".join(f"  {n}\n" for n in targets) + _confirm(immediate))
                if targets
                else " Nothing modified\n"
            )
        raise _err(f" Unknown entity: {entity}")
    if verb in ("delete", "remove"):
        entity, names, where, _ = _sacctmgr_split(args[1:])
        table = {"qos": s.qos, "account": s.accounts, "user": s.users}.get(entity)
        if table is None:
            raise _err(f" Unknown entity: {entity}")
        gone = [n for n in names if n in table]
        if entity == "qos" and any(j.qos in gone and j.state in ("PENDING", "RUNNING") for j in s.jobs):
            raise _err(" Error with request: Job(s) active, cancel job(s) before remove\n  JobId = ...")
        if entity == "user" and "account" in where:
            accts = where["account"][0].split(",")
            done = []
            for n in gone:
                u = s.users[n]
                for a in accts:
                    if a in u.get("accounts", []):
                        u.accounts.remove(a)
                        done.append(f"  C = superpod    A = {a:<10} U = {n}")
                    if a in s.accounts and n in s.accounts[a].users:
                        s.accounts[a].users.remove(n)
                if not u.accounts:
                    del s.users[n]
                elif u.default_account not in u.accounts:
                    u.default_account = u.accounts[0]
            if not done:
                return " Nothing deleted\n"
            return " Deleting user associations...\n" + "\n".join(done) + "\n" + _confirm(immediate)
        if entity == "account":
            for n in gone:
                if _acct_children(s, n):
                    raise _err(
                        f" Error with request: account {n} has child accounts; remove or move them first\n"
                        "sacctmgr: error: Problem deleting accounts"
                    )
            for n in gone:
                for u in s.users.values():
                    if n in u.get("accounts", []):
                        u.accounts.remove(n)
                        if u.default_account == n and u.accounts:
                            u.default_account = u.accounts[0]
        if entity == "user":
            for n in gone:
                for a in s.accounts.values():
                    if n in a.users:
                        a.users.remove(n)
        for n in gone:
            del table[n]
        return (
            (f" Deleting {entity}(s)...\n" + "".join(f"  {n}\n" for n in gone) + _confirm(immediate))
            if gone
            else " Nothing deleted\n"
        )
    if verb in ("show", "list"):
        entity, names, where, _ = _sacctmgr_split(args[1:])
        fmt = where.pop("format", (None, ""))[0]
        if entity == "qos":
            keys = [k.split("%")[0].lower() for k in fmt.split(",")] if fmt else QOS_DEFAULT_COLS
            for k in keys:
                if k not in QOS_COLS:
                    raise _err(f'sacctmgr: error: Invalid field requested: "{k}"')
            cols = [(QOS_COLS[k][0], QOS_COLS[k][1]) for k in keys]
            rows = []
            for q in s.qos.values():
                if names and q.name not in names:
                    continue
                row = []
                for k in keys:
                    v = q.get(QOS_COLS[k][2], "")
                    if v is None:
                        v = ""
                    if k in ("gracetime",) and not v:
                        v = "00:00:00"
                    if k == "preemptmode" and not v:
                        v = "cluster"
                    if k == "usagefactor" and not v:
                        v = "1.000000"
                    if k == "maxwall" and v:
                        v = fmt_limit_upper(parse_time(v))
                    row.append(v)
                rows.append(row)
            return _fmt_table(cols, rows, parsable, noheader)
        if entity in ("assoc", "association", "associations"):
            tree = "tree" in [n.lower() for n in names]
            names = [n for n in names if n.lower() != "tree"]
            return _show_assoc(state, fmt, tree, where, names, parsable, noheader)
        if entity in ("user", "users"):
            withassoc = "withassoc" in [n.lower() for n in names]
            names = [n for n in names if n.lower() not in ("withassoc", "withcoord")]
            keys = [k.split("%")[0].lower() for k in fmt.split(",")] if fmt else ["user", "defaultaccount", "admin"]
            if withassoc and not fmt:
                keys += ["cluster", "account", "partition", "share", "qos", "defaultqos"]
            hdr = {
                "user": ("User", 10),
                "defaultaccount": ("Def Acct", 10),
                "admin": ("Admin", 9),
                "adminlevel": ("Admin", 9),
                "cluster": ("Cluster", 10),
                "account": ("Account", 10),
                "partition": ("Partition", 10),
                "share": ("Share", 9),
                "fairshare": ("Share", 9),
                "qos": ("QOS", 20),
                "defaultqos": ("Def QOS", 9),
            }
            for k in keys:
                if k not in hdr:
                    raise _err(f'sacctmgr: error: Invalid field requested: "{k}"')
            rows = []
            for u in s.users.values():
                if names and u.name not in names:
                    continue
                per_assoc = withassoc or "account" in keys
                accts = list(u.get("accounts") or [u.default_account]) if per_assoc else [None]
                for a in accts:
                    vals = {
                        "user": u.name,
                        "defaultaccount": u.default_account,
                        "admin": "Administ+" if u.name == "root" else "None",
                        "cluster": "superpod",
                        "account": a or "",
                        "partition": "",
                        "share": _assoc_shares(s, a, u.name) if a else "",
                        "qos": ",".join(u.get("qos", [])),
                        "defaultqos": u.get("default_qos", ""),
                    }
                    vals["adminlevel"] = vals["admin"]
                    vals["fairshare"] = vals["share"]
                    rows.append([vals[k] for k in keys])
            return _fmt_table([hdr[k] for k in keys], rows, parsable, noheader)
        if entity in ("account", "accounts"):
            names = [n for n in names if n.lower() not in ("withassoc", "withcoord")]
            keys = [k.split("%")[0].lower() for k in fmt.split(",")] if fmt else ["account", "descr", "org"]
            hdr = {
                "account": ("Account", 10),
                "descr": ("Descr", 20),
                "description": ("Descr", 20),
                "org": ("Org", 20),
                "organization": ("Org", 20),
                "parentname": ("Par Name", 10),
                "fairshare": ("Share", 9),
                "share": ("Share", 9),
            }
            for k in keys:
                if k not in hdr:
                    raise _err(f'sacctmgr: error: Invalid field requested: "{k}"')
            rows = []
            for a in s.accounts.values():
                if names and a.name not in names:
                    continue
                vals = {
                    "account": a.name,
                    "descr": a.description,
                    "org": a.organization,
                    "parentname": "" if a.name == "root" else a.get("parent") or "root",
                    "fairshare": a.fairshare,
                }
                vals["description"], vals["organization"], vals["share"] = vals["descr"], vals["org"], vals["fairshare"]
                rows.append([vals[k] for k in keys])
            return _fmt_table([hdr[k] for k in keys], rows, parsable, noheader)
        if entity == "cluster":
            return _fmt_table(
                [("Cluster", 10), ("ControlHost", 15), ("ControlPort", 12), ("RPC", 5)],
                [["superpod", "10.141.255.254", "6817", "10240"]],
                parsable,
                noheader,
            )
        raise _err(f" Unknown entity: {entity}")
    raise _err(f"sacctmgr: error: Invalid command: {args[0]}")


# ====================================================================== sprio / sshare


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


ASSOC_COLS = {
    "cluster": ("Cluster", 10),
    "account": ("Account", 20),
    "user": ("User", 10),
    "partition": ("Partition", 10),
    "share": ("Share", 9),
    "fairshare": ("Share", 9),
    "priority": ("Priority", 10),
    "grpjobs": ("GrpJobs", 7),
    "grptres": ("GrpTRES", 13),
    "maxjobs": ("MaxJobs", 7),
    "maxtres": ("MaxTRES", 13),
    "qos": ("QOS", 20),
    "defaultqos": ("Def QOS", 9),
    "parentname": ("Par Name", 10),
    "lineage": ("Lineage", 30),
}
ASSOC_DEFAULT = [
    "cluster",
    "account",
    "user",
    "partition",
    "share",
    "priority",
    "grpjobs",
    "grptres",
    "maxjobs",
    "maxtres",
    "qos",
    "defaultqos",
]


def _assoc_walk(s):
    """Associations in hierarchy order: [(depth, account, user|None)]."""
    out, seen = [], set()

    def walk(acct, depth):
        if acct in seen or acct not in s.accounts:
            return
        seen.add(acct)
        out.append((depth, acct, None))
        for un in _acct_users(s, acct):
            out.append((depth + 1, acct, un))
        for child in sorted(_acct_children(s, acct)):
            walk(child, depth + 1)

    walk("root", 0)
    for a in s.accounts:  # orphans (parent removed)
        if a not in seen:
            walk(a, 1)
    return out


def _lineage(s, acct, user=None):
    parts, a, guard = [], acct, 0
    while a and guard < 20:
        parts.insert(0, a)
        a = _acct_parent(s, a)
        guard += 1
    return "/" + "/".join(parts) + "/" + (f"0-{user}" if user else "")


def _show_assoc(state, fmt, tree, where, names, parsable, noheader):
    s = state.slurm
    keys = [k.split("%")[0].lower() for k in fmt.split(",")] if fmt else ASSOC_DEFAULT
    for k in keys:
        if k not in ASSOC_COLS:
            raise _err(f'sacctmgr: error: Invalid field requested: "{k}"')
    want_users = [u for u in where.get("user", ("", "="))[0].split(",") if u] + list(names)
    want_accts = [a for a in where.get("account", ("", "="))[0].split(",") if a]
    rows = []
    for depth, acct, user in _assoc_walk(s):
        if want_users and user not in want_users:
            continue
        if want_accts and acct not in want_accts:
            continue
        a = s.accounts[acct]
        u = s.users.get(user, {}) if user else {}
        if user:
            share = _assoc_shares(s, acct, user)
            qos = ",".join(u.get("qos", ["normal"]))
            dq = u.get("default_qos", "")
        else:
            share = a.get("fairshare", 1)
            qos, dq = "normal", ""
        vals = {
            "cluster": "superpod",
            "account": (" " * depth if tree else "") + acct,
            "user": user or "",
            "partition": "",
            "share": share,
            "fairshare": share,
            "priority": "",
            "grpjobs": "",
            "grptres": a.get("grp_tres", "") if not user else "",
            "maxjobs": "",
            "maxtres": "",
            "qos": qos,
            "defaultqos": dq,
            "parentname": "" if user or acct == "root" else (a.get("parent") or "root"),
            "lineage": _lineage(s, acct, user),
        }
        rows.append([vals[k] for k in keys])
    left = {i for i, k in enumerate(keys) if k == "account"} if tree else set()
    return _fmt_table([ASSOC_COLS[k] for k in keys], rows, parsable, noheader, left)


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
        if accts and acct not in accts and not (acct == "root" and user is None and depth == 0):
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


# ====================================================================== munge authentication


def _munge_key(state, host):
    """The key munged on ``host`` is running with. None = munge not modelled on this host."""
    node = state.nodes.get(host)
    if node is None or "munge" not in node.services:
        return None
    if node.services.get("munge") != "active":
        return "<munged down>"
    return node.get("munge_key_loaded", node.files.get(MUNGE_KEY))


def _munge_ok(state, host) -> bool:
    """Can slurmctld (on the head node) and ``host`` decode each other's credentials?"""
    if host == HEAD:
        return True
    hk, nk = _munge_key(state, HEAD), _munge_key(state, host)
    if hk is None or nk is None:
        return True
    return hk == nk and hk != "<munged down>"


def _key_tag(key) -> str:
    import hashlib

    return hashlib.sha256(str(key).encode()).hexdigest()[:12]


def _munge_log(state, host):
    """Journal evidence for a node whose credentials slurmctld cannot decode."""
    node = state.nodes.get(host)
    if node is None:
        return
    ts = _ts(state).replace("T", " ")
    pid = 4100 + (state.slurm.clock // TICK) % 800
    sd = node.journal.setdefault("slurmd", [])
    if node.services.get("munge") != "active":
        sd.append(
            f'{ts} {host} slurmd[{pid}]: slurmd: error: Munge encode failed: Failed to access "/run/munge/munge.socket.2": No such file or directory'
        )
        sd.append(f"{ts} {host} slurmd[{pid}]: slurmd: error: Unable to register: Protocol authentication error")
        return
    sd.append(f"{ts} {host} slurmd[{pid}]: slurmd: error: Munge decode failed: Invalid credential")
    sd.append(
        f"{ts} {host} slurmd[{pid}]: slurmd: error: slurm_unpack_received_msg: auth_g_verify: REQUEST_PING has authentication error: Invalid authentication credential"
    )
    sd.append(f"{ts} {host} slurmd[{pid}]: slurmd: error: Unable to register: Zero Bytes were transmitted or received")
    if HEAD in state.nodes:
        hj = state.nodes[HEAD].journal.setdefault("slurmctld", [])
        hj.append(f"{ts} {HEAD} slurmctld[2211]: error: Munge decode failed: Invalid credential")
        hj.append(
            f"{ts} {HEAD} slurmctld[2211]: error: slurm_unpack_received_msg: auth_g_verify: "
            f"MESSAGE_NODE_REGISTRATION_STATUS has authentication error: Invalid authentication credential"
        )
        hj.append(
            f"{ts} {HEAD} slurmctld[2211]: error: slurm_unpack_received_msg: [[{host}]:{40000 + pid}] "
            f"Protocol authentication error"
        )


@command("munge", help="munge -n — encode an empty credential (pipe into unmunge to test the key)")
def munge_cmd(ctx, args):
    import base64

    if "-V" in args or "--version" in args:
        return "munge-0.5.14 (2020-01-14)\n"
    key = _munge_key(ctx.state, ctx.host)
    if key is None:
        key = ctx.node.files.get(MUNGE_KEY, "default-munge-key")
    if key == "<munged down>":
        raise _err('munge: Error: Unable to access "/run/munge/munge.socket.2": No such file or directory', 6)
    blob = base64.b64encode(f"{_key_tag(key)}|{ctx.host}|{ctx.state.slurm.get('clock', 0)}".encode()).decode()
    return f"MUNGE:AwQFAAB{blob}:\n"


@command("unmunge", help="unmunge — decode a credential from stdin (munge -n | unmunge, ssh node munge -n | unmunge)")
def unmunge_cmd(ctx, args):
    import base64

    data = (ctx.stdin or "").strip()
    m = re.search(r"MUNGE:AwQFAAB([A-Za-z0-9+/=]+):", data)
    if not m:
        raise _err("unmunge: Error: No credential specified", 2)
    try:
        tag, src, _ = base64.b64decode(m.group(1)).decode().split("|", 2)
    except Exception:
        raise _err("unmunge: Error: Invalid credential format", 9) from None
    key = _munge_key(ctx.state, ctx.host)
    if key is None:
        key = ctx.node.files.get(MUNGE_KEY, "default-munge-key")
    if key == "<munged down>":
        raise _err('unmunge: Error: Unable to access "/run/munge/munge.socket.2": No such file or directory', 6)
    src_ip = ctx.state.nodes[src].ip if src in ctx.state.nodes else "unknown"
    if tag != _key_tag(key):
        raise _err("unmunge: Error: Invalid credential", 15)
    ts = _ts(ctx.state).replace("T", " ")
    return (
        "STATUS:          Success (0)\n"
        f"ENCODE_HOST:     {src} ({src_ip})\n"
        f"ENCODE_TIME:     {ts} +0000\n"
        f"DECODE_TIME:     {ts} +0000\n"
        "TTL:             300\n"
        "CIPHER:          aes128 (4)\n"
        "MAC:             sha256 (5)\n"
        "ZIP:             none (0)\n"
        "UID:             root (0)\n"
        "GID:             root (0)\n"
        "LENGTH:          0\n\n"
    )


@command("mungekey", "create-munge-key", help="mungekey --create [--force] — generate a NEW munge key (rotates it!)")
def mungekey_cmd(ctx, args):
    import hashlib

    force = any(a in ("-f", "--force") for a in args)
    if MUNGE_KEY in ctx.node.files and not force:
        raise _err(f'mungekey: Error: Failed to create "{MUNGE_KEY}": File exists', 1)
    seed = f"{ctx.host}-{ctx.state.slurm.get('clock', 0)}-{len(ctx.executed)}"
    ctx.node.files[MUNGE_KEY] = "munge-key:" + hashlib.sha256(seed.encode()).hexdigest() + "\n"
    return ""


# ====================================================================== CPU / NUMA affinity


def _core_numa(core: int) -> int:
    return 0 if core < CORES_PER_SOCKET else 1


def _gpu_numa(idx: int) -> int:
    """DGX H100: GPUs 0-3 hang off socket 0, GPUs 4-7 off socket 1 (see nvidia-smi topo -m)."""
    return 0 if int(idx) < 4 else 1


def _parse_int_list(spec) -> list[int]:
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _gpu_core_map(state, host):
    """GPU index -> Slurm core ids, as slurmd reports it from gres.conf (or NVML). Returns (map, error)."""
    node = state.nodes.get(host)
    if node is None or not node.gpus:
        return {}, None
    parsed = _parse_gres_conf(node.files.get(GRES_CONF, ""), host)
    ncores = int(node.get("cpus", 224)) // 2
    out, error = {}, None
    for ln in parsed["lines"]:
        if not ln.get("cores") or not ln["files"]:
            continue
        try:
            cores = _parse_int_list(ln["cores"])
        except ValueError:
            cores = [ncores]
        if any(c >= ncores or c < 0 for c in cores):
            error = f"Invalid GRES core specification ({ln['cores']}) on node {host}"
            continue
        for f in ln["files"]:
            m = re.match(r"/dev/nvidia(\d+)$", f)
            if m:
                out[int(m.group(1))] = cores
    if not out and parsed["autodetect"]:
        half = ncores // 2
        out = {
            g.index: list(range(0, half)) if _gpu_numa(g.index) == 0 else list(range(half, ncores)) for g in node.gpus
        }
    return out, error


def _affinity_ok(state, host, cmap) -> bool:
    node = state.nodes.get(host)
    if node is None or not node.gpus:
        return True
    for g in node.gpus:
        cores = cmap.get(g.index)
        if not cores or {_core_numa(c) for c in cores} != {_gpu_numa(g.index)}:
            return False
    return True


def _set_core_map(state, sn, log=False):
    cmap, error = _gpu_core_map(state, sn.name)
    sn.gpu_cores = {str(k): v for k, v in cmap.items()}
    sn.gpu_affinity_ok = _affinity_ok(state, sn.name, cmap)
    node = state.nodes.get(sn.name)
    if log and error and node is not None:
        ts = _ts(state).replace("T", " ")
        node.journal.setdefault("slurmd", []).append(f"{ts} {sn.name} slurmd[4100]: slurmd: error: {error}")


def _alloc_cores(state, j, host, gpu_idx) -> list[int]:
    sn = state.slurm.nodes[host]
    total = int(sn.get("cpus", 224)) // 2
    if j.exclusive:
        return list(range(total))
    used = set()
    for o in state.slurm.jobs:
        if o is not j and o.get("state") == "RUNNING" and host in (o.get("nodelist") or []):
            used.update((o.get("alloc_cores") or {}).get(host, []))
    free = [c for c in range(total) if c not in used]
    need = max(1, math.ceil(int(j.ntasks_per_node) * int(j.cpus_per_task) / 2))
    pref = set()
    for i in gpu_idx:
        pref.update((sn.get("gpu_cores") or {}).get(str(i), []))
    pick = [c for c in free if c in pref][:need]
    if len(pick) < need and "enforce-binding" not in str(j.get("gres_flags", "")):
        pick += [c for c in free if c not in pick][: need - len(pick)]
    return sorted(pick)


def _local_free_cores(state, j, host, gpu_idx) -> int:
    sn = state.slurm.nodes[host]
    used = set()
    for o in state.slurm.jobs:
        if o.get("state") == "RUNNING" and host in (o.get("nodelist") or []):
            used.update((o.get("alloc_cores") or {}).get(host, []))
    pref = set()
    for i in gpu_idx:
        pref.update((sn.get("gpu_cores") or {}).get(str(i), []))
    return len(pref - used)


def os_cpus(cores, ncores=112) -> list[int]:
    """Slurm core ids -> Linux CPU ids on a 2-socket Intel box (core c = CPUs c and c+ncores)."""
    return sorted([c for c in cores] + [c + ncores for c in cores])


def task_binding(state, host):
    """CPU ids the current job step task is bound to on ``host`` (None = not inside a job)."""
    node = state.nodes.get(host)
    jid = node.env.get("SLURM_JOB_ID") if node is not None else None
    j = _find_job(state, jid) if jid else None
    if j is None:
        return None
    total = int(state.slurm.nodes[host].get("cpus", 224)) // 2 if host in state.slurm.nodes else 112
    cores = list((j.get("alloc_cores") or {}).get(host, [])) or list(range(total))
    local, per = _TASK.get("local"), max(1, int(_TASK.get("per_node") or 1))
    bind = str(j.get("cpu_bind") or "")
    if local is not None and per > 1 and "none" not in bind and len(cores) >= per:
        size = len(cores) // per
        cores = cores[local * size : (local + 1) * size]
    return os_cpus(cores, total)


def _numa_local(j, host) -> bool | None:
    gpus = (j.get("alloc_gpus") or {}).get(host, [])
    cores = (j.get("alloc_cores") or {}).get(host, [])
    if not gpus or not cores or j.get("exclusive"):
        return None
    return {_core_numa(c) for c in cores} <= {_gpu_numa(i) for i in gpus}


# ====================================================================== fair share


def _acct_parent(s, name):
    a = s.accounts.get(name)
    return (a.get("parent") or "root") if a is not None and name != "root" else None


def _acct_children(s, name):
    return [a.name for a in s.accounts.values() if a.name != "root" and (a.get("parent") or "root") == name]


def _assoc_shares(s, acct, user) -> int:
    u = s.users.get(user)
    if u is None:
        return 1
    return int((u.get("shares") or {}).get(acct, 1))


def _acct_users(s, acct):
    return [un for un in s.accounts.get(acct, {}).get("users", []) if un in s.users]


def _job_usage(state, j) -> int:
    if j.get("start_time") is None:
        return 0
    end = j.end_time if j.get("end_time") is not None else state.slurm.get("clock", 0)
    ncpu = int(j.num_nodes) * int(j.ntasks_per_node) * int(j.cpus_per_task)
    return max(0, end - j.start_time) * ncpu


def _acct_usage(state, acct) -> int:
    s = state.slurm
    names = {acct}
    stack = [acct]
    while stack:
        for c in _acct_children(s, stack.pop()):
            if c not in names:
                names.add(c)
                stack.append(c)
    return sum(_job_usage(state, j) for j in s.jobs if j.get("account") in names) + sum(
        int(s.accounts[n].get("raw_usage", 0) or 0) for n in names if n in s.accounts
    )


def _user_usage(state, acct, user) -> int:
    s = state.slurm
    extra = int((s.users.get(user, {}).get("raw_usage") or {}).get(acct, 0)) if s.users.get(user) else 0
    return extra + sum(_job_usage(state, j) for j in s.jobs if j.get("account") == acct and j.get("user") == user)


def _level_total(s, parent) -> int:
    tot = sum(int(s.accounts[c].get("fairshare", 1)) for c in _acct_children(s, parent))
    tot += sum(_assoc_shares(s, parent, u) for u in _acct_users(s, parent))
    return max(1, tot)


def _norm_acct(s, name) -> float:
    if name == "root":
        return 1.0
    parent = _acct_parent(s, name)
    return int(s.accounts[name].get("fairshare", 1)) / _level_total(s, parent)


def _global_share(s, acct, user=None) -> float:
    share = 1.0
    if user is not None:
        share = _assoc_shares(s, acct, user) / _level_total(s, acct)
    a, seen = acct, set()
    while a and a != "root" and a not in seen:
        seen.add(a)
        share *= _norm_acct(s, a)
        a = _acct_parent(s, a)
    return share


def _fs_factor(state, user, acct) -> float:
    """Fair-share factor of a user's association (classic 2^(-usage/shares) formula)."""
    s = state.slurm
    u = s.users.get(user, {})
    if not acct or acct == "root" or acct not in s.accounts:
        return float(u.get("fairshare_factor", 0.5))
    total = max(1, _acct_usage(state, "root"))
    eff = _user_usage(state, acct, user) / total
    share = _global_share(s, acct, user)
    return round(2 ** (-eff / share), 6) if share > 0 else 0.0


# ====================================================================== su / runuser


def _known_user(state, name) -> bool:
    if name == "root":
        return True
    if name in state.slurm.users:
        return True
    bcm = state.get("bcm") or {}
    return name in (bcm.get("users") or {})


def _run_as(ctx, user, line, prog):
    from ..core.shell import Shell

    if not _known_user(ctx.state, user):
        raise _err(f"{prog}: user {user} does not exist or the user entry does not contain all the required fields", 1)
    sub = Shell(ctx.state, ctx.host)
    sub.ctx.executed = ctx.executed
    sub.ctx.host_stack = list(ctx.host_stack)
    sub.ctx.cwd = "/root" if user == "root" else f"/home/{user}"
    _USER.append(user)
    try:
        out = sub.run(line, record=False)
    finally:
        _USER.pop()
    return out + ("\n" if out and not out.endswith("\n") else "")


@command("su", help='su - <user> -c "<command>" — run one command as another user (e.g. to test sbatch as alice)')
def su_cmd(ctx, args):
    user, cmd, it = None, None, iter(args)
    for a in it:
        if a in ("-", "-l", "--login"):
            continue
        if a in ("-c", "--command"):
            cmd = next(it, None)
        elif a.startswith("--command="):
            cmd = a.split("=", 1)[1]
        elif a.startswith("-"):
            continue
        elif user is None:
            user = a
    user = user or "root"
    if cmd is None:
        raise _err(
            "su: interactive shells are not supported in the lab; run one command at a time:\n"
            f'  su - {user} -c "sbatch --wrap=hostname"',
            1,
        )
    return _run_as(ctx, user, cmd, "su")


@command("runuser", help="runuser -u <user> -- <command> — run a command as another user")
def runuser_cmd(ctx, args):
    import shlex as _shlex

    user, rest = None, list(args)
    if "-u" in rest:
        i = rest.index("-u")
        user = rest[i + 1] if i + 1 < len(rest) else None
        del rest[i : i + 2]
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not user or not rest:
        raise _err("runuser: usage: runuser -u <user> -- <command> [args...]", 1)
    return _run_as(ctx, user, " ".join(_shlex.quote(a) for a in rest), "runuser")


@command("whoami")
def whoami(ctx, args):
    return _USER[-1] + "\n"


# ====================================================================== hooks


def _service_hook(ctx, name, action):
    state = ctx.state
    if "slurm" not in state:
        return
    _ensure(state)
    host = ctx.host
    if name == "munge":
        node = ctx.node
        if action in ("start", "restart", "reload"):
            key = node.files.get(MUNGE_KEY)
            if key is None:
                ts = _ts(state).replace("T", " ")
                node.journal.setdefault("munge", []).append(
                    f'{ts} {host} munged[1187]: munged: Error: Failed to find keyfile "{MUNGE_KEY}": No such file or directory'
                )
                node.services["munge"] = "failed"
                return
            node["munge_key_loaded"] = key
            ts = _ts(state).replace("T", " ")
            node.journal.setdefault("munge", []).append(
                f"{ts} {host} systemd[1]: Started MUNGE authentication service."
            )
        _GUARD["depth"] += 1
        try:
            if host != HEAD and host in state.slurm.nodes and _munge_ok(state, host):
                if node.services.get("slurmd") == "active" and state.slurm.nodes[host].not_responding:
                    _register(state, host, log=False)
            _refresh(state)
        finally:
            _GUARD["depth"] -= 1
        return
    if name == "slurmd" and host in state.slurm.nodes:
        if action in ("start", "restart", "reload"):
            ok = _register(state, host)
            if ok is False and ctx.node.services.get("slurmd") == "failed":
                return
            _GUARD["depth"] += 1
            try:
                _refresh(state)
            finally:
                _GUARD["depth"] -= 1
        elif action == "stop":
            state.slurm.nodes[host].not_responding = True
            _derive_state(state, state.slurm.nodes[host])
    elif name == "slurmctld" and host == HEAD:
        if action in ("start", "restart", "reload"):
            state.slurm.controller_up = True
            apply_config(state)
            schedule(state)
        elif action == "stop":
            state.slurm.controller_up = False


def _reboot_hook(ctx, node):
    state = ctx.state
    if "slurm" not in state:
        return
    _ensure(state)
    name = node.hostname if "hostname" in node else ctx.host
    if name == HEAD:
        return
    sn = state.slurm.nodes.get(name)
    if sn is None:
        return
    for j in state.slurm.jobs:
        if j.state == "RUNNING" and name in j.nodelist:
            _finish(state, j, "NODE_FAIL", "0:0")
    sn.boot_time = _ts(state)
    expected = sn.get("_expected_reboot")
    if not expected and not sn.drain and not sn.down:
        sn.down = True
        sn.reason = f"Node unexpectedly rebooted boot_time={int((BASE_TIME - _dt.datetime(1970, 1, 1)).total_seconds()) + state.slurm.clock}"
        sn.reason_user = "slurm"
        sn.reason_time = _ts(state)
    node.services["slurmd"] = "active"
    if "munge" in node.services:
        if MUNGE_KEY in node.files:
            node.services["munge"] = "active"
            node["munge_key_loaded"] = node.files[MUNGE_KEY]
    _register(state, name, after_reboot=True)
    sn["_reregister"] = True  # confirm against the real post-boot GPU state on next use


HOOKS_SERVICE.append(_service_hook)
HOOKS_REBOOT.append(_reboot_hook)
