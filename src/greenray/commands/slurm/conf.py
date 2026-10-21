"""slurm.conf, gres.conf and plugstack.conf parsing, plus lookups into the running configuration."""

from __future__ import annotations

import fnmatch
import shlex

from .common import PLUGSTACK, _kv, compress_hostlist, expand_hostlist


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
