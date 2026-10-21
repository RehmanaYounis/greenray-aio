"""CPU/NUMA affinity: GPU-to-core maps from gres.conf and the core binding of job tasks."""

from __future__ import annotations

import math
import re

from .common import _TASK, CORES_PER_SOCKET, GRES_CONF, _find_job, _ts
from .conf import _parse_gres_conf


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
