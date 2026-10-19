"""Storage-path diagnostics for training input pipelines: gdscheck, iostat, nfsiostat, dataloader-bench.

``dataloader-bench`` is a LAB HELPER, not an NVIDIA tool: it stands in for "run the job's input
pipeline on its own and measure samples/s", which on a real cluster you would do with the
framework's own benchmark mode (e.g. a DALI pipeline or a PyTorch DataLoader loop without the model).

State read (all optional, read defensively):
  node.mounts    list of {"source", "target", "fstype", "options"} (options: comma-separated
                 string, as in /proc/mounts). A dict keyed by mount target is accepted too.
                 Missing -> the dataset is assumed to be on the local root filesystem.
  node.training  {"name", "user", "config": "/path/job.env", "gpus": 8, "gpu_rate": samples/s per
                 GPU when never starved, "sample_kb": avg sample size, "launched": {KEY: value}}
                 ``launched`` is the job.env the *running* job was started with; nvidia-smi dmon
                 and iostat/nfsiostat show that job. dataloader-bench reads the current file.
  node.modules   nvidia_fs (GPUDirect Storage kernel driver)
  /etc/cufile.json  "allow_compat_mode": true|false
"""

from __future__ import annotations

import re

from ..core.errors import CommandError
from . import command

GDS_VERSION = "1.10.0.4"
NVFS_VERSION = "2.20"
CUFILE_VERSION = "2.12"

PER_WORKER = {True: 400.0, False: 250.0}  # samples/s per worker: GDS path vs POSIX/compat (bounce buffers)
USABLE_WORKERS_PER_GPU = 26  # 224 CPUs / 8 GPUs, minus the training processes themselves
LOCAL_NVME = {"/raid": 24.0}  # GB/s for local NVMe RAID0 mounts


def _norm(m):
    return m.strip().replace("-", "_")


def nvfs_loaded(node) -> bool:
    return any(_norm(m) == "nvidia_fs" for m in node.get("modules", []))


def ensure_mounts(state, host):
    """Let storage.py fill in a node's default mount table (if that module provides one)."""
    try:
        from . import storage

        storage.mounts(state, host)
    except (ImportError, AttributeError, KeyError):
        pass


def mounts(node) -> list[dict]:
    raw = node.get("mounts") or []
    if isinstance(raw, dict):
        raw = [dict(v, target=v.get("target", k)) for k, v in raw.items()]
    out = []
    for m in raw:
        try:
            out.append(
                {
                    "source": str(m.get("source", m.get("device", ""))),
                    "target": str(m.get("target", m.get("mountpoint", ""))),
                    "fstype": str(m.get("fstype", m.get("type", ""))),
                    "options": m.get("options", ""),
                }
            )
        except AttributeError:
            continue
    for m in out:
        if isinstance(m["options"], (list, tuple)):
            m["options"] = ",".join(m["options"])
    return out


def mount_for(node, path):
    best = None
    for m in mounts(node):
        t = m["target"].rstrip("/") or "/"
        if path == t or path.startswith(t.rstrip("/") + "/") or t == "/":
            if best is None or len(t) > len(best["target"]):
                best = m
    return best


def _opt(options, key, default=None):
    for o in str(options).split(","):
        k, _, v = o.partition("=")
        if k.strip() == key:
            return v.strip() if v else True
    return default


def storage_gbps(m) -> float:
    if m is None:
        return 2.0  # OS drive
    fs = m["fstype"]
    if fs.startswith("nfs"):
        rsize = int(_opt(m["options"], "rsize", 1048576) or 1048576)
        nconnect = int(_opt(m["options"], "nconnect", 1) or 1)
        if _opt(m["options"], "proto") == "rdma":
            return min(10.0, 1.4 * (rsize / 8192) ** 0.28 * nconnect**0.5)
        # NFS over TCP from the head node: capped by its 10/25GbE internalnet link
        return min(1.15, 0.30 * (rsize / 8192) ** 0.28 * nconnect**0.5)
    if fs == "lustre":
        return 22.0
    if fs in ("wekafs", "gpfs", "beegfs"):
        return 20.0
    if fs in ("ext4", "xfs"):
        return LOCAL_NVME.get(m["target"], 2.0)
    return 2.0


def gds_fs_supported(m) -> tuple[bool, str]:
    if m is None:
        return False, "root filesystem on the OS drive"
    fs = m["fstype"]
    if fs == "lustre":
        return True, "DDN EXAScaler (Lustre)"
    if fs.startswith("nfs"):
        if _opt(m["options"], "proto") == "rdma":
            return True, "NFS over RDMA"
        return False, "NFS over TCP is not supported by GDS (needs NFS over RDMA)"
    if fs in ("ext4", "xfs") and m["target"] in LOCAL_NVME:
        return True, f"{fs} on local NVMe"
    if fs in ("wekafs", "gpfs", "beegfs"):
        return True, fs
    return False, f"{fs} is not a GDS-supported filesystem"


def compat_allowed(node) -> bool:
    m = re.search(r'"allow_compat_mode"\s*:\s*(true|false)', node.files.get("/etc/cufile.json", ""))
    return m is None or m.group(1) == "true"


def parse_env(text: str) -> dict:
    out = {}
    for line in str(text).splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def pipeline(node, cfg: dict) -> dict:
    """Model the input pipeline for the job config ``cfg`` on ``node``."""
    tr = node.get("training") or {}
    gpus = int(tr.get("gpus", 8))
    demand = gpus * float(tr.get("gpu_rate", 3000))
    sample_kb = float(tr.get("sample_kb", 150))
    data = cfg.get("DATA_DIR", "")
    try:
        workers = max(0, int(cfg.get("NUM_WORKERS", "0") or 0))
    except ValueError:
        workers = 0
    m = mount_for(node, data)
    gbps = storage_gbps(m)
    want_gds = str(cfg.get("USE_GDS", "0")).lower() in ("1", "true", "yes")
    fs_ok, fs_why = gds_fs_supported(m)
    if not want_gds:
        gds, why = False, "USE_GDS=0 (POSIX reads)"
    elif not nvfs_loaded(node):
        gds, why = False, "nvidia_fs not loaded -> cuFile compatibility mode (POSIX through bounce buffers)"
    elif not fs_ok:
        gds, why = False, f"{fs_why} -> cuFile compatibility mode"
    else:
        gds, why = True, f"GPUDirect Storage ({fs_why})"
    error = ""
    exists = any(f.startswith(data.rstrip("/") + "/") for f in node.files) if data else False
    if not exists:
        error = f"DATA_DIR {data or '(unset)'}: No such file or directory"
    elif want_gds and not gds and not compat_allowed(node):
        error = "cuFileHandleRegister failed: GPUDirect Storage not supported on current file (compat mode disabled in /etc/cufile.json)"
    eff_workers = min(workers, USABLE_WORKERS_PER_GPU)
    storage_rate = gbps * 1e6 / sample_kb
    cpu_rate = gpus * eff_workers * PER_WORKER[gds]
    rate = 0.0 if error else min(storage_rate, cpu_rate, demand)
    if error:
        bound = "error"
    elif rate >= demand * 0.999:
        bound = "gpu"
    elif storage_rate <= cpu_rate:
        bound = "storage"
    else:
        bound = "cpu"
    return {
        "data_dir": data,
        "mount": m,
        "storage_gbps": gbps,
        "gds": gds,
        "gds_reason": why,
        "workers": workers,
        "eff_workers": eff_workers,
        "gpus": gpus,
        "storage_rate": storage_rate,
        "cpu_rate": cpu_rate,
        "rate": rate,
        "demand": demand,
        "ratio": (rate / demand) if demand else 0.0,
        "bound": bound,
        "error": error,
        "sample_kb": sample_kb,
    }


def running_job(node):
    """Pipeline model of the job currently running on ``node`` (or None)."""
    tr = node.get("training")
    if not tr or not tr.get("launched"):
        return None
    return pipeline(node, dict(tr.launched))


def util_samples(ratio: float, n: int) -> list[int]:
    """GPU SM% per 1 s sample for a job that is busy ``ratio`` of the time: bursts, then starves."""
    period = 10
    busy = max(1, round(ratio * period))
    out = []
    for i in range(n):
        if ratio >= 0.97:
            out.append(96 + i % 3)
        elif i % period >= period - busy:
            out.append(97 - (i % 3))
        else:
            out.append([0, 3, 1, 6, 0, 2][i % 6])
    return out


# ---------------------------------------------------------------- gdscheck


def _cuda_int(ver: str) -> int:
    major, _, minor = str(ver).partition(".")
    return int(major or 0) * 1000 + int((minor or "0").split(".")[0]) * 10


DRIVER_ROWS = [
    ("NVMe", "nvme"),
    ("NVMeOF", None),
    ("SCSI", None),
    ("ScaleFlux CSD", None),
    ("NVMesh", None),
    ("DDN EXAScaler", "lustre"),
    ("IBM Spectrum Scale", "gpfs"),
    ("NFS", "nfs"),
    ("BeeGFS", "beegfs"),
    ("WekaFS", "wekafs"),
]


@command(
    "gdscheck",
    "gdscheck.py",
    "/usr/local/cuda/gds/tools/gdscheck",
    "/usr/local/cuda/gds/tools/gdscheck.py",
    help="gdscheck -p — GPUDirect Storage platform check (nvidia_fs driver, supported filesystems, cufile.json)",
)
def gdscheck(ctx, args):
    ensure_mounts(ctx.state, ctx.host)
    if "-p" not in args:
        return (
            "usage: gdscheck [-p] [-v] [-V]\n  -p  gds platform check\n  -v  gds version checks\n  -V  gds fs checks\n"
        )
    node = ctx.node
    if not node.gpus:
        raise CommandError("gdscheck: no CUDA-capable device is detected", 1)
    loaded = nvfs_loaded(node)
    present = {m["fstype"].split(".")[0] if not m["fstype"].startswith("nfs") else "nfs": m for m in mounts(node)}
    out = [
        f" GDS release version: {GDS_VERSION}",
        f" nvidia_fs version:  {NVFS_VERSION if loaded else 'N/A (nvidia_fs.ko not loaded)'} libcufile version: {CUFILE_VERSION}",
        " Platform: x86_64",
        " ============",
        " ENVIRONMENT:",
        " ============",
        " =====================",
        " DRIVER CONFIGURATION:",
        " =====================",
    ]
    for label, key in DRIVER_ROWS:
        sup = False
        if loaded and key == "nvme":
            sup = True
        elif loaded and key in present:
            sup = gds_fs_supported(present[key])[0]
        out.append(f" {label:<19}: {'Supported' if sup else 'Unsupported'}")
    out += [
        " Userspace RDMA     : Unsupported",
        f" --Mellanox PeerDirect : {'Enabled' if any(_norm(m) == 'nvidia_peermem' for m in node.modules) else 'Disabled'}",
        " --rdma library        : Not Loaded (libcufile_rdma.so)",
        " --rdma devices        : Not configured",
        " --rdma_device_status  : Up: 0 Down: 0",
        " =====================",
        " CUFILE CONFIGURATION:",
        " =====================",
        f" properties.use_compat_mode : {'true' if compat_allowed(node) else 'false'}",
        " properties.force_compat_mode : false",
        " properties.gds_rdma_write_support : true",
        " properties.use_poll_mode : false",
        " properties.max_direct_io_size_kb : 16384",
        " properties.max_device_cache_size_kb : 131072",
        " properties.max_device_pinned_mem_size_kb : 33554432",
        " fs.lustre.posix_gds_min_kb: 0",
        " execution.max_io_threads : 4",
        " execution.parallel_io : true",
        " =========",
        " GPU INFO:",
        " =========",
    ]
    for g in node.gpus:
        out.append(
            f" GPU index {g.index} {g.name} bar:1 bar size (MiB):131072 "
            f"{'supports GDS' if loaded else 'GDS not available (nvidia_fs not loaded)'}, IOMMU State: Disabled"
        )
    out += [
        " ==============",
        " PLATFORM INFO:",
        " ==============",
        " IOMMU: disabled",
        " Nvidia Driver Info Status: Supported(Nvidia Open Driver Installed)",
        f" Cuda Driver Version Installed:  {_cuda_int(node.cuda_version)}",
        f" Platform: {node.model.replace(' ', '')}, Arch: x86_64(Linux {node.kernel})",
    ]
    if loaded:
        out.append(" Platform verification succeeded")
    else:
        out.append(" Platform verification error :")
        out.append(" nvidia_fs driver is not loaded: GDS will run in compatibility mode")
    ctx.state.flags.setdefault("gdscheck", {})[ctx.host] = {"nvidia_fs": loaded}
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- iostat / nfsiostat


def _running_reads(node):
    """(mount, kB/s) the running job is reading at, or (None, 0)."""
    p = running_job(node)
    if p is None or p["error"]:
        return None, 0.0
    return p["mount"], p["rate"] * p["sample_kb"]


@command("iostat", help="iostat -x [interval [count]] — block device utilisation (local disks only)")
def iostat(ctx, args):
    ensure_mounts(ctx.state, ctx.host)
    node = ctx.node
    m, kbs = _running_reads(node)
    p = running_job(node)
    local = m is not None and m["fstype"] in ("ext4", "xfs") and m["target"] in LOCAL_NVME
    busy_cpus = 0.0
    iowait = 0.3
    if p:
        busy_cpus = p["gpus"] * p["eff_workers"] * (p["ratio"] if p["bound"] != "cpu" else 1.0) + p["gpus"]
        if p["bound"] == "storage":
            iowait = 6.7 if not local else 2.1
    user = min(95.0, busy_cpus / node.cpus * 100 * 0.8)
    sysp = user * 0.35
    idle = max(0.0, 100 - user - sysp - iowait)
    out = [
        f"Linux {node.kernel} ({ctx.host}) \t10/19/2026 \t_x86_64_\t({node.cpus} CPU)",
        "",
        "avg-cpu:  %user   %nice %system %iowait  %steal   %idle",
        f"          {user:6.2f}    0.00  {sysp:6.2f}  {iowait:6.2f}    0.00  {idle:6.2f}",
        "",
    ]
    if "-c" in args and "-x" not in args and "-d" not in args:
        return "\n".join(out) + "\n"
    hdr = (
        "Device            r/s     rkB/s   rrqm/s  %rrqm r_await rareq-sz     w/s     wkB/s   wrqm/s  %wrqm "
        "w_await wareq-sz  aqu-sz  %util"
    )
    out.append(hdr)
    devs = ["md1", "nvme0n1"] + [f"nvme{i}n1" for i in range(1, 9)]
    for d in devs:
        r = rk = aw = util = 0.0
        rq = 0.0
        if d == "nvme0n1":
            w, wk = 3.1, 41.6
        else:
            w, wk = 0.0, 0.0
        if local and d != "nvme0n1":
            share = 1.0 if d == "md1" else 1 / 8
            rk = kbs * share
            rq = 128.0
            r = rk / rq
            aw = 0.21
            util = min(100.0, rk / (3.2e6 if d != "md1" else 24e6) * 100)
        out.append(
            f"{d:<12} {r:8.2f} {rk:9.2f} {0:8.2f} {0:6.2f} {aw:7.2f} {rq:8.2f} {w:7.2f} {wk:9.2f} "
            f"{0:8.2f} {0:6.2f} {0.12 if w else 0:7.2f} {13.4 if w else 0:8.2f} {r * aw / 1000:7.2f} {util:6.2f}"
        )
    return "\n".join(out) + "\n"


@command("nfsiostat", help="nfsiostat [mountpoint] — per-NFS-mount read/write rates, op size and latency")
def nfsiostat(ctx, args):
    ensure_mounts(ctx.state, ctx.host)
    node = ctx.node
    want = [a for a in args if a.startswith("/")]
    nfs = [m for m in mounts(node) if m["fstype"].startswith("nfs")]
    if want:
        nfs = [m for m in nfs if m["target"] in want]
    if not nfs:
        return "No NFS mount points were found\n"
    rm, kbs = _running_reads(node)
    out = []
    for m in nfs:
        reading = rm is not None and rm["target"] == m["target"]
        rsize_kb = int(_opt(m["options"], "rsize", 1048576) or 1048576) / 1024
        rk = kbs if reading else 0.0
        rops = rk / rsize_kb if rsize_kb else 0.0
        rtt = 0.9 if reading else 0.0
        q = 25.9 if reading and storage_gbps(m) < 1.0 else (3.1 if reading else 0.0)
        out += [
            "",
            f"{m['source']} mounted on {m['target']}:",
            "",
            "           ops/s       rpc bklog",
            f"{rops + 4.2:16.3f}{(12.8 if q > 10 else 0.0):16.3f}",
            "",
            "read:              ops/s            kB/s           kB/op         retrans    avg RTT (ms)    avg exe (ms)  avg queue (ms)            errs",
            f"{rops:24.3f}{rk:16.3f}{(rsize_kb if reading else 0):16.3f}        0 (0.0%){rtt:16.3f}{rtt + q:16.3f}{q:16.3f}        0 (0.0%)",
            "write:             ops/s            kB/s           kB/op         retrans    avg RTT (ms)    avg exe (ms)  avg queue (ms)            errs",
            f"{0.4:24.3f}{1.6:16.3f}{4.0:16.3f}        0 (0.0%){0.7:16.3f}{0.8:16.3f}{0.1:16.3f}        0 (0.0%)",
        ]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- dataloader-bench (lab helper)


@command(
    "dataloader-bench",
    help="LAB HELPER: dataloader-bench [--config job.env] [--data DIR] [--workers N] [--gds 0|1] — input pipeline samples/s",
)
def dataloader_bench(ctx, args):
    ensure_mounts(ctx.state, ctx.host)
    node = ctx.node
    tr = node.get("training")
    if not tr:
        raise CommandError("dataloader-bench: no training job is configured on this node", 1)
    conf_path, overrides = tr.get("config"), {}
    it = iter(args)
    for a in it:
        if a in ("-c", "--config"):
            conf_path = next(it, conf_path)
        elif a in ("-d", "--data"):
            overrides["DATA_DIR"] = next(it, "")
        elif a in ("-w", "--workers"):
            overrides["NUM_WORKERS"] = next(it, "0")
        elif a == "--gds":
            overrides["USE_GDS"] = next(it, "1")
        elif a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            key = {"config": None, "data": "DATA_DIR", "workers": "NUM_WORKERS", "gds": "USE_GDS"}.get(k)
            if k == "config":
                conf_path = v
            elif key:
                overrides[key] = v
    if conf_path not in node.files:
        raise CommandError(f"dataloader-bench: {conf_path}: No such file or directory", 1)
    cfg = parse_env(node.files[conf_path])
    cfg.update(overrides)
    p = pipeline(node, cfg)
    m = p["mount"]
    where = (
        f"{m['fstype']}, {m['source']}"
        + (
            f", rsize={_opt(m['options'], 'rsize')}"
            if m["fstype"].startswith("nfs") and _opt(m["options"], "rsize")
            else ""
        )
        if m
        else "local root filesystem"
    )
    out = [
        "[lab helper] dataloader-bench: synthetic stand-in for the job's input pipeline (not an NVIDIA tool)",
        f"config        : {conf_path}{'  (+ command-line overrides)' if overrides else ''}",
        f"DATA_DIR      : {p['data_dir']}  ({where})",
        f"workers       : {p['workers']} per GPU x {p['gpus']} GPUs"
        + (f"  (only {p['eff_workers']} per GPU can run: CPU cores)" if p["eff_workers"] < p["workers"] else ""),
        f"reader        : {p['gds_reason']}",
    ]
    if p["error"]:
        out.append(f"ERROR         : {p['error']}")
        ctx.state.flags.setdefault("dataloader_bench", {})[ctx.host] = {
            "config": conf_path,
            "from_config": not overrides,
            "ok": False,
            "samples_per_s": 0.0,
            "demand": p["demand"],
            "bound": "error",
        }
        raise CommandError("\n".join(out), 1)
    starve = max(0.0, 1 - p["ratio"]) * 100
    out += [
        f"storage limit : {p['storage_gbps']:.2f} GB/s = {p['storage_rate']:,.0f} samples/s ({p['sample_kb']:.0f} kB/sample)",
        f"worker limit  : {p['cpu_rate']:,.0f} samples/s",
        f"pipeline      : {p['rate']:,.0f} samples/s   GPU demand: {p['demand']:,.0f} samples/s "
        f"({p['gpus']} x {p['demand'] / p['gpus']:,.0f})",
        {
            "gpu": "verdict       : NOT input-bound - the GPUs are the bottleneck (as they should be)",
            "storage": f"verdict       : INPUT-BOUND on storage reads - GPUs wait for data ~{starve:.0f}% of the time",
            "cpu": f"verdict       : INPUT-BOUND on dataloader workers - GPUs wait for data ~{starve:.0f}% of the time",
        }[p["bound"]],
    ]
    ctx.state.flags.setdefault("dataloader_bench", {})[ctx.host] = {
        "config": conf_path,
        "from_config": not overrides,
        "ok": True,
        "samples_per_s": round(p["rate"], 1),
        "demand": p["demand"],
        "bound": p["bound"],
        "gds": p["gds"],
        "data_dir": p["data_dir"],
    }
    return "\n".join(out) + "\n"
