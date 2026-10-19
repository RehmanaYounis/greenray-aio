"""Simulated cluster state.

The whole cluster is one tree of AttrDicts and lists, so scenarios can poke
faults into it with dotted paths (``nodes.dgx-01.gpus.3.xid``) and grade it
with plain Python expressions (``slurm.nodes['dgx-01'].state == 'IDLE'``).

Layout of the default cluster (a small DGX H100 SuperPOD slice managed by BCM):

    bcm-head      BCM head node, slurmctld, k8s control plane, NGC CLI
    dgx-01..04    DGX H100: 8x H100 SXM 80GB, 8x ConnectX-7 NDR IB, slurmd, kubelet
"""

from __future__ import annotations

import copy
import json


class AttrDict(dict):
    """dict with attribute access; nested dicts are converted on assignment."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = wrap(value)

    def __delattr__(self, key):
        del self[key]

    def __setitem__(self, key, value):
        super().__setitem__(key, wrap(value))

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def update(self, *args, **kw):
        for k, v in dict(*args, **kw).items():
            self[k] = v

    def __deepcopy__(self, memo):
        return wrap(copy.deepcopy(dict(self), memo))


class AttrList(list):
    """list that wraps dicts added to it, so ``gpu.processes.append({...})`` stays attribute-accessible."""

    def append(self, value):
        super().append(wrap(value))

    def extend(self, values):
        super().extend(wrap(v) for v in values)

    def insert(self, i, value):
        super().insert(i, wrap(value))

    def __setitem__(self, i, value):
        super().__setitem__(i, wrap(value))

    def __deepcopy__(self, memo):
        return wrap(copy.deepcopy(list(self), memo))


def wrap(value):
    if isinstance(value, (AttrDict, AttrList)):
        return value
    if isinstance(value, dict):
        return AttrDict({k: wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return AttrList(wrap(v) for v in value)
    return value


# ---------------------------------------------------------------- builders

COMPUTE_NODES = ["dgx-01", "dgx-02", "dgx-03", "dgx-04"]
HEAD = "bcm-head"
GPU_UUID_BASE = "GPU-7f3a{node:02d}{idx:02d}-1c2e-4b5a-9d0e-{node:02d}{idx:02d}a1b2c3d4"


def make_gpu(node_no: int, idx: int) -> dict:
    return {
        "index": idx,
        "name": "NVIDIA H100 80GB HBM3",
        "uuid": GPU_UUID_BASE.format(node=node_no, idx=idx),
        "bus_id": f"00000000:{[0x18, 0x2A, 0x3A, 0x5D, 0x9A, 0xAB, 0xBA, 0xDB][idx]:02X}:00.0",
        "serial": f"16523{node_no:02d}0{idx:02d}417",
        "vbios": "96.00.74.00.0C",
        "memory_total_mib": 81559,
        "memory_used_mib": 0,
        "utilization": 0,
        "temp_c": 32 + idx % 3,
        "mem_temp_c": 30 + idx % 4,
        "power_draw_w": 71.0 + idx,
        "power_limit_w": 700,
        "power_default_w": 700,
        "power_min_w": 200,
        "power_max_w": 700,
        "sm_clock_mhz": 345,
        "mem_clock_mhz": 2619,
        "max_sm_clock_mhz": 1980,
        "app_sm_clock_mhz": 1980,
        "pstate": "P0",
        "persistence_mode": True,
        "compute_mode": "Default",
        "ecc_mode": True,
        "ecc_pending": False,
        "ecc_volatile_sbe": 0,
        "ecc_volatile_dbe": 0,
        "ecc_aggregate_sbe": 0,
        "ecc_aggregate_dbe": 0,
        "retired_pages_pending": False,
        "remapped_rows_correctable": 0,
        "remapped_rows_uncorrectable": 0,
        "remap_pending": False,
        "remap_failure": False,
        # clocks_event_reasons / throttle reasons that are active, e.g. ["HW Thermal Slowdown"]
        "throttle_reasons": [],
        "mig_capable": True,
        "mig_mode": False,
        "mig_pending": None,  # None, or True/False waiting for reset
        "gpu_instances": [],  # [{"id": 1, "profile": "1g.10gb", "profile_id": 19, "placement": 0}]
        "compute_instances": [],  # [{"gi": 1, "id": 0, "profile": "1g.10gb"}]
        "xid": None,  # last XID error number seen on this GPU (None = healthy)
        "fallen_off_bus": False,  # XID 79: GPU not visible until node reboot/reset
        "reset_required": False,  # GPU must be reset (nvidia-smi -r) before it is usable
        "nvlink_down": [],  # list of NVLink ids that are inactive
        "nvlink_count": 18,
        "processes": [],  # [{"pid": 1234, "name": "python", "used_mib": 1000}]
        "healthy": True,
    }


def make_ib_port(idx: int) -> dict:
    return {
        "ca": f"mlx5_{idx}",
        "port": 1,
        "state": "Active",  # Active / Down / Initializing / Armed
        "phys_state": "LinkUp",  # LinkUp / Polling / Disabled
        "rate_gbps": 400,
        "expected_rate_gbps": 400,
        "link_layer": "InfiniBand",
        "lid": 0,  # filled in per node
        "sm_lid": 1,
        "fw": "28.39.1002",
        "symbol_errors": 0,
        "link_downed": 0,
        "rcv_errors": 0,
        "netdev": f"ibp{[24, 41, 58, 92, 155, 170, 187, 218][idx]}s0",
        "cable_ok": True,
    }


def make_node(name: str, node_no: int) -> dict:
    gpus = [make_gpu(node_no, i) for i in range(8)]
    ib = [make_ib_port(i) for i in range(8)]
    for i, p in enumerate(ib):
        p["lid"] = node_no * 10 + i + 1
    return {
        "hostname": name,
        "role": "compute",
        "model": "DGX H100",
        "os": "Ubuntu 22.04.4 LTS (DGX OS 6.2)",
        "kernel": "5.15.0-1042-nvidia",
        "ip": f"10.141.0.{node_no}",
        "bmc_ip": f"10.142.0.{node_no}",
        "power": "on",
        "up": True,
        "cpus": 224,
        "real_memory_mb": 2063000,
        "driver_version": "550.90.07",
        "cuda_version": "12.4",
        "driver_loaded": True,
        "gpus": gpus,
        "nvswitches": 4,
        "ib": ib,
        "services": {
            "nvidia-fabricmanager": "active",
            "nvidia-persistenced": "active",
            "nvidia-dcgm": "active",
            "slurmd": "active",
            "kubelet": "active",
            "containerd": "active",
            "docker": "active",
            "openibd": "active",
            "nvsm": "active",
            "chronyd": "active",
        },
        "services_enabled": {},
        "modules": [
            "nvidia",
            "nvidia_uvm",
            "nvidia_modeset",
            "nvidia_drm",
            "nvidia_peermem",
            "mlx5_core",
            "mlx5_ib",
            "ib_core",
            "ib_uverbs",
            "rdma_ucm",
        ],
        "dmesg": [
            "[    0.000000] Linux version 5.15.0-1042-nvidia (buildd@lcy02-amd64-031)",
            "[    4.118221] nvidia: loading out-of-tree module taints kernel.",
            "[    4.532110] NVRM: loading NVIDIA UNIX x86_64 Kernel Module  550.90.07",
            "[    5.010032] mlx5_core 0000:18:00.0: firmware version: 28.39.1002",
            "[    9.882134] nvidia-nvswitch: Probing for NVSwitch devices",
        ],
        "journal": {},  # service -> [log lines]
        "env": {},  # exported environment variables in this host's shell
        "docker_images": ["nvcr.io/nvidia/pytorch:24.05-py3"],
        "containers": [],  # [{"id","image","name","status","gpus"}]
        "enroot_images": [],  # squashfs files imported with enroot
        "pyxis_installed": True,
        "files": {
            "/etc/hostname": name + "\n",
            "/etc/slurm/gres.conf": "AutoDetect=nvml\nName=gpu Type=h100 File=/dev/nvidia[0-7]\n",
            "/etc/slurm/plugstack.conf": "required /usr/lib/x86_64-linux-gnu/slurm/spank_pyxis.so\n",
            "/etc/nvidia-container-runtime/config.toml": '[nvidia-container-cli]\nldconfig = "@/sbin/ldconfig.real"\n',
            "/etc/docker/daemon.json": json.dumps(
                {
                    "default-runtime": "nvidia",
                    "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}},
                },
                indent=4,
            )
            + "\n",
            "/etc/enroot/enroot.conf": "ENROOT_RUNTIME_PATH /run/enroot/user-$(id -u)\nENROOT_DATA_PATH /tmp/enroot-data/user-$(id -u)\n",
        },
    }


def make_head() -> dict:
    head = make_node(HEAD, 0)
    head.update(
        {
            "role": "head",
            "model": "x86 server",
            "ip": "10.141.255.254",
            "gpus": [],
            "nvswitches": 0,
            "ib": [make_ib_port(0)],
            "services": {
                "cmd": "active",  # BCM cluster management daemon
                "slurmctld": "active",
                "slurmdbd": "active",
                "mysql": "active",
                "kube-apiserver": "active",
                "kubelet": "active",
                "containerd": "active",
                "docker": "active",
                "opensm": "inactive",  # SM normally runs on the UFM / managed switch
                "chronyd": "active",
                "nfs-server": "active",
            },
            "modules": ["mlx5_core", "mlx5_ib", "ib_core"],
            "docker_images": [],
            "pyxis_installed": False,
        }
    )
    head["ib"][0]["lid"] = 1
    head["files"] = {
        "/etc/hostname": HEAD + "\n",
        "/etc/slurm/slurm.conf": SLURM_CONF,
        "/etc/slurm/gres.conf": "AutoDetect=nvml\nName=gpu Type=h100 File=/dev/nvidia[0-7]\n",
        "/etc/slurm/cgroup.conf": "ConstrainCores=yes\nConstrainDevices=yes\nConstrainRAMSpace=yes\n",
        "/root/.ngc/config": "",
        "/root/gpu-operator-values.yaml": "driver:\n  enabled: true\ntoolkit:\n  enabled: true\n",
    }
    return head


SLURM_CONF = """\
# slurm.conf - managed by Base Command Manager
ClusterName=superpod
SlurmctldHost=bcm-head
SlurmUser=slurm
AuthType=auth/munge
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
GresTypes=gpu
ProctrackType=proctrack/cgroup
TaskPlugin=task/affinity,task/cgroup
PriorityType=priority/multifactor
PriorityWeightFairshare=10000
PriorityWeightAge=1000
PriorityWeightQOS=5000
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageTRES=gres/gpu
JobAcctGatherType=jobacct_gather/cgroup
NodeName=dgx-[01-04] CPUs=224 Sockets=2 CoresPerSocket=56 ThreadsPerCore=2 RealMemory=2063000 Gres=gpu:h100:8 State=UNKNOWN
PartitionName=defq Nodes=dgx-[01-04] Default=YES MaxTime=INFINITE State=UP
"""


def make_slurm() -> dict:
    return {
        "controller_up": True,
        "nodes": {
            n: {
                "name": n,
                "state": "IDLE",  # IDLE / ALLOCATED / MIXED / DRAIN / DRAINING / DOWN
                "reason": "",
                "gres": "gpu:h100:8",
                "gres_used": "gpu:h100:0",
                "cpus": 224,
                "alloc_cpus": 0,
                "real_memory": 2063000,
                "features": "",
                "weight": 1,
            }
            for n in COMPUTE_NODES
        },
        "partitions": {
            "defq": {
                "name": "defq",
                "nodes": "dgx-[01-04]",
                "default": True,
                "state": "UP",
                "max_time": "INFINITE",
                "allow_qos": "ALL",
                "allow_accounts": "ALL",
                "priority_tier": 1,
            },
        },
        "jobs": [],  # see slurm command module for job fields
        "next_job_id": 1001,
        "qos": {"normal": {"name": "normal", "priority": 0, "max_tres_per_user": "", "preempt": ""}},
        "accounts": {"root": {"name": "root", "fairshare": 1, "users": ["root"]}},
        "users": {"root": {"name": "root", "default_account": "root"}},
        "reconfigure_needed": False,
    }


def make_k8s() -> dict:
    nodes = {
        HEAD: {
            "name": HEAD,
            "ready": True,
            "roles": "control-plane",
            "schedulable": True,
            "labels": {"kubernetes.io/hostname": HEAD, "node-role.kubernetes.io/control-plane": ""},
            "taints": [{"key": "node-role.kubernetes.io/control-plane", "effect": "NoSchedule", "value": ""}],
            "capacity": {"cpu": "64", "memory": "512Gi"},
            "version": "v1.29.4",
        },
    }
    for n in COMPUTE_NODES:
        nodes[n] = {
            "name": n,
            "ready": True,
            "roles": "<none>",
            "schedulable": True,
            "labels": {"kubernetes.io/hostname": n},
            "taints": [],
            "capacity": {"cpu": "224", "memory": "2015Gi"},  # nvidia.com/gpu appears once device plugin runs
            "version": "v1.29.4",
        }
    return {
        "nodes": nodes,
        "namespaces": ["default", "kube-system", "kube-flannel"],
        "pods": [],  # {"namespace","name","status","node","ready","restarts","requests",...}
        "deployments": [],
        "helm_repos": {},  # name -> url
        "helm_releases": [],  # {"name","namespace","chart","version","values":{}, "status"}
        "clusterpolicy": None,  # set when gpu-operator is installed
        "events": [],
    }


def make_bcm() -> dict:
    return {
        "version": "10.0",
        "devices": {
            HEAD: {
                "hostname": HEAD,
                "type": "HeadNode",
                "category": None,
                "status": "UP",
                "ip": "10.141.255.254",
                "mac": "52:54:00:aa:00:fe",
                "softwareimage": None,
                "roles": ["slurmserver", "boot"],
            },
            **{
                n: {
                    "hostname": n,
                    "type": "PhysicalNode",
                    "category": "dgx-h100",
                    "status": "UP",
                    "ip": f"10.141.0.{i + 1}",
                    "mac": f"7c:c2:55:1a:00:{i + 1:02x}",
                    "softwareimage": None,  # None = inherited from category
                    "roles": [],
                    "installmode": "AUTO",
                    "power": "on",
                    "provisioned_image_version": 1,
                }
                for i, n in enumerate(COMPUTE_NODES)
            },
        },
        "categories": {
            "default": {"name": "default", "softwareimage": "default-image", "roles": [], "nodes_hint": ""},
            "dgx-h100": {
                "name": "dgx-h100",
                "softwareimage": "dgx-os-6.2-h100-image",
                "roles": ["slurmclient"],
                "installmode": "AUTO",
                "disksetup": "dgx-h100.xml",
            },
        },
        "softwareimages": {
            "default-image": {
                "name": "default-image",
                "path": "/cm/images/default-image",
                "kernel": "5.15.0-1042-nvidia",
                "version": 1,
            },
            "dgx-os-6.2-h100-image": {
                "name": "dgx-os-6.2-h100-image",
                "path": "/cm/images/dgx-os-6.2-h100-image",
                "kernel": "5.15.0-1042-nvidia",
                "version": 1,
            },
        },
        "users": {"root": {"name": "root", "uid": 0, "groups": ["root"]}},
        "uncommitted": {},
        "image_updates": [],  # nodes that ran imageupdate
    }


def make_cluster() -> AttrDict:
    nodes = {HEAD: make_head()}
    for i, n in enumerate(COMPUTE_NODES, start=1):
        nodes[n] = make_node(n, i)
    return wrap(
        {
            "nodes": nodes,
            "slurm": make_slurm(),
            "k8s": make_k8s(),
            "bcm": make_bcm(),
            "ngc": {
                "configured": False,
                "api_key": "",
                "org": "",
                "team": "",
                "registry_logged_in": {},  # host -> bool (docker login nvcr.io)
                "catalog": {
                    "nvidia/pytorch": ["24.05-py3", "24.04-py3", "23.12-py3"],
                    "nvidia/tensorflow": ["24.05-tf2-py3"],
                    "nvidia/tritonserver": ["24.05-py3"],
                    "nvidia/nemo": ["24.05"],
                    "nvidia/cuda": ["12.4.1-base-ubuntu22.04", "12.4.1-devel-ubuntu22.04"],
                    "nvidia/k8s/dcgm-exporter": ["3.3.6-3.4.2-ubuntu22.04"],
                },
            },
            "dcgm": {
                "groups": {},  # id -> {"name","gpus":[...]}
                "health_watches": {},  # group id -> "pmi..." systems watched
                "fieldgroups": {},
                "stats_enabled": False,
                "diag_results": {},  # "host" -> {"level": int, "result": "Pass"/"Fail", "details": [...]}
            },
            "flags": {},  # free-form markers scenarios/commands may set
        }
    )


def clone(state: AttrDict) -> AttrDict:
    return wrap(copy.deepcopy(state))


# ---------------------------------------------------------------- path helpers


def _step(obj, key):
    if isinstance(obj, list):
        return obj[int(key)]
    return obj[key]


def get_path(state, path: str):
    obj = state
    for key in path.split("."):
        obj = _step(obj, key)
    return obj


def set_path(state, path: str, value):
    keys = path.split(".")
    obj = state
    for key in keys[:-1]:
        obj = _step(obj, key)
    last = keys[-1]
    if isinstance(obj, list):
        obj[int(last)] = wrap(value)
    else:
        obj[last] = wrap(value)


def append_path(state, path: str, value):
    get_path(state, path).append(wrap(value))
