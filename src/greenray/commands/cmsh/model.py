"""The BCM data model: categories, software images, provisioning and the node package lists."""

from __future__ import annotations

import copy
import fnmatch
import re

from ...core.errors import CommandError
from ...core.state import HEAD
from .. import HOOKS_REBOOT

NOW = "Mon Oct 19 10:15:13 2026"

# ---------------------------------------------------------------- package model

BASE_PKGS = {
    "linux-image-5.15.0-1042-nvidia": "5.15.0-1042.42",
    "openssh-server": "1:8.9p1-3ubuntu0.10",
    "rdma-core": "2307mlnx47-1.2310055",
    "cm-node-installer": "10.0-100124",
}
DGX_PKGS = {
    **BASE_PKGS,
    "nvidia-driver-550-server": "550.90.07-0ubuntu1",
    "nvidia-fabricmanager-550": "550.90.07-1",
    "datacenter-gpu-manager": "1:3.3.6",
    "nvidia-container-toolkit": "1.14.6-1",
    "nvidia-container-toolkit-base": "1.14.6-1",
    "mlnx-ofed-kernel-utils": "24.01.OFED.24.01.0.3.3.1-1",
    "slurm23.02-client": "23.02.7-100124",
}
# what the (simulated) apt repositories offer
REPO = {
    "nvidia-container-toolkit": ["1.14.6-1", "1.15.0-1", "1.16.2-1"],
    "nvidia-container-toolkit-base": ["1.14.6-1", "1.15.0-1", "1.16.2-1"],
    "linux-image-5.15.0-1053-nvidia": ["5.15.0-1053.59"],
    "linux-image-5.15.0-1042-nvidia": ["5.15.0-1042.42"],
    "datacenter-gpu-manager": ["1:3.3.5", "1:3.3.6", "1:3.3.7"],
    "nvidia-fabricmanager-550": ["550.54.15-1", "550.90.07-1"],
    "htop": ["3.0.5-7build2"],
    "iperf3": ["3.9-1"],
    "nvtop": ["1.2.2-1"],
}
INSTALL_MODES = ("AUTO", "FULL", "MAIN", "NOSYNC")

KIND_COLL = {
    "device": "devices",
    "category": "categories",
    "softwareimage": "softwareimages",
    "partition": "partitions",
    "user": "users",
    "group": "groups",
    "profile": "profiles",
}
KIND_TITLE = {
    "device": "Devices",
    "category": "Categories",
    "softwareimage": "SoftwareImages",
    "partition": "Partitions",
    "user": "Users",
    "group": "Groups",
    "profile": "Profiles",
}
MODES = ("device", "category", "softwareimage", "partition", "user", "group", "profile", "main")
EXCLUDE_LISTS = (
    "excludelistgrab",
    "excludelistgrabnew",
    "excludelistupdate",
    "excludelistsyncinstall",
    "excludelistfullinstall",
)
DEFAULT_EXCLUDE = (
    "# no-new-files: entries keep files that exist, but never add new ones\n"
    "- /proc/*\n- /sys/*\n- /dev/*\n- /run/*\n- /tmp/*\n- /var/tmp/*\n- /lost+found\n"
    "- /etc/hostname\n- /raid/*\n- /home/*\n- /cm/shared/*\n- /var/log/*\n- /var/spool/slurm/*\n"
    "- /var/lib/docker/*\n- /var/lib/containerd/*\n"
)
DEFAULT_PROFILES = {
    "admin": "Full access to everything in the cluster management daemon",
    "cmhealth": "Health check scripts (push health data)",
    "cmpam": "PAM module (cm-pam) access checks",
    "litenode": "Lite daemon on non-provisioned devices",
    "monitoringpush": "Push monitoring data",
    "node": "Compute node CMDaemon",
    "portal": "User portal: read jobs, workload and monitoring data",
    "power": "Power control only",
    "readonly": "Read-only access to all objects (no changes, no power, no exec)",
}
FSMOUNT_PROPS = ("mountpoint", "device", "filesystem", "mountoptions", "dump", "fsck")

# cmsh property name -> state field, per kind
PROPS = {
    "device": {
        "category": "category",
        "softwareimage": "softwareimage",
        "installmode": "installmode",
        "nextinstallmode": "nextinstallmode",
        "ip": "ip",
        "mac": "mac",
        "hostname": "hostname",
        "type": "type",
        "status": "status",
    },
    "category": {
        "name": "name",
        "softwareimage": "softwareimage",
        "installmode": "installmode",
        "disksetup": "disksetup",
        **{x: x for x in EXCLUDE_LISTS},
    },
    "softwareimage": {
        "name": "name",
        "path": "path",
        "kernelversion": "kernel",
        "kernelparameters": "kernelparameters",
        "revision": "version",
    },
    "partition": {"name": "name", "clustername": "clustername", "headnode": "headnode"},
    "user": {
        "name": "name",
        "id": "uid",
        "uid": "uid",
        "password": "password",
        "email": "email",
        "loginshell": "shell",
        "homedirectory": "home",
        "surname": "surname",
        "profile": "profile",
    },
    "group": {"name": "name", "id": "gid", "groupid": "gid", "members": "members"},
    "profile": {"name": "name", "description": "description"},
}
READONLY = {
    "device": {"type", "status", "hostname"},
    "softwareimage": {"name", "revision"},
    "category": {"name"},
    "partition": {"name", "headnode"},
    "user": {"name"},
    "group": {"name"},
    "profile": {"name"},
}


# ---------------------------------------------------------------- lazy state helpers


def bcm_init(state):
    """Fill in the BCM fields this module needs but state.make_bcm() does not create."""
    bcm = state.bcm
    if "partitions" not in bcm:
        bcm.partitions = {
            "base": {
                "name": "base",
                "clustername": "superpod",
                "headnode": HEAD,
                "bmcsettings": {"username": "bright", "password": "Bmc!Superpod24", "userid": 4},
            }
        }
    bcm.setdefault("bmc_actual_password", bcm.partitions.base.bmcsettings.password)
    bcm.setdefault("synclog", {})
    bcm.setdefault("dryruns", [])
    bcm.setdefault("events", [])
    for name, img in bcm.softwareimages.items():
        if "packages" not in img:
            img.packages = dict(DGX_PKGS if "dgx" in name else BASE_PKGS)
        img.setdefault("kernelparameters", "")
    from ..storage import DEFAULT_FSMOUNTS

    for cat in bcm.categories.values():
        cat.setdefault("installmode", "AUTO")
        cat.setdefault("disksetup", "default.xml")
        cat.setdefault("roles", [])
        cat.setdefault("bmcsettings", {})
        cat.setdefault("fsmounts", copy.deepcopy(DEFAULT_FSMOUNTS))
        for x in EXCLUDE_LISTS:
            cat.setdefault(x, DEFAULT_EXCLUDE)
    if "groups" not in bcm:
        bcm.groups = {"root": {"name": "root", "gid": 0, "members": []}}
        for u in bcm.users.values():
            if u.name != "root":
                bcm.groups.setdefault(u.name, {"name": u.name, "gid": u.get("uid", 1000), "members": []})
    if "profiles" not in bcm:
        bcm.profiles = {n: {"name": n, "description": d, "builtin": True} for n, d in DEFAULT_PROFILES.items()}
    bcm.setdefault("grab_dryruns", [])
    for name, dev in bcm.devices.items():
        dev.setdefault("fsmounts", [])
        dev.setdefault("roles", [])
        dev.setdefault("bmcsettings", {})
        dev.setdefault("nextinstallmode", None)
        if dev.type != "HeadNode":
            dev.setdefault("installmode", None)
            dev.setdefault("synced_image", effective_image(state, name))
            dev.setdefault("reboots", 0)
            dev.setdefault("provisioned_image_version", 1)
    return bcm


def effective_image(state, dev_name):
    dev = state.bcm.devices[dev_name]
    if dev.get("softwareimage"):
        return dev.softwareimage
    cat = state.bcm.categories.get(dev.get("category") or "")
    return cat.softwareimage if cat else None


def effective_roles(state, dev_name):
    dev = state.bcm.devices[dev_name]
    cat = state.bcm.categories.get(dev.get("category") or "") or {}
    return sorted(set(dev.get("roles", [])) | set(cat.get("roles", [])))


def effective_installmode(state, dev_name):
    dev = state.bcm.devices[dev_name]
    if dev.get("nextinstallmode"):
        return dev.nextinstallmode
    if dev.get("installmode"):
        return dev.installmode
    cat = state.bcm.categories.get(dev.get("category") or "") or {}
    return cat.get("installmode") or "AUTO"


def effective_bmc(state, dev_name):
    part = state.bcm.partitions.base.bmcsettings
    dev = state.bcm.devices[dev_name]
    cat = state.bcm.categories.get(dev.get("category") or "") or {}
    out = dict(part)
    for layer in (cat.get("bmcsettings") or {}, dev.get("bmcsettings") or {}):
        out.update({k: v for k, v in layer.items() if v not in (None, "")})
    return out


def effective_fsmounts(state, dev_name) -> list:
    """Category fsmounts overlaid with the device's own entries (same mountpoint wins)."""
    dev = state.bcm.devices[dev_name]
    cat = state.bcm.categories.get(dev.get("category") or "") or {}
    out = {e.get("mountpoint"): dict(e) for e in cat.get("fsmounts", [])}
    for e in dev.get("fsmounts", []) or []:
        out[e.get("mountpoint")] = dict(e)
    return list(out.values())


def exclude_patterns(text) -> list[str]:
    out = []
    for raw in str(text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("no-new-files:"):
            line = line[len("no-new-files:") :].strip()
        if line.startswith("- "):
            out.append(line[2:].strip())
        elif line.startswith("-"):
            out.append(line[1:].strip())
    return [p for p in out if p]


def is_excluded(path, patterns) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        base = pat[:-2] if pat.endswith("/*") else pat
        if path == base or path.startswith(base.rstrip("/") + "/"):
            return True
    return False


def image_files(state, img_name):
    """Files of a software image: {path_inside_image: content}."""
    img = state.bcm.softwareimages[img_name]
    prefix = img.path.rstrip("/") + "/"
    head = state.nodes[HEAD].files
    return {"/" + k[len(prefix) :]: v for k, v in head.items() if k.startswith(prefix)}


def node_packages(state, host):
    node = state.nodes[host]
    if "packages" not in node:
        dev = state.bcm.devices.get(host)
        img = dev.get("synced_image") if dev else None
        src = state.bcm.softwareimages.get(img or "")
        node.packages = dict(src.packages) if src and "packages" in src else dict(DGX_PKGS)
    return node.packages


def _has_driver(pkgs):
    return any(p.startswith("nvidia-driver-") for p in pkgs)


def _kernels(pkgs):
    return sorted(p[len("linux-image-") :] for p in pkgs if p.startswith("linux-image-"))


def sync_image(state, host, full=False):
    """Copy the node's effective software image onto it (files + packages)."""
    bcm = bcm_init(state)
    img_name = effective_image(state, host)
    img = bcm.softwareimages[img_name]
    node = state.nodes[host]
    node.packages = dict(img.packages)
    for path, content in image_files(state, img_name).items():
        node.files[path] = content
    if full:  # FULL install re-partitions local disks: anything on /raid is gone
        for p in [p for p in node.files if p.startswith("/raid/")]:
            del node.files[p]
    dev = bcm.devices[host]
    dev.synced_image = img_name
    dev.provisioned_image_version = dev.get("provisioned_image_version", 1) + 1
    return img_name


def provision(ctx, host, reason="reboot"):
    """Node (re)boots: PXE boot, node-installer syncs the image, roles are applied."""
    state = ctx.state
    bcm = bcm_init(state)
    dev = bcm.devices[host]
    node = state.nodes[host]
    mode = effective_installmode(state, host)
    node.power = "on"
    node.up = True
    for fn in HOOKS_REBOOT:
        fn(ctx, node)
    img_name = effective_image(state, host)
    if mode != "NOSYNC":
        sync_image(state, host, full=(mode == "FULL"))
    img = bcm.softwareimages[img_name if mode != "NOSYNC" else dev.synced_image]
    node.kernel = img.kernel
    if _has_driver(node_packages(state, host)):
        node.driver_loaded = True
        for svc in ("nvidia-persistenced", "nvidia-dcgm", "nvidia-fabricmanager"):
            if svc in node.services and node.services_enabled.get(svc, True):
                node.services[svc] = "active"
    else:
        node.driver_loaded = False
        node.modules = [m for m in node.modules if not m.startswith("nvidia")]
        for svc in ("nvidia-persistenced", "nvidia-dcgm", "nvidia-fabricmanager"):
            if svc in node.services:
                node.services[svc] = "inactive"
    roles = effective_roles(state, host)
    if "slurmclient" in roles:
        node.services["slurmd"] = "active"
        if host not in state.slurm.nodes:
            state.slurm.nodes[host] = {
                "name": host,
                "state": "IDLE",
                "reason": "",
                "gres": "gpu:h100:8",
                "gres_used": "gpu:h100:0",
                "cpus": node.cpus,
                "alloc_cpus": 0,
                "real_memory": 2063000,
                "features": "",
                "weight": 1,
            }
    if "mounts" in node:
        from ..storage import apply_fsmounts

        apply_fsmounts(state, host, effective_fsmounts(state, host))
    dev.nextinstallmode = None
    dev.status = "UP"
    dev.reboots = dev.get("reboots", 0) + 1
    bcm.events.append(f"{NOW} [notice] {host}: node provisioned ({reason}, install mode {mode}, image {img_name})")
    return mode, img_name


# ---------------------------------------------------------------- node lists


def expand_nodes(spec: str) -> list[str]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            a, b = part.split("..", 1)
            ma, mb = re.match(r"^(.*?)(\d+)$", a), re.match(r"^(.*?)(\d+)$", b)
            if not ma or not mb:
                raise CommandError(f"Invalid node range: {part}")
            width = len(ma.group(2))
            for i in range(int(ma.group(2)), int(mb.group(2)) + 1):
                out.append(f"{ma.group(1)}{i:0{width}d}")
            continue
        m = re.match(r"^(.*)\[(\d+)-(\d+)\](.*)$", part)
        if m:
            width = len(m.group(2))
            for i in range(int(m.group(2)), int(m.group(3)) + 1):
                out.append(f"{m.group(1)}{i:0{width}d}{m.group(4)}")
            continue
        out.append(part)
    return out


def _split_cmds(text: str) -> list[str]:
    """Split a cmsh command string on ';' and newlines, honouring quotes."""
    out, buf, quote = [], [], None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch in ";\n":
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf).strip())
    return [c for c in out if c and not c.startswith("#")]


def _dots(name, status, extra=""):
    return f"{name} {'.' * max(3, 26 - len(name))} [{status:^10}]" + (f" {extra}" if extra else "")
