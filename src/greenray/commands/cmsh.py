"""NVIDIA Base Command Manager (BCM) cluster management shell: ``cmsh``.

Only the one-shot forms are supported because the lab shell is line based:

    cmsh -c "device; use dgx-01; set category dgx-h100; commit"
    cmsh -f /root/commands.cmsh

Modes: device, category, softwareimage, partition, user, group, profile, main.
Submodes: roles, bmcsettings, fsmounts (category/device).
Changes are *staged* in ``state.bcm.uncommitted`` until ``commit``, exactly as
in real cmsh (forgetting ``commit`` is the classic mistake). Staged changes are
kept for the whole lab session so that a later ``cmsh -c "device; commit"``
still finds them.

Software images live on the head node as files under ``/cm/images/<name>/``
(``bcm-head`` file entries with that prefix). ``imageupdate`` and
(re)provisioning copy those files and the image's package list to the node.

Also here: ``chroot`` / ``cm-chroot-sw-img`` into a software image and ``cmha``.
"""

from __future__ import annotations

import copy
import fnmatch
import re
import shlex

from ..core import audit
from ..core.errors import CommandError
from ..core.state import HEAD, wrap
from . import HOOKS_REBOOT, command

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
    from .storage import DEFAULT_FSMOUNTS

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
        from .storage import apply_fsmounts

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


def _table(headers, rows, widths=None):
    widths = widths or [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)
    ]
    widths = [max(w, len(h)) for w, h in zip(widths, headers)]
    line = " ".join(f"{h:<{w}}" for h, w in zip(headers, widths)).rstrip()
    sep = " ".join("-" * w for w in widths)
    body = [" ".join(f"{str(c):<{w}}" for c, w in zip(r, widths)).rstrip() for r in rows]
    return "\n".join([line, sep, *body])


def _dots(name, status, extra=""):
    return f"{name} {'.' * max(3, 26 - len(name))} [{status:^10}]" + (f" {extra}" if extra else "")


class CmshError(Exception):
    pass


# ---------------------------------------------------------------- the shell


class Cmsh:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = ctx.state
        self.bcm = bcm_init(ctx.state)
        self.mode = None
        self.obj = None
        self.sub = None  # "roles" | "bmcsettings" | "fsmounts"
        self.fsobj = None  # current mountpoint inside the fsmounts submode
        self.out: list[str] = []
        self.errors = 0

    # ------------------------------------------------ staging
    def _key(self, kind, name):
        return f"{kind}:{name}"

    def coll(self, kind):
        return self.bcm[KIND_COLL[kind]]

    def exists(self, kind, name):
        st = self.bcm.uncommitted.get(self._key(kind, name))
        if st and st.op == "remove":
            return False
        return name in self.coll(kind) or bool(st and st.op == "add")

    def view(self, kind, name) -> dict:
        st = self.bcm.uncommitted.get(self._key(kind, name))
        base = copy.deepcopy(dict(self.coll(kind).get(name, {})))
        if st:
            base.update(copy.deepcopy(dict(st.fields)))
        return wrap(base)

    def stage(self, kind, name, field, value):
        key = self._key(kind, name)
        st = self.bcm.uncommitted.get(key)
        if st is None:
            self.bcm.uncommitted[key] = {"kind": kind, "name": name, "op": "modify", "fields": {}}
            st = self.bcm.uncommitted[key]
        st.fields[field] = value

    def modified(self, kind, name):
        return self._key(kind, name) in self.bcm.uncommitted

    # ------------------------------------------------ output
    def p(self, text=""):
        self.out.append(text)

    def err(self, text):
        self.errors += 1
        raise CmshError(text)

    @property
    def prompt(self):
        if not self.mode:
            return f"[{HEAD}]% "
        mstar = "*" if any(v.kind == self.mode for v in self.bcm.uncommitted.values()) else ""
        p = f"[{HEAD}->{self.mode}{mstar}"
        if self.obj:
            p += f"[{self.obj}{'*' if self.modified(self.mode, self.obj) else ''}]"
        if self.sub:
            p += f"->{self.sub}"
            if self.sub == "fsmounts" and self.fsobj:
                p += f"[{self.fsobj}]"
        return p + "]% "

    # ------------------------------------------------ dispatch
    def run_line(self, line):
        try:
            argv = shlex.split(line)
        except ValueError as e:
            self.errors += 1
            self.p(f"Error: {e}")
            return
        if not argv:
            return
        try:
            self.execute(argv)
        except CmshError as e:
            self.p(str(e))

    def execute(self, argv):
        cmd, args = argv[0], argv[1:]
        if cmd in MODES:
            self.mode, self.obj, self.sub = cmd, None, None
            if args:
                self.execute(args)
            return
        if cmd in ("home", "top"):
            self.mode = self.obj = self.sub = None
            return
        if cmd in ("exit", ".."):
            if self.sub == "fsmounts" and self.fsobj:
                self.fsobj = None
            elif self.sub:
                self.sub = None
            elif self.obj:
                self.obj = None
            else:
                self.mode = None
            return
        if cmd in ("quit",):
            return
        if cmd == "help":
            return self.help()
        if cmd == "refresh":
            return self.refresh(args)
        if cmd == "modified":
            rows = [(v.kind, v.name, v.op, ",".join(v.fields)) for v in self.bcm.uncommitted.values()]
            self.p(_table(["Type", "Name", "Operation", "Changed"], rows) if rows else "No modified objects")
            return
        if self.mode is None:
            if cmd in ("versioninfo", "licenseinfo"):
                return getattr(self, "main_" + cmd)()
            self.err(f"Unknown command: {cmd}. Enter a mode first ({', '.join(MODES)})")
        if self.mode == "main":
            fn = getattr(self, "main_" + cmd, None)
            if not fn:
                self.err(f"Unknown command in main mode: {cmd}")
            return fn()
        if self.sub == "roles":
            return self.roles_cmd(cmd, args)
        if self.sub == "bmcsettings":
            return self.bmc_cmd(cmd, args)
        if self.sub == "fsmounts":
            return self.fs_cmd(cmd, args)
        common = {
            "list": self.list_,
            "use": self.use,
            "show": self.show,
            "get": self.get,
            "set": self.set_,
            "clear": self.clear,
            "commit": self.commit,
            "clone": self.clone,
            "add": self.add,
            "remove": self.remove,
            "roles": self.enter_roles,
            "bmcsettings": self.enter_bmc,
            "fsmounts": self.enter_fsmounts,
            "usedby": self.usedby,
            "append": self.append,
            "removefrom": self.removefrom,
        }
        if cmd in common:
            return common[cmd](args)
        if self.mode == "device":
            fn = {
                "status": self.dev_status,
                "power": self.dev_power,
                "imageupdate": self.dev_imageupdate,
                "synclog": self.dev_synclog,
                "reboot": self.dev_reboot,
                "drain": self.dev_drain,
                "undrain": self.dev_undrain,
                "pexec": self.dev_pexec,
                "grabimage": self.dev_grabimage,
            }.get(cmd)
            if fn:
                return fn(args)
        if self.mode == "softwareimage" and cmd == "kernelversion":
            return self.get(["kernelversion"] if self.obj else args[:1] + ["kernelversion"])
        self.err(f"Unknown command: {cmd}")

    def help(self):
        base = "list, use <obj>, show, get <prop>, set <prop> <value>, clear <prop>, clone <src> <new>, add, remove, commit, refresh, exit"
        extra = {
            "device": "status, power on|off|reset|status -n <nodes>, imageupdate [-w] -n <nodes>|-c <category>, grabimage [-w] [-i <image>] <node>, synclog, reboot, drain, undrain, pexec, roles, bmcsettings, fsmounts",
            "category": "roles, bmcsettings, fsmounts, usedby, append/removefrom excludelist<grab|grabnew|update|syncinstall|fullinstall> '- /path/*'",
            "user": "set password|id|email|profile <value>, show",
            "group": "add <name>, set id <gid>, append members <user>, removefrom members <user>",
            "profile": "list, show <profile> (assign one with user; use <u>; set profile <name>)",
            "softwareimage": "kernelversion, usedby",
            "partition": "bmcsettings",
            "main": "versioninfo, licenseinfo",
        }
        if not self.mode:
            self.p("Modes: " + ", ".join(MODES))
        else:
            self.p(f"{self.mode} mode: {base}")
            if self.mode in extra:
                self.p("  also: " + extra[self.mode])

    def refresh(self, args):
        keys = [
            k
            for k, v in self.bcm.uncommitted.items()
            if v.kind == self.mode and (self.obj is None or v.name == self.obj)
        ]
        for k in keys:
            del self.bcm.uncommitted[k]

    # ------------------------------------------------ objects
    def _need_mode(self):
        if self.mode not in KIND_COLL:
            self.err(
                "This command needs a mode with objects (device, category, softwareimage, partition, user, group, profile)"
            )

    def _target(self, args, n_rest):
        """Resolve the object for get/set: current object or first arg."""
        if self.obj and len(args) <= n_rest:
            return self.obj, args
        if args and self.exists(self.mode, args[0]):
            return args[0], args[1:]
        if self.obj:
            return self.obj, args
        self.err("No object selected. Use 'use <name>' first")

    def list_(self, args):
        self._need_mode()
        kind = self.mode
        names = sorted(
            {*self.coll(kind), *[v.name for v in self.bcm.uncommitted.values() if v.kind == kind and v.op == "add"]}
        )
        names = [n for n in names if self.exists(kind, n)]
        if kind == "device":
            rows = []
            for n in names:
                d = self.view(kind, n)
                rows.append(
                    (
                        d.type,
                        n,
                        d.get("mac", "").upper(),
                        d.get("category") or "",
                        d.get("ip", ""),
                        "internalnet",
                        f"[{self._status(n):^8}]",
                    )
                )
            self.p(_table(["Type", "Hostname (key)", "MAC", "Category", "Ip", "Network", "Status"], rows))
        elif kind == "category":
            rows = [
                (
                    n,
                    self.view(kind, n).get("softwareimage") or "",
                    sum(1 for d in self.bcm.devices.values() if d.get("category") == n),
                )
                for n in names
            ]
            self.p(_table(["Name (key)", "Software image", "Nodes"], rows, [24, 30, 6]))
        elif kind == "softwareimage":
            rows = []
            for n in names:
                v = self.view(kind, n)
                users = sum(1 for d in self.bcm.devices if d != HEAD and effective_image(self.state, d) == n)
                rows.append((n, v.get("path", ""), v.get("kernel", ""), users))
            self.p(_table(["Name (key)", "Path", "Kernel version", "Nodes"], rows))
        elif kind == "partition":
            rows = [
                (n, self.view(kind, n).get("clustername", ""), self.view(kind, n).get("headnode", "")) for n in names
            ]
            self.p(_table(["Name (key)", "Cluster name", "Head node"], rows))
        elif kind == "user":
            rows = []
            for n in names:
                u = self.view(kind, n)
                rows.append((n, u.get("uid", ""), n if n != "root" else "root", ",".join(self._member_of(n))))
            self.p(_table(["Name (key)", "ID", "Primary group", "Secondary groups"], rows))
        elif kind == "group":
            rows = [
                (n, self.view(kind, n).get("gid", ""), ",".join(self.view(kind, n).get("members", []))) for n in names
            ]
            self.p(_table(["Name (key)", "ID", "Members"], rows))
        elif kind == "profile":
            rows = [(n, "yes" if self.view(kind, n).get("builtin") else "no") for n in names]
            self.p(_table(["Name (key)", "Built-in"], rows))

    def use(self, args):
        self._need_mode()
        if not args:
            self.err("Usage: use <name>")
        name = args[0]
        if not self.exists(self.mode, name):
            self.err(f"Error: {name} not found")
        self.obj, self.sub = name, None

    def _status(self, name):
        if name == HEAD:
            return "UP"
        node = self.state.nodes.get(name)
        dev = self.bcm.devices[name]
        if dev.get("status") not in (None, "UP", "DOWN"):
            return dev.status
        if node is None:
            return dev.get("status") or "DOWN"
        return "UP" if (node.up and node.power == "on") else "DOWN"

    def show(self, args):
        self._need_mode()
        name = args[0] if args else self.obj
        if not name:
            self.err("No object selected. Use 'use <name>' or 'show <name>'")
        if not self.exists(self.mode, name):
            self.err(f"Error: {name} not found")
        v = self.view(self.mode, name)
        rows = []
        if self.mode == "device":
            cat = v.get("category") or ""
            catv = self.view("category", cat) if cat and self.exists("category", cat) else {}
            img = v.get("softwareimage") or ""
            img_disp = img if img else (f"{catv.get('softwareimage', '')} (from category {cat})" if cat else "")
            im = v.get("installmode") or ""
            im_disp = im if im else (f"{catv.get('installmode', 'AUTO')} (from category {cat})" if cat else "AUTO")
            rows = [
                ("Activation", "Mon Oct 12 09:12:44 2026"),
                ("BMC settings", "<submode>"),
                ("Category", cat),
                ("Hostname", name),
                ("Install mode", im_disp),
                ("IP", v.get("ip", "")),
                ("MAC", v.get("mac", "").upper()),
                ("Network", "internalnet"),
                ("Next install mode", v.get("nextinstallmode") or ""),
                ("Power control", "ipmi0" if v.type != "HeadNode" else "none"),
                (
                    "Provisioned image",
                    f"{v.get('synced_image', '')} (version {v.get('provisioned_image_version', '')})"
                    if v.type != "HeadNode"
                    else "",
                ),
                (
                    "Roles",
                    ", ".join(v.get("roles", []))
                    or (
                        f"{', '.join(catv.get('roles', []))} (from category {cat})"
                        if catv and catv.get("roles")
                        else ""
                    ),
                ),
                ("Software image", img_disp),
                ("Status", self._status(name)),
                ("Type", v.type),
            ]
        elif self.mode == "category":
            rows = [
                ("Name", name),
                ("BMC settings", "<submode>"),
                ("Disk setup", v.get("disksetup", "")),
                ("Install mode", v.get("installmode", "AUTO")),
                ("Roles", ", ".join(v.get("roles", []))),
                ("Software image", v.get("softwareimage", "")),
                ("FS mounts", f"<{len(v.get('fsmounts', []))} in submode>"),
                *[
                    (x.replace("excludelist", "Exclude list "), f"<{len(exclude_patterns(v.get(x, '')))} entries>")
                    for x in EXCLUDE_LISTS
                ],
                ("Nodes", ",".join(sorted(n for n, d in self.bcm.devices.items() if d.get("category") == name))),
            ]
        elif self.mode == "softwareimage":
            pk = v.get("packages", {})
            rows = [
                ("Name", name),
                ("Path", v.get("path", "")),
                ("Kernel version", v.get("kernel", "")),
                ("Kernel parameters", v.get("kernelparameters", "")),
                ("Kernel modules", "<submode>"),
                ("Revision", v.get("version", "")),
                ("Installed kernels", ", ".join(_kernels(pk))),
                ("Boot FSPart", "98784247813"),
                ("Enable SOL", "no"),
            ]
        elif self.mode == "partition":
            rows = [
                ("Name", name),
                ("Cluster name", v.get("clustername", "")),
                ("Head node", v.get("headnode", "")),
                ("BMC settings", "<submode>"),
                ("Time zone", "UTC"),
                ("Name servers", "10.141.255.254"),
            ]
        elif self.mode == "user":
            rows = [
                ("Name", name),
                ("ID", v.get("uid", "")),
                ("Password", "*********" if v.get("password") else "< not set >"),
                ("Login shell", v.get("shell", "/bin/bash")),
                ("Home directory", v.get("home", f"/home/{name}")),
                ("Email", v.get("email", "")),
                ("Primary group", name),
                ("Groups", ",".join(self._member_of(name))),
                ("Profile", v.get("profile", "") or "< not set >"),
            ]
        elif self.mode == "group":
            rows = [("Name", name), ("ID", v.get("gid", "")), ("Members", ",".join(v.get("members", [])))]
        elif self.mode == "profile":
            rows = [
                ("Name", name),
                ("Description", v.get("description", "")),
                ("Non user", "no"),
                ("Built-in", "yes" if v.get("builtin") else "no"),
            ]
        self.p(_table(["Parameter", "Value"], rows, [32, 48]))

    def _prop(self, prop):
        fields = PROPS[self.mode]
        if prop not in fields:
            self.err(f"Error: unknown property '{prop}' (properties: {', '.join(sorted(fields))})")
        return fields[prop]

    def get(self, args):
        self._need_mode()
        name, rest = self._target(args, 1)
        if not rest:
            self.err("Usage: get <property>")
        field = self._prop(rest[0])
        v = self.view(self.mode, name)
        if self.mode == "device" and field == "softwareimage" and not v.get("softwareimage"):
            cat = v.get("category")
            val = self.view("category", cat).get("softwareimage", "") if cat else ""
        elif self.mode == "device" and field == "status":
            val = self._status(name)
        elif self.mode == "user" and field == "password":
            val = "*********" if v.get("password") else ""
        elif field in EXCLUDE_LISTS:
            val = str(v.get(field, "")).rstrip("\n")
        elif field == "members":
            val = ",".join(v.get("members", []))
        else:
            val = v.get(field)
        self.p("" if val is None else str(val))

    def set_(self, args):
        self._need_mode()
        name, rest = self._target(args, 2)
        if len(rest) < 2:
            self.err("Usage: set <property> <value>")
        prop, value = rest[0], " ".join(rest[1:])
        field = self._prop(prop)
        if prop in READONLY.get(self.mode, ()):
            self.err(f"Error: property '{prop}' is read-only")
        if field == "members":
            value = [m for m in value.replace(" ", ",").split(",") if m]
            for m in value:
                self._need_user(m)
        elif field in EXCLUDE_LISTS:
            value = value.replace("\\n", "\n").rstrip("\n") + "\n"
        else:
            value = self._validate(prop, value, name)
        self.stage(self.mode, name, field, value)

    def _need_user(self, user):
        if user != "root" and not self.exists("user", user):
            self.err(f"Error: user '{user}' does not exist")

    def _member_of(self, user):
        out = []
        for g in sorted(set(self.bcm.groups) | {v.name for v in self.bcm.uncommitted.values() if v.kind == "group"}):
            if self.exists("group", g) and user in self.view("group", g).get("members", []):
                out.append(g)
        return out

    def _listprop(self, args, verb):
        self._need_mode()
        name, rest = self._target(args, 2)
        if len(rest) < 2:
            self.err(f"Usage: {verb} <property> <value>")
        prop, value = rest[0], " ".join(rest[1:])
        field = self._prop(prop)
        cur = self.view(self.mode, name)
        return name, prop, field, value, cur

    def append(self, args):
        name, prop, field, value, cur = self._listprop(args, "append")
        if field == "members":
            members = list(cur.get("members", []))
            for m in [m for m in value.replace(" ", ",").split(",") if m]:
                self._need_user(m)
                if m not in members:
                    members.append(m)
            self.stage(self.mode, name, field, members)
        elif field in EXCLUDE_LISTS:
            text = str(cur.get(field, "")).rstrip("\n")
            self.stage(self.mode, name, field, (text + "\n" if text else "") + value + "\n")
        else:
            self.err(f"Error: property '{prop}' is not a list (use set)")

    def removefrom(self, args):
        name, prop, field, value, cur = self._listprop(args, "removefrom")
        if field == "members":
            gone = [m for m in value.replace(" ", ",").split(",") if m]
            self.stage(self.mode, name, field, [m for m in cur.get("members", []) if m not in gone])
        elif field in EXCLUDE_LISTS:
            lines = str(cur.get(field, "")).splitlines()
            if value not in [ln.strip() for ln in lines]:
                self.err(f"Error: '{value}' is not in {prop}")
            self.stage(self.mode, name, field, "\n".join(ln for ln in lines if ln.strip() != value) + "\n")
        else:
            self.err(f"Error: property '{prop}' is not a list")

    def _validate(self, prop, value, name):
        if prop == "category" and not self.exists("category", value):
            self.err(f"Error: category '{value}' does not exist")
        if prop == "softwareimage" and not self.exists("softwareimage", value):
            self.err(f"Error: software image '{value}' does not exist")
        if prop in ("installmode", "nextinstallmode"):
            value = value.upper()
            if value not in INSTALL_MODES:
                self.err(f"Error: invalid install mode '{value}', valid: {', '.join(INSTALL_MODES)}")
        if prop == "kernelversion":
            pk = self.view("softwareimage", name).get("packages", {})
            if value not in _kernels(pk):
                path = self.view("softwareimage", name).get("path", "")
                self.err(
                    f"Error: kernel version {value} not found in {path}/lib/modules "
                    f"(install the linux-image package inside the image first)"
                )
        if prop == "profile" and value and not self.exists("profile", value):
            self.err(f"Error: profile '{value}' does not exist (see: profile; list)")
        if prop in ("id", "uid", "groupid"):
            try:
                value = int(value)
            except ValueError:
                self.err("Error: ID must be a number")
        return value

    def clear(self, args):
        self._need_mode()
        name, rest = self._target(args, 1)
        if not rest:
            self.err("Usage: clear <property>")
        field = self._prop(rest[0])
        if self.mode in ("device",) and field in ("softwareimage", "installmode", "nextinstallmode"):
            self.stage(self.mode, name, field, None)
        elif self.mode == "user" and field in ("password", "email", "profile"):
            self.stage(self.mode, name, field, "")
        else:
            self.err(f"Error: property '{rest[0]}' cannot be cleared")

    def clone(self, args):
        self._need_mode()
        if self.mode not in ("category", "softwareimage", "device", "user", "profile"):
            self.err(f"clone is not supported in {self.mode} mode")
        if len(args) == 1 and self.obj:
            args = [self.obj, args[0]]
        if len(args) != 2:
            self.err("Usage: clone <source> <new-name>")
        src, new = args
        if not self.exists(self.mode, src):
            self.err(f"Error: {src} not found")
        if self.exists(self.mode, new):
            self.err(f"Error: {new} already exists")
        obj = dict(self.view(self.mode, src))
        obj["name"] = new
        obj.pop("builtin", None)
        if self.mode == "softwareimage":
            obj["path"] = f"/cm/images/{new}"
            obj["version"] = 1
            obj["cloned_from"] = src
        if self.mode == "device":
            obj["hostname"] = new
            obj["ip"] = ""
            obj["mac"] = "00:00:00:00:00:00"
        self.bcm.uncommitted[self._key(self.mode, new)] = {"kind": self.mode, "name": new, "op": "add", "fields": obj}
        self.obj = new

    def add(self, args):
        self._need_mode()
        kind = self.mode
        if kind == "device":
            if len(args) < 2:
                self.err("Usage: add physicalnode <hostname> [ip]")
            typ = {"physicalnode": "PhysicalNode"}.get(args[0].lower())
            if not typ:
                self.err(f"Error: unsupported device type {args[0]}")
            name = args[1]
            obj = {
                "hostname": name,
                "type": typ,
                "category": "default",
                "status": "DOWN",
                "ip": args[2] if len(args) > 2 else "",
                "mac": "00:00:00:00:00:00",
                "softwareimage": None,
                "roles": [],
                "installmode": None,
                "power": "off",
                "provisioned_image_version": 0,
                "bmcsettings": {},
                "nextinstallmode": None,
            }
        elif kind == "user":
            if not args:
                self.err("Usage: add <username>")
            name = args[0]
            used = [u.get("uid", 0) for u in self.bcm.users.values()]
            used += [g.get("gid", 0) for g in self.bcm.groups.values()]
            obj = {
                "name": name,
                "uid": max([999, *used]) + 1,
                "groups": [name],
                "password": "",
                "home": f"/home/{name}",
                "shell": "/bin/bash",
                "profile": "",
            }
        elif kind == "group":
            if not args:
                self.err("Usage: add <groupname>")
            name = args[0]
            used = [g.get("gid", 0) for g in self.bcm.groups.values()]
            used += [u.get("uid", 0) for u in self.bcm.users.values()]
            obj = {"name": name, "gid": max([999, *used]) + 1, "members": []}
        elif kind == "category":
            if not args:
                self.err("Usage: add <name>")
            name = args[0]
            obj = {
                "name": name,
                "softwareimage": "default-image",
                "roles": [],
                "installmode": "AUTO",
                "disksetup": "default.xml",
                "bmcsettings": {},
            }
        else:
            self.err(f"add is not supported in {kind} mode (use clone)")
        if self.exists(kind, name):
            self.err(f"Error: {name} already exists")
        self.bcm.uncommitted[self._key(kind, name)] = {"kind": kind, "name": name, "op": "add", "fields": obj}
        self.obj = name

    def remove(self, args):
        self._need_mode()
        name = args[0] if args else self.obj
        if not name or not self.exists(self.mode, name):
            self.err(f"Error: {name} not found")
        if self.mode == "profile" and self.view("profile", name).get("builtin"):
            self.err(f"Error: {name} is a built-in profile and cannot be removed")
        if self.mode == "softwareimage" and any(c.get("softwareimage") == name for c in self.bcm.categories.values()):
            self.err(f"Error: software image {name} is in use by a category")
        self.bcm.uncommitted[self._key(self.mode, name)] = {
            "kind": self.mode,
            "name": name,
            "op": "remove",
            "fields": {},
        }
        if self.obj == name:
            self.obj = None

    def usedby(self, args):
        self._need_mode()
        name = args[0] if args else self.obj
        if not name:
            self.err("Usage: usedby <name>")
        rows = []
        if self.mode == "softwareimage":
            rows += [
                ("Category", c, "softwareimage")
                for c, v in sorted(self.bcm.categories.items())
                if v.get("softwareimage") == name
            ]
            rows += [
                ("Device", d, "softwareimage")
                for d, v in sorted(self.bcm.devices.items())
                if v.get("softwareimage") == name
            ]
        elif self.mode == "category":
            rows += [
                ("Device", d, "category") for d, v in sorted(self.bcm.devices.items()) if v.get("category") == name
            ]
        self.p(_table(["Entity", "Name", "Property"], rows))

    # ------------------------------------------------ commit
    def commit(self, args):
        if "-a" in args or "--all" in args:
            keys = list(self.bcm.uncommitted)
        elif self.obj:
            keys = [self._key(self.mode, self.obj)]
        else:
            keys = [k for k, v in self.bcm.uncommitted.items() if v.kind == self.mode]
        keys = [k for k in keys if k in self.bcm.uncommitted]
        for k in keys:
            self._apply(self.bcm.uncommitted[k])
            del self.bcm.uncommitted[k]

    def _apply(self, st):
        kind, name, coll = st.kind, st.name, self.coll(st.kind)
        fields = copy.deepcopy(dict(st.fields))
        if st.op == "remove":
            coll.pop(name, None)
            return
        if st.op == "add":
            coll[name] = fields
            if kind == "softwareimage":
                self._copy_image(fields.get("cloned_from"), name)
            if kind == "user" and name not in self.bcm.groups:
                self.bcm.groups[name] = {"name": name, "gid": fields.get("uid", 1000), "members": []}
            if kind == "user":
                self.p(f"{NOW} [notice] {HEAD}: User {name} added to LDAP, home directory /home/{name} created")
            return
        old = coll[name]
        before = dict(old)
        old.update(fields)
        if "fsmounts" in fields and kind in ("category", "device"):
            self._push_fsmounts(kind, name)
        if kind == "softwareimage" and "kernel" in fields and fields["kernel"] != before.get("kernel"):
            old.version = old.get("version", 1) + 1
            self.p(f"{NOW} [notice] {HEAD}: Initial ramdisk for image {name} is being generated")
            self.p(f"{NOW} [notice] {HEAD}: Initial ramdisk for image {name} was generated successfully")
        if kind == "device" and name != HEAD and ("category" in fields or "softwareimage" in fields):
            new_img = effective_image(self.state, name)
            if new_img != old.get("synced_image"):
                self.p(
                    f"{NOW} [warning] {HEAD}: {name}: software image changed to {new_img}; "
                    f"reboot the node (or run imageupdate) to provision it"
                )
        if kind == "category" and "softwareimage" in fields and fields["softwareimage"] != before.get("softwareimage"):
            nodes = [d for d, v in self.bcm.devices.items() if v.get("category") == name]
            if nodes:
                self.p(
                    f"{NOW} [notice] {HEAD}: category {name}: software image changed to {fields['softwareimage']}. "
                    f"Running nodes keep their current files until imageupdate or reboot "
                    f"({', '.join(sorted(nodes))})"
                )

    def _copy_image(self, src, new):
        if not src or src not in self.bcm.softwareimages:
            return
        head = self.state.nodes[HEAD].files
        sp = self.bcm.softwareimages[src].path.rstrip("/") + "/"
        dp = f"/cm/images/{new}/"
        for k in [k for k in head if k.startswith(sp)]:
            head[dp + k[len(sp) :]] = head[k]
        self.bcm.softwareimages[new].packages = dict(self.bcm.softwareimages[src].get("packages", {}))
        self.p(f"{NOW} [notice] {HEAD}: Started to copy: /cm/images/{src} -> /cm/images/{new}")
        self.p(f"{NOW} [notice] {HEAD}: Copied: /cm/images/{src} -> /cm/images/{new}")
        self.p(f"{NOW} [notice] {HEAD}: Initial ramdisk for image {new} was generated successfully")

    # ------------------------------------------------ roles / bmcsettings submodes
    def enter_roles(self, args):
        if self.mode not in ("device", "category") or not self.obj:
            self.err("roles: select a device or category first (use <name>)")
        self.sub = "roles"
        if args:
            self.roles_cmd(args[0], args[1:])

    def roles_cmd(self, cmd, args):
        roles = list(self.view(self.mode, self.obj).get("roles", []))
        if cmd == "list":
            self.p(_table(["Name (key)"], [(r,) for r in roles]))
        elif cmd in ("assign", "add"):
            if not args:
                self.err("Usage: assign <role>")
            if args[0] not in (
                "slurmclient",
                "slurmserver",
                "boot",
                "provisioning",
                "monitoring",
                "kubernetes::node",
                "storage",
                "login",
            ):
                self.err(f"Error: unknown role {args[0]}")
            if args[0] not in roles:
                roles.append(args[0])
            self.stage(self.mode, self.obj, "roles", roles)
        elif cmd in ("unassign", "remove"):
            if not args or args[0] not in roles:
                self.err(f"Error: role {args[0] if args else ''} is not assigned")
            roles.remove(args[0])
            self.stage(self.mode, self.obj, "roles", roles)
        elif cmd == "commit":
            self.commit(args)
        elif cmd == "show":
            self.p(_table(["Parameter", "Value"], [("Roles", ", ".join(roles))], [32, 48]))
        else:
            self.err(f"Unknown command in roles submode: {cmd}")

    def enter_bmc(self, args):
        if self.mode not in ("device", "category", "partition") or not self.obj:
            self.err("bmcsettings: select a device, category or partition first (use <name>)")
        self.sub = "bmcsettings"
        if args:
            self.bmc_cmd(args[0], args[1:])

    def bmc_cmd(self, cmd, args):
        cur = dict(self.view(self.mode, self.obj).get("bmcsettings", {}) or {})
        if cmd in ("show", "list"):
            rows = []
            for key, label in (("username", "User name"), ("password", "Password"), ("userid", "User ID")):
                val = cur.get(key)
                if key == "password" and val:
                    val = "*********"
                inherited = "" if val not in (None, "") else "(inherited)"
                rows.append((label, val if val not in (None, "") else inherited))
            self.p(_table(["Parameter", "Value"], rows, [32, 48]))
        elif cmd == "get":
            if not args:
                self.err("Usage: get <username|password|userid>")
            val = cur.get(args[0], "")
            self.p("*********" if args[0] == "password" and val else str(val or ""))
        elif cmd == "set":
            if len(args) < 2 or args[0] not in ("username", "password", "userid"):
                self.err("Usage: set <username|password|userid> <value>")
            cur[args[0]] = int(args[1]) if args[0] == "userid" else args[1]
            self.stage(self.mode, self.obj, "bmcsettings", cur)
        elif cmd == "clear":
            if not args:
                self.err("Usage: clear <username|password|userid>")
            cur.pop(args[0], None)
            self.stage(self.mode, self.obj, "bmcsettings", cur)
        elif cmd == "commit":
            self.commit(args)
        else:
            self.err(f"Unknown command in bmcsettings submode: {cmd}")

    # ------------------------------------------------ fsmounts submode
    def enter_fsmounts(self, args):
        if self.mode not in ("device", "category") or not self.obj:
            self.err("fsmounts: select a category or device first (use <name>)")
        self.sub, self.fsobj = "fsmounts", None
        if args:
            self.fs_cmd(args[0], args[1:])

    def _fs_list(self):
        return [dict(e) for e in self.view(self.mode, self.obj).get("fsmounts", []) or []]

    def _fs_find(self, entries, mp):
        mp = mp.rstrip("/") or "/"
        for e in entries:
            if (e.get("mountpoint", "").rstrip("/") or "/") == mp:
                return e
        return None

    def fs_cmd(self, cmd, args):
        entries = self._fs_list()
        if cmd == "list":
            rows = [
                (e.get("device", ""), e.get("mountpoint", ""), e.get("filesystem", ""), e.get("mountoptions", ""))
                for e in entries
            ]
            self.p(_table(["Device", "Mountpoint (key)", "Filesystem", "Mount options"], rows))
            return
        if cmd == "add":
            if not args:
                self.err("Usage: add <mountpoint>")
            if self._fs_find(entries, args[0]):
                self.err(f"Error: fsmount {args[0]} already exists")
            entries.append(
                {
                    "mountpoint": args[0].rstrip("/") or "/",
                    "device": "",
                    "filesystem": "nfs",
                    "mountoptions": "defaults",
                }
            )
            self.stage(self.mode, self.obj, "fsmounts", entries)
            self.fsobj = args[0].rstrip("/") or "/"
            return
        if cmd == "use":
            if not args or not self._fs_find(entries, args[0]):
                self.err(f"Error: fsmount {args[0] if args else ''} not found")
            self.fsobj = args[0].rstrip("/") or "/"
            return
        if cmd == "remove":
            mp = args[0] if args else self.fsobj
            if not mp or not self._fs_find(entries, mp):
                self.err(f"Error: fsmount {mp} not found")
            entries = [e for e in entries if e is not self._fs_find(entries, mp)]
            self.stage(self.mode, self.obj, "fsmounts", entries)
            if self.fsobj == (mp.rstrip("/") or "/"):
                self.fsobj = None
            return
        if cmd in ("set", "get", "show", "clear"):
            mp = self.fsobj
            if (
                args
                and self._fs_find(entries, args[0])
                and (cmd in ("show",) or len(args) > (2 if cmd == "set" else 1))
            ):
                mp, args = args[0], args[1:]
            e = self._fs_find(entries, mp) if mp else None
            if e is None:
                self.err("Error: no fsmount selected (use <mountpoint>)")
            if cmd == "show":
                rows = [
                    (k.capitalize() if k != "mountoptions" else "Mount options", e.get(k, "")) for k in FSMOUNT_PROPS
                ]
                self.p(_table(["Parameter", "Value"], rows, [32, 48]))
                return
            if not args or args[0] not in FSMOUNT_PROPS:
                self.err(f"Usage: {cmd} <{'|'.join(FSMOUNT_PROPS)}>{' <value>' if cmd == 'set' else ''}")
            if cmd == "get":
                self.p(str(e.get(args[0], "")))
                return
            if cmd == "clear":
                e[args[0]] = ""
            else:
                if len(args) < 2:
                    self.err("Usage: set <property> <value>")
                val = " ".join(args[1:])
                if args[0] == "mountpoint":
                    val = val.rstrip("/") or "/"
                    self.fsobj = val
                e[args[0]] = val
            self.stage(self.mode, self.obj, "fsmounts", entries)
            return
        if cmd == "commit":
            return self.commit(args)
        self.err(f"Unknown command in fsmounts submode: {cmd}")

    def _push_fsmounts(self, kind, name):
        from .storage import apply_fsmounts

        if kind == "category":
            hosts = sorted(d for d, v in self.bcm.devices.items() if v.get("category") == name and d != HEAD)
        else:
            hosts = [name] if name != HEAD else []
        for h in hosts:
            node = self.state.nodes.get(h)
            if node is None or not node.get("up", True) or node.get("power", "on") != "on":
                continue
            msgs = apply_fsmounts(self.state, h, effective_fsmounts(self.state, h))
            self.p(f"{NOW} [notice] {HEAD}: {h}: fsmounts updated, /etc/fstab rewritten and mounts refreshed")
            for m in msgs:
                self.p(f"{NOW} [warning] {HEAD}: {m}")

    # ------------------------------------------------ grabimage (node -> image)
    def dev_grabimage(self, args):
        wait = "-w" in args or "--wait" in args
        image, rest, it = None, [], iter(a for a in args if a not in ("-w", "--wait"))
        for a in it:
            if a in ("-i", "--image"):
                image = next(it, None)
            else:
                rest.append(a)
        nodes, _ = self._nodes_from(rest)
        if len(nodes) != 1:
            self.err("Usage: grabimage [-w] [-i <image>] <node>   (one node at a time)")
        n = nodes[0]
        if n == HEAD:
            self.err("Error: grabimage cannot be used on the head node")
        node = self.state.nodes[n]
        if not node.up or node.power != "on":
            self.errors += 1
            self.p(f"{n}: Node is DOWN, cannot grab its filesystem")
            return
        img = image or effective_image(self.state, n)
        if img not in self.bcm.softwareimages:
            self.err(f"Error: software image '{img}' does not exist (clone and commit it first)")
        cat = self.bcm.categories.get(self.bcm.devices[n].get("category") or "") or {}
        lst = "excludelistgrabnew" if image and image != effective_image(self.state, n) else "excludelistgrab"
        pats = exclude_patterns(cat.get(lst, DEFAULT_EXCLUDE))
        imgobj = self.bcm.softwareimages[img]
        path = imgobj.path.rstrip("/")
        dry = "no" if wait else "yes, no data changes!"
        lines = [
            f"{NOW} [notice] {HEAD}: Provisioning started: sending {n}:/ to {HEAD}:{path}, mode GRAB, dry run = {dry}",
            f"{NOW} [notice] {HEAD}: Provisioning completed: sent {n}:/ to {HEAD}:{path}, mode GRAB, dry run = {dry}",
        ]
        current = image_files(self.state, img)
        wanted = {p: c for p, c in node.files.items() if not is_excluded(p, pats)}
        changes = []
        node_pk, img_pk = node_packages(self.state, n), imgobj.get("packages", {})
        for p in sorted(set(node_pk) | set(img_pk)):
            if node_pk.get(p) != img_pk.get(p):
                changes.append(f"  {p}: {img_pk.get(p) or '(absent)'} -> {node_pk.get(p) or '(removed)'}")
        for p in sorted(wanted):
            if p not in current:
                changes.append(f"  >f+++++++++ {p.lstrip('/')}")
            elif current[p] != wanted[p]:
                changes.append(f"  >f.st...... {p.lstrip('/')}")
        for p in sorted(current):
            if p not in wanted and not is_excluded(p, pats):
                changes.append(f"  *deleting   {p.lstrip('/')}")
        skipped = sorted(p for p in node.files if is_excluded(p, pats))
        note = [f"  (excluded by {lst}: {len(skipped)} file(s))"] if skipped else []
        self.bcm.synclog[n] = lines + (changes or ["(no file changes)"]) + note
        self.out.extend(lines)
        if not wait:
            self.bcm.grab_dryruns.append(n)
            self.p(f"grabimage -i {img} {n} {'.' * 10} [ COMPLETED ] (dry run; review with: synclog {n})")
            return
        head = self.state.nodes[HEAD].files
        for p in list(current):
            if p not in wanted and not is_excluded(p, pats):
                head.pop(path + p, None)
        for p, c in wanted.items():
            head[path + p] = c
        imgobj.packages = dict(node_pk)
        imgobj.version = imgobj.get("version", 1) + 1
        self.bcm.setdefault("grabs", []).append({"node": n, "image": img})
        self.p(f"grabimage -w -i {img} {n} {'.' * 10} [ COMPLETED ]")

    # ------------------------------------------------ device operations
    def _nodes_from(self, args, allow_positional=True):
        nodes, rest, it = [], [], iter(args)
        for a in it:
            if a in ("-n", "--nodes"):
                nodes += expand_nodes(next(it, ""))
            elif a in ("-c", "--category"):
                cat = next(it, "")
                if cat not in self.bcm.categories:
                    self.err(f"Error: category '{cat}' does not exist")
                nodes += sorted(n for n, d in self.bcm.devices.items() if d.get("category") == cat)
            else:
                rest.append(a)
        if not nodes and allow_positional:
            pos = [a for a in rest if not a.startswith("-") and a in self.bcm.devices]
            nodes = pos
            rest = [a for a in rest if a not in pos]
        if not nodes and self.obj:
            nodes = [self.obj]
        for n in nodes:
            if n not in self.bcm.devices:
                self.err(f"Error: device {n} not found")
        return nodes, rest

    def dev_status(self, args):
        nodes, _ = self._nodes_from(args)
        nodes = nodes or sorted(self.bcm.devices)
        for n in nodes:
            st = self._status(n)
            extra = ""
            node = self.state.nodes.get(n)
            if node is not None and st == "DOWN" and node.power != "on":
                extra = "(power off)"
            self.p(_dots(n, st, extra))

    def _bmc_ok(self, n):
        return effective_bmc(self.state, n).get("password") == self.bcm.bmc_actual_password

    def dev_power(self, args):
        if not args:
            self.err("Usage: power <status|on|off|reset|cycle> [-n nodes | -c category]")
        action = args[0]
        if action not in ("status", "on", "off", "reset", "cycle"):
            self.err(f"Error: unknown power action {action}")
        nodes, _ = self._nodes_from(args[1:])
        nodes = nodes or sorted(n for n in self.bcm.devices if n != HEAD)
        for n in nodes:
            if n == HEAD:
                self.p(_dots("ipmi0", "FAILED", f"{n} (no power control for the head node)"))
                continue
            node = self.state.nodes[n]
            if not self._bmc_ok(n):
                self.errors += 1
                self.p(_dots("ipmi0", "FAILED", f"{n} (Unable to establish IPMI v2 / RMCP+ session)"))
                continue
            dev = self.bcm.devices[n]
            if action == "status":
                self.p(_dots("ipmi0", node.power.upper(), n))
            elif action == "off":
                if audit.running_jobs_on(self.state, n):
                    audit.record(self.state, audit.POWER_OFF_WITH_JOBS, HEAD, "cmsh power off", n)
                node.power, node.up, dev.status, dev.power = "off", False, "DOWN", "off"
                sl = self.state.slurm.nodes.get(n)
                if sl is not None and not str(sl.state).startswith("DRAIN"):
                    sl.state, sl.reason = "DOWN", "Not responding"
                    sl["down"] = True
                self.p(_dots("ipmi0", "OFF", n))
            elif action == "on":
                if node.power == "on" and node.up:
                    self.p(_dots("ipmi0", "ON", n))
                    continue
                dev.power = "on"
                mode, img = provision(self.ctx, n, "power on")
                self.p(_dots("ipmi0", "ON", n))
                self.p(f"{NOW} [notice] {HEAD}: {n} [ INSTALLING ] (node-installer: install mode {mode}, image {img})")
                self.p(f"{NOW} [notice] {HEAD}: {n} [   UP   ]")
            else:  # reset / cycle
                if audit.running_jobs_on(self.state, n):
                    audit.record(self.state, audit.POWER_OFF_WITH_JOBS, HEAD, f"cmsh power {action}", n)
                dev.power = "on"
                mode, img = provision(self.ctx, n, "power " + action)
                self.p(_dots("ipmi0", "RESET" if action == "reset" else "ON", n))
                self.p(f"{NOW} [notice] {HEAD}: {n} [   UP   ] (install mode {mode}, image {img})")

    def dev_reboot(self, args):
        nodes, _ = self._nodes_from(args)
        if not nodes:
            self.err("Usage: reboot -n <nodes>")
        for n in nodes:
            node = self.state.nodes.get(n)
            if n == HEAD:
                self.err("Refusing to reboot the head node from cmsh in the lab")
            if not node.up or node.power != "on":
                self.errors += 1
                self.p(f"{n}: Node is DOWN; use 'power on -n {n}' (reboot needs a running OS)")
                continue
            mode, img = provision(self.ctx, n, "reboot")
            self.p(f"{n}: Reboot in progress ...")
            self.p(f"{NOW} [notice] {HEAD}: {n} [ INSTALLING ] (node-installer: install mode {mode}, image {img})")
            self.p(f"{NOW} [notice] {HEAD}: {n} [   UP   ]")

    def dev_imageupdate(self, args):
        wait = "-w" in args or "--wait" in args
        nodes, _ = self._nodes_from([a for a in args if a not in ("-w", "--wait")])
        if not nodes:
            self.err("Usage: imageupdate [-w] -n <nodes> | -c <category>")
        if not wait:
            self.p("Performing dry run (use synclog command to review result, then pass -w to perform real update)...")
        for n in nodes:
            if n == HEAD:
                continue
            node = self.state.nodes[n]
            if not node.up or node.power != "on":
                self.errors += 1
                self.p(f"{n}: Node is DOWN, skipping imageupdate")
                continue
            img = effective_image(self.state, n)
            path = self.bcm.softwareimages[img].path
            dry = "yes, no data changes!" if not wait else "no"
            lines = [
                f"{NOW} [notice] {HEAD}: Provisioning started: sending {HEAD}:{path} to {n}:/, mode UPDATE, dry run = {dry}",
                f"{NOW} [notice] {HEAD}: Provisioning completed: sent {HEAD}:{path} to {n}:/, mode UPDATE, dry run = {dry}",
            ]
            changes = self._diff(n, img)
            self.bcm.synclog[n] = lines + (changes or ["(no file changes)"])
            self.out.extend(lines)
            if wait:
                old_kernel = node.kernel
                sync_image(self.state, n)
                self.bcm.image_updates.append(n)
                new_kernel = self.bcm.softwareimages[img].kernel
                if new_kernel != old_kernel:
                    self.p(
                        f"{NOW} [warning] {HEAD}: {n}: image kernel {new_kernel} differs from running kernel "
                        f"{old_kernel}; a reboot is required for the new kernel to be used"
                    )
                self.p(f"imageupdate -w -n {n} {'.' * 10} [ COMPLETED ]")
            else:
                self.bcm.dryruns.append(n)
                self.p(f"imageupdate -n {n} {'.' * 10} [ COMPLETED ] (dry run)")

    def _diff(self, n, img):
        node_pk = node_packages(self.state, n)
        img_pk = self.bcm.softwareimages[img].get("packages", {})
        out = []
        for p in sorted(set(node_pk) | set(img_pk)):
            a, b = node_pk.get(p), img_pk.get(p)
            if a != b:
                out.append(f"  {p}: {a or '(absent)'} -> {b or '(removed)'}")
        for path, content in sorted(image_files(self.state, img).items()):
            have = self.state.nodes[n].files.get(path)
            if have is None:
                out.append(f"  >f+++++++++ {path.lstrip('/')}")
            elif have != content:
                out.append(f"  >f.st...... {path.lstrip('/')}")
        return out

    def dev_synclog(self, args):
        nodes, _ = self._nodes_from(args)
        if not nodes:
            self.err("Usage: synclog <node>")
        for n in nodes:
            log = self.bcm.synclog.get(n)
            self.p(f"Sync log for {n}:")
            self.p("\n".join(log) if log else "  (no provisioning has been run for this node)")

    def _slurm_drain(self, args, drain):
        reason = ""
        clean, it = [], iter(args)
        for a in it:
            if a in ("-r", "--reason"):
                reason = next(it, "")
            else:
                clean.append(a)
        nodes, _ = self._nodes_from(clean)
        if not nodes:
            self.err(f"Usage: {'drain' if drain else 'undrain'} -n <nodes> [--reason text]")
        rows = []
        for n in nodes:
            sl = self.state.slurm.nodes.get(n)
            if sl is None:
                rows.append(("slurm", n, "Not in WLM", ""))
                continue
            if drain:
                sl.state = "DRAINING" if sl.state in ("ALLOCATED", "MIXED", "DRAINING") else "DRAIN"
                sl.reason = reason or "drained by cmsh"
                sl["drain"] = True
                sl["reason_user"] = "root"
                rows.append(("slurm", n, "Draining" if sl.state == "DRAINING" else "Drained", sl.reason))
            else:
                sl.state, sl.reason = "IDLE", ""
                sl["drain"] = False
                sl["down"] = False
                rows.append(("slurm", n, "Ok", ""))
        self.p(_table(["Engine", "Node", "Status", "Reason"], rows, [24, 24, 16, 24]))

    def dev_drain(self, args):
        self._slurm_drain(args, True)

    def dev_undrain(self, args):
        self._slurm_drain(args, False)

    def dev_pexec(self, args):
        cmd = [a for a in args if a not in ("-n", "-c")]
        nodes, rest = self._nodes_from(args, allow_positional=False)
        if not nodes or not rest:
            self.err('Usage: pexec -n <nodes> "<command>"')
        from ..core.shell import Shell

        line = " ".join(rest)
        for n in nodes:
            node = self.state.nodes.get(n)
            if node is None or not node.up:
                self.p(f"[{n}]\nssh: connect to host {n} port 22: No route to host")
                continue
            sub = Shell(self.state, n)
            sub.ctx.executed = self.ctx.executed
            self.p(f"[{n}]")
            res = sub.run(line)
            if res:
                self.p(res)
        del cmd

    # ------------------------------------------------ main mode
    def main_versioninfo(self):
        rows = [
            ("Cluster Manager", self.bcm.version),
            ("CMDaemon", "3.0"),
            ("CMDaemon Build Index", "157254"),
            ("CMDaemon Build Hash", "a8d0c4e1b7"),
            ("Database Version", "37573"),
            ("Ubuntu", "22.04"),
            ("Head node", HEAD),
        ]
        self.p("Version Information")
        self.p(_table(["Component", "Value"], rows, [24, 48]))

    def main_licenseinfo(self):
        nodes = len([d for d in self.bcm.devices.values() if d.type == "PhysicalNode"])
        rows = [
            ("Licensee", "/C=US/O=Lab/CN=superpod"),
            ("Edition", "Base Command Manager Advanced"),
            ("Licensed nodes", "64"),
            ("Node count", str(nodes + 1)),
            ("Expiration date", "31/Dec/2027"),
            ("MAC address", "52:54:00:AA:00:FE"),
        ]
        self.p(_table(["Parameter", "Value"], rows, [24, 48]))

    # ------------------------------------------------ end of session
    def finish(self):
        if self.bcm.uncommitted:
            pending = ", ".join(
                f"{v.kind} {v.name} ({v.op}: {', '.join(v.fields) or '-'})" for v in self.bcm.uncommitted.values()
            )
            self.p(self.prompt.rstrip())
            self.p(f"Warning: uncommitted changes: {pending}. Nothing is applied until you run 'commit'.")


@command("cmsh", help='BCM shell: cmsh -c "device; use dgx-01; set category dgx-h100; commit" | cmsh -f file')
def cmsh(ctx, args):
    if ctx.host != HEAD:
        raise CommandError("bash: cmsh: command not found (cmsh runs on the BCM head node)", 127)
    script = None
    it = iter(args)
    for a in it:
        if a in ("-c", "--command"):
            script = next(it, "")
        elif a in ("-f", "--file"):
            path = next(it, "")
            if path not in ctx.node.files:
                raise CommandError(f"cmsh: cannot open {path}: No such file or directory")
            script = ctx.node.files[path]
        elif a in ("-h", "--help"):
            return (
                'Usage: cmsh [-c "cmd; cmd; ..."] [-f file]\n'
                "Modes: device, category, softwareimage, partition, user, main\n"
            )
    if script is None:
        raise CommandError(
            "cmsh: the interactive cmsh shell is not available in this lab.\n"
            'Use one-shot mode, e.g.  cmsh -c "device; list"   or   cmsh -f /root/script.cmsh'
        )
    sh = Cmsh(ctx)
    for line in _split_cmds(script):
        sh.run_line(line)
    sh.finish()
    text = "\n".join(sh.out)
    text = text + "\n" if text else ""
    if sh.errors:
        raise CommandError(text.rstrip("\n"), 1)
    return text


# ---------------------------------------------------------------- chroot into a software image


class _ImgNode(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


class _ImgCtx:
    """A minimal Context whose filesystem is a software image directory."""

    def __init__(self, ctx, files):
        self.state = ctx.state
        self.host = ctx.host
        self.stdin = ctx.stdin
        self.cwd = "/"
        self.executed = ctx.executed
        self.host_stack = []
        self.node = _ImgNode(
            files=files,
            env={},
            up=True,
            gpus=[],
            ib=[],
            services={},
            modules=[],
            kernel=ctx.node.kernel,
            hostname=ctx.host,
        )


def _apt(img, args):
    pkgs = img.packages
    action = next((a for a in args if not a.startswith("-")), "")
    names = [a for a in args if not a.startswith("-")][1:]
    if action in ("update",):
        return (
            "Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease\n"
            "Hit:2 https://repo.download.nvidia.com/baseos/ubuntu/jammy/x86_64 dgx InRelease\n"
            "Reading package lists... Done\n"
        )
    if action == "list":
        return "".join(f"{p}/jammy,now {v} amd64 [installed]\n" for p, v in sorted(pkgs.items()))
    if action in ("install", "upgrade"):
        out = ["Reading package lists... Done", "Building dependency tree... Done"]
        changes = []
        for spec in names:
            name, _, ver = spec.partition("=")
            if name not in REPO:
                raise CommandError("\n".join(out + [f"E: Unable to locate package {name}"]), 100)
            versions = REPO[name]
            ver = ver or versions[-1]
            if ver not in versions:
                raise CommandError("\n".join(out + [f"E: Version '{ver}' for '{name}' was not found"]), 100)
            changes.append((name, pkgs.get(name), ver))
            if name == "nvidia-container-toolkit" and "nvidia-container-toolkit-base" in pkgs:
                changes.append(("nvidia-container-toolkit-base", pkgs.get("nvidia-container-toolkit-base"), ver))
        if not changes or all(o == n for _, o, n in changes):
            return "\n".join(out + ["0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded."]) + "\n"
        up = [c for c in changes if c[1] and c[1] != c[2]]
        new = [c for c in changes if not c[1]]
        if up:
            out += ["The following packages will be upgraded:", "  " + " ".join(c[0] for c in up)]
        if new:
            out += ["The following NEW packages will be installed:", "  " + " ".join(c[0] for c in new)]
        out.append(f"{len(up)} upgraded, {len(new)} newly installed, 0 to remove and 0 not upgraded.")
        for name, old, ver in changes:
            out.append(f"Unpacking {name} ({ver}){' over (' + old + ')' if old else ''} ...")
        for name, _old, ver in changes:
            pkgs[name] = ver
            out.append(f"Setting up {name} ({ver}) ...")
            if name.startswith("linux-image-"):
                kver = name[len("linux-image-") :]
                out.append(f"update-initramfs: Generating /boot/initrd.img-{kver}")
                out.append(
                    "W: Not updating the boot loader inside a chroot; BCM generates the ramdisk "
                    "when the image's kernelversion is set and committed"
                )
        return "\n".join(out) + "\n"
    if action in ("remove", "purge"):
        for n in names:
            pkgs.pop(n, None)
        return f"Removing {' '.join(names)} ...\n"
    raise CommandError(f"E: Invalid operation {action}", 100)


@command(
    "chroot", "cm-chroot-sw-img", help="chroot /cm/images/<image> <command> — run a command inside a BCM software image"
)
def chroot(ctx, args):
    if not args:
        raise CommandError("chroot: missing operand\nUsage: chroot NEWROOT COMMAND [ARG]...", 125)
    path = args[0].rstrip("/")
    bcm = bcm_init(ctx.state)
    img_name = next((n for n, i in bcm.softwareimages.items() if i.path.rstrip("/") == path), None)
    if ctx.host != HEAD or img_name is None:
        raise CommandError(f"chroot: cannot change root directory to '{args[0]}': No such file or directory", 125)
    img = bcm.softwareimages[img_name]
    cmd = args[1:]
    if not cmd:
        raise CommandError(
            "chroot: interactive shells are not supported in the lab; give the command to run, e.g.\n"
            f"  cm-chroot-sw-img {path} apt-get install -y <package>",
            1,
        )
    name = cmd[0]
    if name in ("bash", "sh") and len(cmd) >= 3 and cmd[1] == "-c":
        cmd = shlex.split(cmd[2])
        name = cmd[0]
    if name in ("apt-get", "apt"):
        return _apt(img, cmd[1:])
    if name in ("dpkg", "dpkg-query"):
        pat = next((a for a in cmd[1:] if not a.startswith("-")), "")
        rows = [
            "Desired=Unknown/Install/Remove/Purge/Hold",
            "||/ Name                              Version                       Architecture Description",
            "+++-=================================-=============================-============-===========",
        ]
        hits = [
            f"ii  {p:<33} {v:<29} amd64        "
            for p, v in sorted(img.packages.items())
            if not pat or re.search(pat.replace("*", ".*"), p)
        ]
        if pat and not hits:
            raise CommandError(f"dpkg-query: no packages found matching {pat}", 1)
        return "\n".join(rows + hits) + "\n"
    if name == "uname":
        # classic trap: uname inside a chroot reports the *running* (head node) kernel
        return (ctx.node.kernel if "-r" in cmd else f"Linux {ctx.host} {ctx.node.kernel} x86_64 GNU/Linux") + "\n"
    if name == "ls" and len(cmd) > 1 and cmd[-1].rstrip("/") == "/lib/modules":
        return "\n".join(_kernels(img.packages)) + "\n"
    if name == "nvidia-ctk" and "--version" in cmd:
        v = img.packages.get("nvidia-container-toolkit", "").split("-")[0]
        if not v:
            raise CommandError("bash: nvidia-ctk: command not found", 127)
        return f"NVIDIA Container Toolkit CLI version {v}\ncommit: 4f6dbd4\n"
    if name == "exit":
        return ""
    # generic file tools operate on the image's files
    from . import dispatch

    prefix = img.path.rstrip("/")
    head = ctx.state.nodes[HEAD].files
    before = image_files(ctx.state, img_name)
    view = dict(before)
    sub = _ImgCtx(ctx, view)
    out = dispatch(sub, cmd)
    for p in set(before) - set(view):
        head.pop(prefix + p, None)
    for p, c in view.items():
        if before.get(p) != c:
            head[prefix + p] = c
    return out


@command("cmha", help="cmha status — BCM head node high-availability status")
def cmha(ctx, args):
    if ctx.host != HEAD:
        raise CommandError("bash: cmha: command not found", 127)
    if args[:1] == ["status"] or not args:
        return (
            "Node Status: running in active master mode\n\n"
            "Failover status:\n"
            f"{HEAD}* -> (no passive head node configured)\n"
            "  HA is not configured on this cluster. Run cmha-setup to add a passive head node.\n"
        )
    raise CommandError("usage: cmha status")


# ---------------------------------------------------------------- LDAP-backed identity lookups


def _ldap_user(state, name):
    bcm = bcm_init(state)
    return bcm.users.get(name)


def _groups_of(state, name):
    bcm = bcm_init(state)
    return sorted((g for g in bcm.groups.values() if name in g.get("members", [])), key=lambda g: g.get("gid", 0))


@command("id", help="id [user] — uid, gid and groups (BCM users come from LDAP on every node)")
def id_cmd(ctx, args):
    from .slurm import _USER

    names = [a for a in args if not a.startswith("-")]
    name = names[0] if names else _USER[-1]
    if name == "root":
        return "uid=0(root) gid=0(root) groups=0(root)\n"
    u = _ldap_user(ctx.state, name)
    if u is None:
        raise CommandError(f"id: '{name}': no such user", 1)
    uid = u.get("uid", 1000)
    prim = ctx.state.bcm.groups.get(name, {}).get("gid", uid)
    groups = [f"{prim}({name})"] + [f"{g.get('gid')}({g.name})" for g in _groups_of(ctx.state, name) if g.name != name]
    return f"uid={uid}({name}) gid={prim}({name}) groups={','.join(groups)}\n"


@command("getent", help="getent passwd|group [name] — look up users/groups (LDAP)")
def getent_cmd(ctx, args):
    if not args or args[0] not in ("passwd", "group"):
        raise CommandError("Usage: getent passwd|group [key]", 1)
    bcm = bcm_init(ctx.state)
    keys = args[1:]
    out = []
    if args[0] == "passwd":
        users = [("root", {"uid": 0, "home": "/root", "shell": "/bin/bash"})] + [
            (n, u) for n, u in bcm.users.items() if n != "root"
        ]
        for n, u in users:
            if keys and n not in keys:
                continue
            uid = u.get("uid", 0)
            gid = bcm.groups.get(n, {}).get("gid", uid)
            out.append(
                f"{n}:x:{uid}:{gid}:{u.get('surname', '') or n}:{u.get('home', '/home/' + n)}:{u.get('shell', '/bin/bash')}"
            )
    else:
        for n, g in bcm.groups.items():
            if keys and n not in keys:
                continue
            out.append(f"{n}:x:{g.get('gid', 0)}:{','.join(g.get('members', []))}")
    if keys and not out:
        raise CommandError("", 2)
    return "\n".join(out) + "\n"
