"""BlueField-3 DPU tooling: mlxconfig, rshim, bfb-install, and crictl on the DPU's Arm OS.

Model (set up by :func:`attach_dpu`, normally from a scenario's ``python`` setup op):

  nodes[<host>].dpus = [{
      "name": "bf3-0", "mst": "/dev/mst/mt41692_pciconf0", "pci": "c1:00.0", "rdma": "mlx5_8",
      "netdev": "enp193s0f0np0", "rshim": "rshim0", "arm_host": "dpu-01",
      "fw": "32.38.1002", "bfb": None | "2.7.0",          # DOCA version of the installed BFB bundle
      "cfg":      {PARAM: int},                           # values the firmware runs with now
      "cfg_next": {PARAM: int},                           # values `mlxconfig q` shows (next boot)
  }]
  nodes[<arm_host>]  the DPU's Arm OS as its own ssh-able node (role "dpu"). It is down until a
                     BFB has been installed and the card is in DPU mode. bfb-install re-creates it
                     from scratch (the eMMC is re-imaged, so anything written there before is gone).

  mlxconfig ``s``/``r`` change ``cfg_next``; rebooting the host copies it to ``cfg`` (on real
  hardware a mode change needs a full host power cycle). DPU mode = INTERNAL_CPU_MODEL
  EMBEDDED_CPU(1), the PAGE_SUPPLIER / ESWITCH_MANAGER / IB_VPORT0 owners ECPF(0) and
  INTERNAL_CPU_OFFLOAD_ENGINE ENABLED(0); NIC mode (EXT_HOST_PF, offload engine DISABLED(1))
  leaves the Arm cores without an OS.

  The host's ``rshim`` systemd unit exposes /dev/rshim0/{boot,console,misc}; bfb-install pushes
  the image through /dev/rshim0/boot.

  Files reach the DPU with ``scp`` (storage.py). On the Arm node, a standalone kubelet runs
  every Pod manifest in /etc/kubelet.d (static pods), pulling images from NGC (``DOCA_IMAGES``).
  ``crictl`` there shows those pods; elsewhere it falls through to the Kubernetes-backed crictl
  in container.py.
"""

from __future__ import annotations

import copy
import re

import yaml

from ..core.errors import CommandError
from ..core.state import make_node, wrap
from . import HOOKS_REBOOT, HOOKS_SERVICE, REGISTRY, command
from . import container as _container  # noqa: F401  (registers the generic crictl we wrap below)

KUBELET_DIR = "/etc/kubelet.d/"

# NGC repositories for DOCA service containers (tags follow <service version>-doca<DOCA version>)
DOCA_IMAGES = {
    "nvidia/doca/doca_telemetry": ["1.17.2-doca2.7.0", "1.16.5-doca2.6.0"],
    "nvidia/doca/doca_hbn": ["2.2.0-doca2.7.0", "2.1.0-doca2.6.0"],
    "nvidia/doca/doca_blueman": ["1.0.5-doca2.7.0"],
    "nvidia/doca/doca_firefly": ["1.4.0-doca2.7.0"],
}
IMAGE_SIZE = {"doca_telemetry": "354MB", "doca_hbn": "1.21GB", "doca_blueman": "212MB", "doca_firefly": "143MB"}

# parameter -> {value: name}; order is the order `mlxconfig q` prints them in
PARAMS = {
    "NUM_OF_VFS": {0: "0"},
    "SRIOV_EN": {0: "False", 1: "True"},
    "LINK_TYPE_P1": {1: "IB", 2: "ETH"},
    "LINK_TYPE_P2": {1: "IB", 2: "ETH"},
    "INTERNAL_CPU_MODEL": {0: "SEPARATED_HOST", 1: "EMBEDDED_CPU"},
    "INTERNAL_CPU_PAGE_SUPPLIER": {0: "ECPF", 1: "EXT_HOST_PF"},
    "INTERNAL_CPU_ESWITCH_MANAGER": {0: "ECPF", 1: "EXT_HOST_PF"},
    "INTERNAL_CPU_IB_VPORT0": {0: "ECPF", 1: "EXT_HOST_PF"},
    "INTERNAL_CPU_OFFLOAD_ENGINE": {0: "ENABLED", 1: "DISABLED"},
}
DEFAULTS = {
    "NUM_OF_VFS": 0,
    "SRIOV_EN": 0,
    "LINK_TYPE_P1": 2,
    "LINK_TYPE_P2": 2,
    "INTERNAL_CPU_MODEL": 1,
    "INTERNAL_CPU_PAGE_SUPPLIER": 0,
    "INTERNAL_CPU_ESWITCH_MANAGER": 0,
    "INTERNAL_CPU_IB_VPORT0": 0,
    "INTERNAL_CPU_OFFLOAD_ENGINE": 0,
}
NIC_MODE = {
    "INTERNAL_CPU_PAGE_SUPPLIER": 1,
    "INTERNAL_CPU_ESWITCH_MANAGER": 1,
    "INTERNAL_CPU_IB_VPORT0": 1,
    "INTERNAL_CPU_OFFLOAD_ENGINE": 1,
}
DESCRIPTION = (
    "NVIDIA BlueField-3 P-Series DPU; 400GbE / NDR IB (default mode); Dual-port QSFP112; "
    "PCIe Gen5.0 x16; 16 Arm cores; 32GB on-board DDR; integrated BMC"
)
CX7_PARAMS = {"LINK_TYPE_P1": "IB(1)", "SRIOV_EN": "False(0)", "NUM_OF_VFS": "0", "ADVANCED_PCI_SETTINGS": "False(0)"}


# ---------------------------------------------------------------- setup helper


def attach_dpu(state, host="dgx-01", arm_host="dpu-01", nic_mode=False, bfb=None, rshim="inactive"):
    """Give ``host`` a BlueField-3 and create its Arm OS as node ``arm_host`` (down until a BFB runs)."""
    node = state.nodes[host]
    cfg = dict(DEFAULTS)
    if nic_mode:
        cfg.update(NIC_MODE)
    node.dpus = [
        {
            "name": "bf3-0",
            "mst": "/dev/mst/mt41692_pciconf0",
            "pci": "c1:00.0",
            "rdma": "mlx5_8",
            "netdev": "enp193s0f0np0",
            "numa": 1,
            "rshim": "rshim0",
            "arm_host": arm_host,
            "fw": "32.38.1002",
            "psid": "MT_0000001010",
            "bfb": bfb,
            "cfg": dict(cfg),
            "cfg_next": dict(cfg),
        }
    ]
    node.services["rshim"] = rshim
    node.services_enabled["rshim"] = rshim == "active"
    if rshim == "active":
        _rshim_files(node, node.dpus[0], True)
    state.nodes[arm_host] = _arm_node(arm_host, host, bfb or "2.2.0")
    state.nodes[arm_host].up = bool(bfb) and _dpu_mode(node.dpus[0]["cfg"])
    return node.dpus[0]


def _arm_node(name, parent, doca):
    n = make_node(name, 0)
    n.update(
        {
            "hostname": name,
            "role": "dpu",
            "parent": parent,
            "model": "BlueField-3 DPU (Arm)",
            "os": f"Ubuntu 22.04.4 LTS (DOCA {doca} BlueField bundle)",
            "kernel": "5.15.0-1035-bluefield",
            "ip": "10.141.1.1",
            "bmc_ip": "10.142.1.1",
            "cpus": 16,
            "real_memory_mb": 31000,
            "gpus": [],
            "nvswitches": 0,
            "ib": [],
            "driver_loaded": False,
            "pyxis_installed": False,
            "docker_images": [],
            "services": {
                "kubelet": "active",
                "containerd": "active",
                "openvswitch-switch": "active",
                "ssh": "active",
                "chronyd": "active",
            },
            "modules": ["mlx5_core", "mlx5_ib", "ib_core", "ib_uverbs", "rdma_ucm"],
            "dmesg": [
                f"[    0.000000] Linux version 5.15.0-1035-bluefield (buildd@bos02-arm64-012) DOCA {doca}",
                "[    3.201117] mlx5_core 0000:03:00.0: firmware version: 32.41.1000",
            ],
            "cri_images": [],
            "files": {
                "/etc/hostname": name + "\n",
                "/etc/mlnx-release": f"bf-bundle-{doca}_ubuntu-22.04_prod\n",
            },
        }
    )
    return wrap(n)


DPU_MODE = {k: DEFAULTS[k] for k in ("INTERNAL_CPU_MODEL", *NIC_MODE)}


def _dpu_mode(cfg) -> bool:
    """Embedded CPU owns page supply, eswitch and IB vport0, and the offload engine is enabled."""
    return all(cfg.get(k) == v for k, v in DPU_MODE.items())


def _rshim_files(node, d, present):
    base = f"/dev/{d.rshim}/"
    for f in ("boot", "console", "misc", "rshim"):
        node.files.pop(base + f, None)
    if present:
        node.files[base + "boot"] = ""
        node.files[base + "console"] = ""
        node.files[base + "rshim"] = ""
        node.files[base + "misc"] = (
            "DISPLAY_LEVEL   0 (0:basic, 1:advanced, 2:log)\n"
            "BOOT_MODE       1 (0:rshim, 1:emmc, 2:emmc-boot-swap)\n"
            "BOOT_TIMEOUT    150 (seconds)\n"
            "DROP_MODE       0 (0:normal, 1:drop)\n"
            "SW_RESET        0 (1: reset)\n"
            f"DEV_NAME        pcie-0000:{d.pci[:-1]}2\n"
            "DEV_INFO        BlueField-3(Rev 1)\n"
            "OPN_STR         N/A\n"
            "UP_TIME         5622(s)\n"
        )


# ---------------------------------------------------------------- mlxconfig


def _find_dpu(ctx, dev):
    for d in ctx.node.get("dpus", []):
        if dev in (d.mst, d.pci, "0000:" + d.pci, d.rdma, d.mst.rsplit("/", 1)[-1]):
            if dev.startswith("/dev/mst/") and not ctx.node.get("mst_started"):
                raise CommandError(f"-E- Failed to open device: {dev}. No such file or directory (run 'mst start')")
            return d
    return None


def _val_name(param, v):
    names = PARAMS.get(param, {})
    if param == "NUM_OF_VFS":
        return str(v)
    return f"{names.get(v, '?')}({v})"


def _parse_val(param, raw):
    names = PARAMS[param]
    raw = raw.strip()
    if raw.isdigit():
        v = int(raw)
        if param != "NUM_OF_VFS" and v not in names:
            raise CommandError(f'-E- Invalid value "{raw}" for parameter {param}')
        return v
    for v, n in names.items():
        if n.lower() == raw.lower():
            return v
    raise CommandError(f'-E- Invalid value "{raw}" for parameter {param}')


def _header(d, dev):
    return [
        "",
        "Device #1:",
        "----------",
        "",
        "Device type:        BlueField3",
        "Name:               900-9D3B6-00CN-AB0_Ax",
        f"Description:        {DESCRIPTION}",
        f"Device:             {dev}",
        "",
    ]


@command(
    "mlxconfig",
    help="mlxconfig -d <dev> [-e] q [PARAM] | -y s PARAM=VAL | -y r — NIC/DPU firmware configuration (e.g. DPU vs NIC mode)",
)
def mlxconfig(ctx, args):
    dev, yes, show_all, rest = None, False, False, []
    it = iter(args)
    for a in it:
        if a in ("-d", "--dev"):
            dev = next(it, None)
        elif a in ("-y", "--yes"):
            yes = True
        elif a in ("-e", "--enable_verbosity"):
            show_all = True
        else:
            rest.append(a)
    if not dev:
        raise CommandError("-E- Device not specified. Use: mlxconfig -d <device> q")
    if not rest:
        raise CommandError("-E- No command given. Valid commands: q[uery], s[et], r[eset]")
    cmd, params = rest[0].lower(), rest[1:]
    d = _find_dpu(ctx, dev)
    if d is None:
        return _mlxconfig_cx7(ctx, dev, cmd)
    if cmd in ("q", "query"):
        want = [p.upper() for p in params] or list(PARAMS)
        for p in want:
            if p not in PARAMS:
                raise CommandError(f"-E- The Device doesn't support {p} parameter")
        out = _header(d, dev)
        if show_all:
            out.append(f"{'Configurations:':<52}{'Default':<20}{'Current':<20}Next Boot")
            for p in want:
                star = "*" if d.cfg_next[p] != DEFAULTS[p] or d.cfg_next[p] != d.cfg[p] else " "
                out.append(
                    f"{star}       {p:<44}{_val_name(p, DEFAULTS[p]):<20}{_val_name(p, d.cfg[p]):<20}"
                    f"{_val_name(p, d.cfg_next[p])}"
                )
            out.append("The '*' shows parameters with next value different from default/current value.")
        else:
            out.append(f"{'Configurations:':<52}Next Boot")
            for p in want:
                out.append(f"        {p:<44}{_val_name(p, d.cfg_next[p])}")
        ctx.state.flags.setdefault("mlxconfig_queried", []).append(ctx.host)
        return "\n".join(out) + "\n"
    if cmd in ("s", "set", "r", "reset"):
        if cmd in ("r", "reset"):
            new = dict(DEFAULTS)
            changes = [(p, d.cfg_next[p], v) for p, v in new.items() if d.cfg_next[p] != v]
            prompt = " Reset configuration for device " + dev + "? (y/n) [n] : "
        else:
            if not params:
                raise CommandError("-E- No parameters to set. Use: mlxconfig -d <dev> s PARAM=VALUE")
            new, changes = dict(d.cfg_next), []
            for kv in params:
                if "=" not in kv:
                    raise CommandError(f'-E- Bad format "{kv}", expected PARAM=VALUE')
                p, raw = kv.split("=", 1)
                p = p.upper()
                if p not in PARAMS:
                    raise CommandError(f"-E- The Device doesn't support {p} parameter")
                new[p] = _parse_val(p, raw)
                changes.append((p, d.cfg_next[p], new[p]))
            prompt = " Apply new Configuration? (y/n) [n] : "
        out = _header(d, dev)
        out.append(f"{'Configurations:':<52}{'Next Boot':<20}New")
        for p, old, nv in changes:
            out.append(f"        {p:<44}{_val_name(p, old):<20}{_val_name(p, nv)}")
        out.append("")
        if not yes and not ctx.stdin.strip().lower().startswith("y"):
            out.append(prompt + "n")
            out.append("Aborted by user")
            raise CommandError("\n".join(out), 1)
        out.append(prompt + "y")
        out.append("Applying... Done!")
        out.append("-I- Please reboot machine to load new configurations.")
        d.cfg_next = new
        return "\n".join(out) + "\n"
    raise CommandError(f'-E- Unknown command "{rest[0]}". Valid commands: q[uery], s[et], r[eset]')


def _mlxconfig_cx7(ctx, dev, cmd):
    from .ib import _mlx_dev

    i, _p = _mlx_dev(ctx, dev)
    if cmd not in ("q", "query"):
        raise CommandError("-E- The simulator only supports 'q' on ConnectX-7 HCAs")
    out = [
        "",
        "Device #1:",
        "----------",
        "",
        "Device type:        ConnectX7",
        "Name:               MCX75310AAS-NEA_Ax",
        "Description:        NVIDIA ConnectX-7 HHHL Adapter card; 400Gb/s NDR IB; Single-port OSFP; PCIe 5.0 x16",
        f"Device:             {dev}",
        "",
        f"{'Configurations:':<52}Next Boot",
    ]
    out += [f"        {k:<44}{v}" for k, v in CX7_PARAMS.items()]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- bfb-install


@command("bfb-install", help="bfb-install --bfb <file.bfb> --rshim rshim0 [--config bf.cfg] — flash a BlueField bundle")
def bfb_install(ctx, args):
    bfb = rshim = cfg = None
    it = iter(args)
    for a in it:
        if a in ("-b", "--bfb"):
            bfb = next(it, None)
        elif a in ("-r", "--rshim"):
            rshim = next(it, None)
        elif a in ("-c", "--config"):
            cfg = next(it, None)
    if not bfb or not rshim:
        raise CommandError(
            "Usage: bfb-install [options]\n  -b, --bfb <bfb_file>       BFB image file\n"
            "  -c, --config <config_file> Optional configuration file (bf.cfg)\n"
            "  -r, --rshim <device>       rshim device: rshim<N> or /dev/rshim<N>\n",
            1,
        )
    node = ctx.node
    if bfb not in node.files:
        raise CommandError(f"Error: {bfb} not found", 1)
    if cfg and cfg not in node.files:
        raise CommandError(f"Error: {cfg} not found", 1)
    rname = rshim.removeprefix("/dev/").strip("/")
    d = next((x for x in node.get("dpus", []) if x.rshim == rname), None)
    if d is None or node.services.get("rshim") != "active" or f"/dev/{rname}/boot" not in node.files:
        raise CommandError(f"Error: /dev/{rname}/boot not found (is the rshim service running?)", 1)
    m = re.search(r"bf-bundle-(\d+\.\d+\.\d+)", bfb)
    version = m.group(1) if m else "unknown"
    out = [
        "Pushing bfb",
        f"1.41GiB 0:01:52 [12.8MiB/s] [{'=' * 40}>] 100%",
        "Collecting BlueField booting status. Press Ctrl+C to stop…",
        " INFO[PSC]: PSC BL1 START",
        " INFO[BL2]: start",
        " INFO[BL2]: boot mode (rshim)",
        " INFO[BL2]: DDR POST passed",
        " INFO[BL2]: UEFI loaded",
        " INFO[BL31]: start",
        " INFO[BL31]: runtime",
        " INFO[UEFI]: eMMC init",
        " INFO[UEFI]: eMMC probed",
    ]
    if not _dpu_mode(d.cfg):
        out += [
            " INFO[UEFI]: exit Boot Service",
            "Error: timed out waiting for the BlueField OS installer (no 'Linux up' event).",
            "       The Arm cores are not running an OS: check the DPU operation mode (mlxconfig q).",
        ]
        raise CommandError("\n".join(out), 1)
    out += [
        " INFO[UEFI]: PCIe enum start",
        " INFO[UEFI]: PCIe enum end",
        " INFO[UEFI]: exit Boot Service",
        " INFO[MISC]: Linux up",
        " INFO[MISC]: DPU is ready",
        " INFO[MISC]: Ubuntu installation started",
        " INFO[MISC]: Installing OS image",
        " INFO[MISC]: Updating NIC firmware...",
        " INFO[MISC]: NIC firmware update done",
        " INFO[MISC]: Ubuntu installation completed",
        " INFO[MISC]: Installation finished",
    ]
    d.bfb = version
    d.fw = "32.41.1000"
    arm = _arm_node(d.arm_host, ctx.host, version)
    if cfg:
        arm.files["/root/bf.cfg"] = node.files[cfg]
    ctx.state.nodes[d.arm_host] = arm
    ctx.state.flags.setdefault("bfb_installed", {})[d.arm_host] = version
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- static pods on the DPU


def _norm_image(ref):
    ref = ref.strip()
    name, _, tag = ref.rpartition(":")
    if not name or "/" in tag:
        name, tag = ref, "latest"
    return name, tag


def _ref(image):
    name, tag = _norm_image(image)
    return f"{name}:{tag}"


def _image_ok(ref):
    name, tag = _norm_image(ref)
    repo = name.removeprefix("nvcr.io/")
    return name.startswith("nvcr.io/") and tag in DOCA_IMAGES.get(repo, [])


def static_pods(state, host):
    """Evaluate /etc/kubelet.d on a DPU: [{name, file, containers:[{name,image}], status, error}]."""
    node = state.nodes[host]
    if node.services.get("kubelet") != "active":
        return []
    pods = []
    journal = node.journal.setdefault("kubelet", [])
    for path in sorted(node.files):
        if not path.startswith(KUBELET_DIR) or path.rsplit("/", 1)[-1].startswith("."):
            continue
        if "/" in path[len(KUBELET_DIR) :]:
            continue
        try:
            doc = yaml.safe_load(node.files[path]) or {}
        except yaml.YAMLError as e:
            msg = f'kubelet[1388]: E1019 config.go:212] "Could not process manifest file" err="{str(e).splitlines()[0]}" path="{path}"'
            if msg not in journal:
                journal.append(msg)
            continue
        if not isinstance(doc, dict) or doc.get("kind") != "Pod":
            continue
        meta = doc.get("metadata") or {}
        conts = [
            {"name": c.get("name", "c"), "image": str(c.get("image", ""))}
            for c in ((doc.get("spec") or {}).get("containers") or [])
        ]
        pod = {
            "name": f"{meta.get('name', 'pod')}-{host}",
            "static_name": meta.get("name", "pod"),
            "namespace": meta.get("namespace", "default"),
            "file": path,
            "containers": conts,
            "status": "Running",
            "error": "",
        }
        for c in conts:
            if _image_ok(c["image"]):
                ref = _ref(c["image"])
                if ref not in node.cri_images:
                    node.cri_images.append(ref)
            else:
                pod["status"] = "ImagePullBackOff"
                pod["error"] = (
                    f'Failed to pull image "{c["image"]}": rpc error: code = NotFound desc = failed to pull and '
                    f'unpack image "{c["image"]}": not found'
                )
                msg = f"kubelet[1388]: E1019 pod_workers.go:1298] {pod['error']} pod={pod['name']}"
                if msg not in journal:
                    journal.append(msg)
        pods.append(pod)
    return pods


def _cid(seed, n=13):
    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()[:n]


def _cols(rows):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("   ".join(f"{str(c):<{w}}" for c, w in zip(r, widths)).rstrip() for r in rows) + "\n"


_generic_crictl = REGISTRY.get("crictl")


@command(
    "crictl",
    override=True,
    help="crictl ps [-a] | pods | images | pull <image> | logs <id>  (CRI view of pods on this node)",
)
def crictl(ctx, args):
    node = ctx.node
    if node.get("role") != "dpu":
        if _generic_crictl is None:
            raise CommandError("bash: crictl: command not found", 127)
        return _generic_crictl(ctx, args)
    if node.services.get("containerd") != "active":
        raise CommandError(
            'E1019 10:42:17.000000    2211 remote_runtime.go:277] "ListContainers with filter from runtime service failed" '
            'err="rpc error: code = Unavailable desc = connection error: desc = \\"transport: Error while dialing: '
            'dial unix /run/containerd/containerd.sock: connect: no such file or directory\\""',
            1,
        )
    sub = args[0] if args else "ps"
    pods = static_pods(ctx.state, ctx.host)
    if sub == "pull":
        ref = next((a for a in args[1:] if not a.startswith("-")), None)
        if not ref:
            raise CommandError("crictl pull: image name required", 1)
        if not _image_ok(ref):
            raise CommandError(
                f'E1019 10:42:17.000000    2290 remote_image.go:180] "PullImage from image service failed" '
                f'err="rpc error: code = NotFound desc = failed to pull and unpack image \\"{ref}\\": not found" '
                f'image="{ref}"\nFATA[0002] pulling image: rpc error: code = NotFound desc = not found',
                1,
            )
        full = _ref(ref)
        if full not in node.cri_images:
            node.cri_images.append(full)
        return f"Image is up to date for sha256:{_cid(full, 64)}\n"
    if sub == "images":
        rows = [("IMAGE", "TAG", "IMAGE ID", "SIZE")]
        for ref in node.cri_images:
            name, tag = _norm_image(ref)
            rows.append((name, tag, _cid(ref), IMAGE_SIZE.get(name.rsplit("/", 1)[-1], "300MB")))
        return _cols(rows)
    if sub == "pods":
        rows = [("POD ID", "CREATED", "STATE", "NAME", "NAMESPACE", "ATTEMPT", "RUNTIME")]
        for p in pods:
            rows.append(
                (
                    _cid(p["file"]),
                    "About a minute ago",
                    "Ready" if p["status"] == "Running" else "NotReady",
                    p["name"],
                    p["namespace"],
                    "0",
                    "(default)",
                )
            )
        return _cols(rows)
    if sub == "ps":
        rows = [("CONTAINER", "IMAGE", "CREATED", "STATE", "NAME", "ATTEMPT", "POD ID", "POD")]
        running = []
        for p in pods:
            if p["status"] != "Running":
                continue
            for c in p["containers"]:
                running.append(c["name"])
                rows.append(
                    (
                        _cid(p["file"] + c["name"]),
                        _cid(_ref(c["image"])),
                        "About a minute ago",
                        "Running",
                        c["name"],
                        "0",
                        _cid(p["file"]),
                        p["name"],
                    )
                )
        ctx.state.flags.setdefault("dpu_running", {})[ctx.host] = running
        return _cols(rows)  # -a would add exited containers; this model has none
    if sub == "logs":
        cid = next((a for a in args[1:] if not a.startswith("-")), "")
        for p in pods:
            for c in p["containers"]:
                if cid and _cid(p["file"] + c["name"]).startswith(cid) and p["status"] == "Running":
                    return (
                        f"{c['name']}: DOCA Telemetry Service starting (DOCA {ctx.node.os.split('DOCA ')[-1].split()[0]})\n"
                        "Initializing providers: sysfs, ethtool, hcaperf\n"
                        "Prometheus exporter listening on 0.0.0.0:9100\n"
                        if "telemetry" in c["image"]
                        else f"{c['name']}: started\n"
                    )
        raise CommandError(
            f'rpc error: code = NotFound desc = an error occurred when try to find container "{cid}": not found', 1
        )
    raise CommandError(f"No help topic for '{sub}'", 1)


# ---------------------------------------------------------------- hooks


def _on_service(ctx, name, action):
    node = ctx.node
    if name == "rshim" and node.get("dpus"):
        up = node.services.get("rshim") == "active"
        for d in node.dpus:
            _rshim_files(node, d, up)
        if up and action in ("start", "restart"):
            d = node.dpus[0]
            node.journal.setdefault("rshim", []).extend(
                [
                    f"rshim[2214]: Probing pcie-0000:{d.pci[:-1]}2(vfio)",
                    f"rshim[2214]: create rshim pcie-0000:{d.pci[:-1]}2",
                    f"rshim[2214]: {d.rshim} attached",
                ]
            )


def _on_reboot(ctx, node):
    dpus = node.get("dpus")
    if not dpus:
        return
    node["mst_started"] = False
    if not node.services_enabled.get("rshim", True):
        node.services["rshim"] = "inactive"
    for d in dpus:
        d.cfg = copy.deepcopy(dict(d.cfg_next))
        _rshim_files(node, d, node.services.get("rshim") == "active")
        arm = ctx.state.nodes.get(d.arm_host)
        if arm is not None:
            arm.up = bool(d.bfb) and _dpu_mode(d.cfg)


HOOKS_SERVICE.append(_on_service)
HOOKS_REBOOT.append(_on_reboot)
