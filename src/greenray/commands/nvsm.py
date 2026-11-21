"""``nvsm``, the DGX system management CLI: health, versions, alerts, a support dump, firmware.

Firmware lives in ``node.firmware`` (BMC, BIOS, NVSwitch per switch, ConnectX). The one fault this
models is an inconsistent NVSwitch set after an interrupted update: NVLink traffic across the
mismatched switches is down, which ``nvidia-smi nvlink -s`` and the NCCL tests already show from
``gpu.nvlink_down``; ``nvsm show health`` names the cause. ``nvsm update firmware`` stages the
update and a reboot applies it.
"""

from __future__ import annotations

from ..core import audit
from ..core.errors import CommandError
from . import HOOKS_REBOOT, command
from .common import has, opt
from .ops import FM, fm_version

NVSWITCH_FW = "96.10.73.00.02"
OLD_NVSWITCH_FW = "96.10.42.00.01"
FIRMWARE_TYPES = ("nvswitch", "bmc", "bios", "all")
CURRENT = {"bmc": "24.09.11", "bios": "1.07", "nvswitch": NVSWITCH_FW}
FW_LINKS = list(range(9, 18))  # the NVLinks that cross the two switches left on the old firmware


def firmware(node) -> dict:
    """The node's firmware record, created on first use for nodes made before this module existed."""
    if "firmware" not in node:
        node["firmware"] = {
            "bmc": CURRENT["bmc"],
            "bios": CURRENT["bios"],
            "nvswitch": [NVSWITCH_FW] * node.get("nvswitches", 0),
            "cx7": node.ib[0].fw if node.get("ib") else "",
        }
    node.setdefault("firmware_pending", {})
    return node.firmware


def nvswitch_consistent(node) -> bool:
    return len(set(firmware(node)["nvswitch"])) <= 1


def break_nvswitch_firmware(node, old: str = OLD_NVSWITCH_FW) -> None:
    """Scenario helper: an interrupted bundle left two switches on the old firmware, links across them down."""
    fw = firmware(node)
    fw["nvswitch"] = [NVSWITCH_FW, NVSWITCH_FW, old, old]
    for g in node.gpus:
        g.nvlink_down = sorted(set(g.nvlink_down) | set(FW_LINKS))
    node["nvlink_fw_fault"] = True
    node.dmesg.append(
        "[  212.418007] nvidia-nvswitch2: NVSwitch firmware 96.10.42.00.01 does not match nvidia-nvswitch0 "
        "(96.10.73.00.02); inter-switch links disabled"
    )


def _apply_pending(ctx, node) -> None:
    fw = firmware(node)
    pending = node.get("firmware_pending", {})
    if not pending:
        return
    for kind in list(pending):
        if kind == "nvswitch":
            fw["nvswitch"] = [NVSWITCH_FW] * len(fw["nvswitch"])
        else:
            fw[kind] = CURRENT[kind]
    node.firmware_pending = {}
    if node.get("nvlink_fw_fault") and nvswitch_consistent(node):
        for g in node.gpus:
            g.nvlink_down = [l for l in g.nvlink_down if l not in FW_LINKS]
        del node["nvlink_fw_fault"]
        node.dmesg.append("[    9.102311] nvidia-nvswitch: 4 switches online, firmware 96.10.73.00.02, 72 links active")


HOOKS_REBOOT.append(_apply_pending)


# ---------------------------------------------------------------- reports


def _checks(ctx) -> list[tuple[str, str, str]]:
    node = ctx.node
    fw = firmware(node)
    out = []
    gpus = [g for g in node.gpus if not g.fallen_off_bus]
    out.append(
        (
            "GPU",
            "OK" if len(gpus) == len(node.gpus) and node.driver_loaded else "FAIL",
            f"{len(gpus)} of {len(node.gpus)} GPUs present",
        )
    )
    hot = max((g.temp_c for g in node.gpus), default=0)
    out.append(("GPU temperature", "OK" if hot < 85 else "FAIL", f"max {hot} C"))
    dbe = sum(g.ecc_volatile_dbe for g in node.gpus)
    out.append(("GPU memory ECC", "OK" if not dbe else "FAIL", f"{dbe} uncorrectable errors"))
    if node.nvswitches:
        versions = fw["nvswitch"]
        ok = nvswitch_consistent(node)
        out.append(
            (
                "NVSwitch firmware",
                "OK" if ok else "FAIL",
                f"{versions[0]} on all {len(versions)} switches" if ok else "versions differ: " + ", ".join(versions),
            )
        )
        down = sum(len(g.nvlink_down) for g in node.gpus)
        total = sum(g.nvlink_count for g in node.gpus)
        out.append(("NVLink", "OK" if not down else "FAIL", f"{total - down} of {total} links active"))
        out.append(
            ("Fabric Manager", "OK" if node.services.get(FM) == "active" else "FAIL", node.services.get(FM, "absent"))
        )
    fans = node.get("fans", [{"name": "FAN1", "rpm": 9800, "status": "ok"}])
    bad = [f["name"] for f in fans if f["status"] != "ok"]
    out.append(("Fans", "OK" if not bad else "FAIL", ", ".join(bad) or f"{len(fans)} fans nominal"))
    out.append(("PSUs", "OK", "PSU1-PSU6 present, redundancy OK"))
    out.append(("NVMe", "OK", "RAID 0 /raid, 8 drives, no SMART warnings"))
    out.append(("ConnectX firmware", "OK", f"{fw['cx7']} on {len(node.ib)} adapters"))
    return out


def _show_health(ctx) -> str:
    checks = _checks(ctx)
    bad = [c for c in checks if c[1] != "OK"]
    lines = [
        f"Health Summary: {'Unhealthy' if bad else 'Healthy'}",
        f"Checks: {len(checks) - len(bad)} healthy, {len(bad)} unhealthy",
        "",
    ]
    for name, status, detail in checks:
        lines.append(f"  [{status:>4}] {name:<20} {detail}")
    if bad:
        lines += ["", "Alerts:"] + [f"  {name}: {detail}" for name, _, detail in bad]
    return "\n".join(lines) + "\n"


def _show_versions(ctx) -> str:
    node = ctx.node
    fw = firmware(node)
    rows = [
        ("System", f"{node.model} (DGX OS 6.2)"),
        ("BMC firmware", fw["bmc"]),
        ("BIOS", fw["bios"]),
        ("GPU driver", node.driver_version),
        ("CUDA", node.cuda_version),
        ("Fabric Manager", fm_version(node) if node.nvswitches else "n/a"),
    ]
    for g in node.gpus:
        rows.append((f"GPU{g.index} VBIOS", g.vbios))
    for i, v in enumerate(fw["nvswitch"]):
        rows.append((f"NVSwitch{i} firmware", v))
    for p in node.ib:
        rows.append((f"{p.ca} firmware", p.fw))
    pending = node.get("firmware_pending", {})
    for kind, v in pending.items():
        rows.append((f"{kind} pending", f"{v} (applied at next reboot)"))
    return "\n".join(f"{k:<22} {v}" for k, v in rows) + "\n"


def _show_alerts(ctx) -> str:
    from .cmsh.model import NOW

    bad = [c for c in _checks(ctx) if c[1] != "OK"]
    if not bad:
        return "No active alerts\n"
    lines = [f"{len(bad)} active alert(s)"]
    for i, (name, _, detail) in enumerate(bad, 1):
        lines += [
            f"alert{i}",
            "  severity:  Critical",
            f"  component: {name}",
            f"  message:   {detail}",
            f"  time:      {NOW}",
        ]
    return "\n".join(lines) + "\n"


def _dump_health(ctx) -> str:
    path = f"/tmp/nvsm-health-{ctx.host}-20261019_101513.tar.xz"
    ctx.node.files[path] = f"(xz) nvsm health dump for {ctx.host}: show health, show versions, dmesg, journal, SEL\n"
    return f"Collecting health information... done\nHealth dump written to {path}\n"


def _update_firmware(ctx, args) -> str:
    node = ctx.node
    kind = opt(args, "--type", "-t")
    if kind not in FIRMWARE_TYPES:
        raise CommandError(f"nvsm: --type must be one of {', '.join(FIRMWARE_TYPES)}", 2)
    jobs = audit.running_jobs_on(ctx.state, ctx.host)
    if jobs and not has(args, "--force"):
        raise CommandError(
            f"nvsm: refusing to update firmware while {len(jobs)} job(s) run on {ctx.host} "
            f"({', '.join(str(j.id) for j in jobs)}); drain the node first or pass --force",
            1,
        )
    if jobs:
        audit.record(
            ctx.state, audit.FIRMWARE_WITH_JOBS, ctx.host, "nvsm update firmware", ", ".join(str(j.id) for j in jobs)
        )
    kinds = ["bmc", "bios", "nvswitch"] if kind == "all" else [kind]
    fw = firmware(node)
    lines = []
    for k in kinds:
        if k == "nvswitch" and not node.nvswitches:
            lines.append("nvswitch: no NVSwitch devices on this system")
            continue
        target = NVSWITCH_FW if k == "nvswitch" else CURRENT[k]
        node.firmware_pending[k] = target
        current = ", ".join(fw["nvswitch"]) if k == "nvswitch" else fw[k]
        lines.append(f"{k}: {current} -> {target} staged")
    lines.append("Firmware update staged; it is applied when the system reboots.")
    return "\n".join(lines) + "\n"


@command(
    "nvsm",
    help=(
        "nvsm show health | show versions | show alerts | dump health | update firmware --type <nvswitch|bmc|bios|all> [--force]\n"
        "  --type TYPE   firmware to update: nvswitch, bmc, bios or all\n"
        "  --force       update even while jobs run on the node (not recommended)"
    ),
)
def nvsm(ctx, args):
    node = ctx.node
    if not node.gpus or node.get("role") == "head":
        raise CommandError("bash: nvsm: command not found", 127)
    if node.services.get("nvsm") != "active":
        raise CommandError("nvsm: cannot connect to nvsm daemon; is nvsm.service running?", 1)
    if len(args) >= 2 and args[0] == "show":
        what = args[1]
        if what == "health":
            return _show_health(ctx)
        if what == "versions":
            return _show_versions(ctx)
        if what == "alerts":
            return _show_alerts(ctx)
        if what == "gpus":
            return _show_gpus(ctx)
        if what == "storage":
            return _show_storage(ctx)
        if what == "power":
            return _show_power(ctx)
        if what == "fans":
            return _show_fans(ctx)
        if what == "version":
            return "nvsm 24.09.01\n"
    if len(args) >= 2 and args[0] == "dump" and args[1] == "health":
        return _dump_health(ctx)
    if len(args) >= 2 and args[0] == "update" and args[1] == "firmware":
        return _update_firmware(ctx, args[2:])
    raise CommandError(
        "usage: nvsm show health|versions|alerts|gpus|storage|power|fans|version | dump health | update firmware --type <type>",
        2,
    )


def _show_gpus(ctx) -> str:
    node = ctx.node
    out = ["/systems/localhost/gpus"]
    for g in node.gpus:
        props = {
            "Id": f"GPU{g.index}",
            "Name": g.name,
            "Health": "Critical" if g.fallen_off_bus or g.ecc_volatile_dbe else ("Warning" if g.nvlink_down else "OK"),
            "State": "Absent" if g.fallen_off_bus else "Enabled",
            "PCIeAddress": g.bus_id,
            "UUID": g.uuid,
            "VBIOS": g.vbios,
            "TemperatureCelsius": g.temp_c,
            "PowerDrawWatts": g.power_draw_w,
            "MemoryTotalMiB": g.memory_total_mib,
            "EccUncorrectableErrors": g.ecc_volatile_dbe,
            "NvlinkDownLinks": len(g.nvlink_down),
            "MigMode": "Enabled" if g.mig_mode else "Disabled",
        }
        out.append(f"  GPU{g.index}:")
        out += [f"    {k:<24} = {v}" for k, v in props.items()]
    return "\n".join(out) + "\n"


def _show_storage(ctx) -> str:
    out = ["/systems/localhost/storage", "  Health                    = OK", "  Drives:"]
    for i in range(10):
        size = "1.92 TB" if i < 2 else "3.84 TB"
        role = "OS (RAID 1 /)" if i < 2 else "Data (RAID 0 /raid)"
        out.append(f"    nvme{i}n1   {size}   Health = OK   SMART = passed   Role = {role}")
    out += ["  Volumes:", "    md0   RAID 1   1.7T   /        clean", "    md1   RAID 0   28T    /raid    clean"]
    return "\n".join(out) + "\n"


def _show_power(ctx) -> str:
    node = ctx.node
    gpu_w = sum(g.power_draw_w for g in node.gpus if not g.fallen_off_bus)
    total = 1200 + gpu_w
    out = [
        "/systems/localhost/power",
        f"  TotalPowerDrawWatts       = {total}",
        f"  GpuPowerDrawWatts         = {gpu_w}",
        "  Health                    = OK",
        "  PowerSupplies:",
    ]
    for i in range(1, 7):
        out.append(f"    PSU{i}   Status = OK   Input = AC 230 V   Output = {total // 6} W   Firmware = 2.3")
    out.append("  Redundancy                = Fully redundant (N+2)")
    return "\n".join(out) + "\n"


def _show_fans(ctx) -> str:
    node = ctx.node
    fans = node.get("fans") or [{"name": f"FAN{i}", "rpm": 9800 + 40 * i, "status": "ok"} for i in range(1, 11)]
    out = [
        "/systems/localhost/fans",
        "  Health                    = " + ("OK" if all(f["status"] == "ok" for f in fans) else "Critical"),
    ]
    for f in fans:
        out.append(f"    {f['name']:<8} Speed = {f['rpm']} RPM   Status = {'OK' if f['status'] == 'ok' else 'FAIL'}")
    return "\n".join(out) + "\n"
