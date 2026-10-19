"""nvidia-smi emulation: status table, queries, persistence, power, reset, ECC, MIG, topology."""

from __future__ import annotations

from ..core.errors import CommandError
from . import HOOKS_REBOOT, command

# H100 80GB MIG GPU-instance profiles: name -> (profile id, compute slices, memory slices, max count)
MIG_PROFILES = {
    "1g.10gb": (19, 1, 1, 7),
    "1g.10gb+me": (20, 1, 1, 1),
    "1g.20gb": (15, 1, 2, 4),
    "2g.20gb": (14, 2, 2, 3),
    "3g.40gb": (9, 3, 4, 2),
    "4g.40gb": (5, 4, 4, 1),
    "7g.80gb": (0, 7, 8, 1),
}
PROFILE_BY_ID = {v[0]: k for k, v in MIG_PROFILES.items()}
PROFILE_MEM_MIB = {
    "1g.10gb": 9856,
    "1g.10gb+me": 9856,
    "1g.20gb": 19968,
    "2g.20gb": 19968,
    "3g.40gb": 40192,
    "4g.40gb": 40192,
    "7g.80gb": 80384,
}


def _require_driver(ctx):
    if not ctx.node.gpus:
        raise CommandError(
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
            "Make sure that the latest NVIDIA driver is installed and running.",
            9,
        )
    if not ctx.node.driver_loaded:
        raise CommandError(
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
            "Make sure that the latest NVIDIA driver is installed and running.",
            9,
        )


def _opt(args, *names, default=None):
    for i, a in enumerate(args):
        for n in names:
            if a == n and i + 1 < len(args):
                return args[i + 1]
            if n.startswith("--") and a.startswith(n + "="):
                return a.split("=", 1)[1]
            if not n.startswith("--") and a.startswith(n) and len(a) > len(n) and a[len(n)] != "-":
                return a[len(n) :]
    return default


def _selected(ctx, args):
    sel = _opt(args, "-i", "--id")
    gpus = ctx.node.gpus
    if sel is None:
        return list(gpus)
    out = []
    for s in sel.split(","):
        s = s.strip()
        match = [
            g for g in gpus if str(g.index) == s or g.uuid == s or (":" in s and g.bus_id.lower().endswith(s.lower()))
        ]
        if not match:
            raise CommandError(f"No devices were found matching the id {s}", 6)
        out.extend(match)
    return out


def _visible(gpus):
    return [g for g in gpus if not g.fallen_off_bus]


def _lost_msgs(ctx):
    return [
        f"Unable to determine the device handle for GPU{g.bus_id}: Unknown Error"
        for g in ctx.node.gpus
        if g.fallen_off_bus
    ]


# ---------------------------------------------------------------- table


def _table(ctx):
    n = ctx.node
    lost = _lost_msgs(ctx)
    if lost:
        return "\n".join(lost) + "\n"
    lines = [
        "+-----------------------------------------------------------------------------------------+",
        f"| NVIDIA-SMI {n.driver_version:<22}Driver Version: {n.driver_version:<15}CUDA Version: {n.cuda_version:<9}|",
        "|-----------------------------------------+------------------------+----------------------+",
        "| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |",
        "| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |",
        "|                                         |                        |               MIG M. |",
        "|=========================================+========================+======================|",
    ]
    for g in n.gpus:
        pm = "On" if g.persistence_mode else "Off"
        ecc = "ERR!" if g.reset_required else str(g.ecc_volatile_dbe)
        mig = "Enabled" if g.mig_mode else "Disabled"
        if g.mig_pending is not None:
            mig += "*"
        util = "ERR!" if g.reset_required else f"{g.utilization}%"
        lines += [
            f"| {g.index:>3}  {g.name:<24} {pm:>4} | {g.bus_id:<16} Off | {ecc:>20} |",
            f"| N/A   {g.temp_c:>2}C    {g.pstate:<3} {int(g.power_draw_w):>10}W / {g.power_limit_w:>4}W | "
            f"{g.memory_used_mib:>8}MiB / {g.memory_total_mib}MiB | {util:>8} {g.compute_mode:>11} |",
            f"|                                         |                        | {mig:>20} |",
            "+-----------------------------------------+------------------------+----------------------+",
        ]
    if any(g.mig_mode for g in n.gpus):
        lines += [
            "",
            "+-----------------------------------------------------------------------------------------+",
            "| MIG devices:                                                                            |",
            "| GPU  GI  CI  MIG |              Memory-Usage |  Vol|        Shared                      |",
            "+==================+===========================+=====================================+",
        ]
        for g in n.gpus:
            for ci in g.compute_instances:
                gi = next((x for x in g.gpu_instances if x.id == ci.gi), None)
                prof = gi.profile if gi else ci.profile
                lines.append(
                    f"| {g.index:>3} {ci.gi:>3} {ci.id:>3} {g.compute_instances.index(ci):>3} |"
                    f"   0MiB / {PROFILE_MEM_MIB.get(prof, 0):>5}MiB ({prof:<9}) |                                     |"
                )
        lines.append("+------------------+---------------------------+-------------------------------------+")
    lines += [
        "",
        "+-----------------------------------------------------------------------------------------+",
        "| Processes:                                                                              |",
        "|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |",
        "|=========================================================================================|",
    ]
    procs = [(g, p) for g in n.gpus for p in g.processes]
    if not procs:
        lines.append("|  No running processes found                                                             |")
    for g, p in procs:
        lines.append(f"|  {g.index:>3}   N/A  N/A  {p.pid:>9}      C   {p.name:<40} {p.used_mib:>6}MiB |")
    lines.append("+-----------------------------------------------------------------------------------------+")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- -q


def _query_full(ctx, gpus, sections):
    def want(s):
        return not sections or s in sections

    out = [
        "==============NVSMI LOG==============\n",
        f"Driver Version                            : {ctx.node.driver_version}",
        f"CUDA Version                              : {ctx.node.cuda_version}",
        f"Attached GPUs                             : {len(ctx.node.gpus)}",
    ]
    for g in gpus:
        out.append(f"GPU {g.bus_id}")
        out.append(f"    Product Name                          : {g.name}")
        out.append(f"    Persistence Mode                      : {'Enabled' if g.persistence_mode else 'Disabled'}")
        out.append(f"    Serial Number                         : {g.serial}")
        out.append(f"    GPU UUID                              : {g.uuid}")
        out.append(f"    VBIOS Version                         : {g.vbios}")
        out.append("    MIG Mode")
        out.append(f"        Current                           : {'Enabled' if g.mig_mode else 'Disabled'}")
        pend = g.mig_mode if g.mig_pending is None else g.mig_pending
        out.append(f"        Pending                           : {'Enabled' if pend else 'Disabled'}")
        if want("PERFORMANCE"):
            out.append("    Clocks Event Reasons")
            for r in (
                "GPU Idle",
                "Applications Clocks Setting",
                "SW Power Cap",
                "HW Slowdown",
                "HW Thermal Slowdown",
                "HW Power Brake Slowdown",
                "SW Thermal Slowdown",
                "Sync Boost",
            ):
                active = r in g.throttle_reasons or (r == "GPU Idle" and not g.throttle_reasons and g.utilization == 0)
                out.append(f"        {r:<34}: {'Active' if active else 'Not Active'}")
        if want("MEMORY"):
            out.append("    FB Memory Usage")
            out.append(f"        Total                             : {g.memory_total_mib} MiB")
            out.append(f"        Used                              : {g.memory_used_mib} MiB")
            out.append(f"        Free                              : {g.memory_total_mib - g.memory_used_mib} MiB")
        if want("UTILIZATION"):
            out.append(f"    Utilization\n        Gpu                               : {g.utilization} %")
        if want("ECC"):
            out.append("    ECC Mode")
            out.append(f"        Current                           : {'Enabled' if g.ecc_mode else 'Disabled'}")
            pend = (not g.ecc_mode) if g.ecc_pending else g.ecc_mode
            out.append(f"        Pending                           : {'Enabled' if pend else 'Disabled'}")
            out.append("    ECC Errors\n        Volatile")
            out.append(f"            SRAM Correctable              : {g.ecc_volatile_sbe}")
            out.append(f"            DRAM Uncorrectable            : {g.ecc_volatile_dbe}")
            out.append("        Aggregate")
            out.append(f"            SRAM Correctable              : {g.ecc_aggregate_sbe}")
            out.append(f"            DRAM Uncorrectable            : {g.ecc_aggregate_dbe}")
        if want("ROW_REMAPPER") or want("PAGE_RETIREMENT"):
            out.append("    Remapped Rows")
            out.append(f"        Correctable Error                 : {g.remapped_rows_correctable}")
            out.append(f"        Uncorrectable Error               : {g.remapped_rows_uncorrectable}")
            out.append(f"        Pending                           : {'Yes' if g.remap_pending else 'No'}")
            out.append(f"        Remapping Failure Occurred        : {'Yes' if g.remap_failure else 'No'}")
        if want("TEMPERATURE"):
            out.append("    Temperature")
            out.append(f"        GPU Current Temp                  : {g.temp_c} C")
            out.append("        GPU Shutdown Temp                 : 92 C")
            out.append("        GPU Slowdown Temp                 : 87 C")
            out.append(f"        Memory Current Temp               : {g.mem_temp_c} C")
        if want("POWER"):
            out.append("    GPU Power Readings")
            out.append(f"        Power Draw                        : {g.power_draw_w:.2f} W")
            out.append(f"        Current Power Limit               : {g.power_limit_w:.2f} W")
            out.append(f"        Default Power Limit               : {g.power_default_w:.2f} W")
            out.append(f"        Min Power Limit                   : {g.power_min_w:.2f} W")
            out.append(f"        Max Power Limit                   : {g.power_max_w:.2f} W")
        if want("CLOCK"):
            out.append("    Clocks")
            out.append(f"        SM                                : {g.sm_clock_mhz} MHz")
            out.append(f"        Memory                            : {g.mem_clock_mhz} MHz")
            out.append(f"    Applications Clocks\n        Graphics                          : {g.app_sm_clock_mhz} MHz")
            out.append(f"    Max Clocks\n        SM                                : {g.max_sm_clock_mhz} MHz")
        if want("COMPUTE"):
            out.append(f"    Compute Mode                          : {g.compute_mode}")
    return "\n".join(_lost_msgs(ctx) + out) + "\n"


QUERY_FIELDS = {
    "index": lambda n, g: g.index,
    "name": lambda n, g: g.name,
    "uuid": lambda n, g: g.uuid,
    "serial": lambda n, g: g.serial,
    "pci.bus_id": lambda n, g: g.bus_id,
    "driver_version": lambda n, g: n.driver_version,
    "vbios_version": lambda n, g: g.vbios,
    "persistence_mode": lambda n, g: "Enabled" if g.persistence_mode else "Disabled",
    "pstate": lambda n, g: g.pstate,
    "temperature.gpu": lambda n, g: g.temp_c,
    "temperature.memory": lambda n, g: g.mem_temp_c,
    "utilization.gpu": lambda n, g: (g.utilization, " %"),
    "memory.total": lambda n, g: (g.memory_total_mib, " MiB"),
    "memory.used": lambda n, g: (g.memory_used_mib, " MiB"),
    "memory.free": lambda n, g: (g.memory_total_mib - g.memory_used_mib, " MiB"),
    "power.draw": lambda n, g: (f"{g.power_draw_w:.2f}", " W"),
    "power.limit": lambda n, g: (f"{g.power_limit_w:.2f}", " W"),
    "power.default_limit": lambda n, g: (f"{g.power_default_w:.2f}", " W"),
    "power.max_limit": lambda n, g: (f"{g.power_max_w:.2f}", " W"),
    "clocks.sm": lambda n, g: (g.sm_clock_mhz, " MHz"),
    "clocks.max.sm": lambda n, g: (g.max_sm_clock_mhz, " MHz"),
    "clocks.mem": lambda n, g: (g.mem_clock_mhz, " MHz"),
    "compute_mode": lambda n, g: g.compute_mode,
    "mig.mode.current": lambda n, g: "Enabled" if g.mig_mode else "Disabled",
    "mig.mode.pending": lambda n, g: (
        "Enabled" if (g.mig_mode if g.mig_pending is None else g.mig_pending) else "Disabled"
    ),
    "ecc.mode.current": lambda n, g: "Enabled" if g.ecc_mode else "Disabled",
    "ecc.errors.corrected.volatile.total": lambda n, g: g.ecc_volatile_sbe,
    "ecc.errors.uncorrected.volatile.total": lambda n, g: g.ecc_volatile_dbe,
    "ecc.errors.uncorrected.aggregate.total": lambda n, g: g.ecc_aggregate_dbe,
    "remapped_rows.pending": lambda n, g: "Yes" if g.remap_pending else "No",
    "remapped_rows.failure": lambda n, g: "Yes" if g.remap_failure else "No",
    "clocks_event_reasons.active": lambda n, g: (
        "0x0040"
        if "HW Thermal Slowdown" in g.throttle_reasons
        else ("0x0004" if "SW Power Cap" in g.throttle_reasons else "0x0000")
    ),
    "clocks_throttle_reasons.active": lambda n, g: (
        "0x0040"
        if "HW Thermal Slowdown" in g.throttle_reasons
        else ("0x0004" if "SW Power Cap" in g.throttle_reasons else "0x0000")
    ),
    "clocks_event_reasons.hw_thermal_slowdown": lambda n, g: (
        "Active" if "HW Thermal Slowdown" in g.throttle_reasons else "Not Active"
    ),
    "clocks_throttle_reasons.hw_thermal_slowdown": lambda n, g: (
        "Active" if "HW Thermal Slowdown" in g.throttle_reasons else "Not Active"
    ),
    "clocks_event_reasons.sw_power_cap": lambda n, g: (
        "Active" if "SW Power Cap" in g.throttle_reasons else "Not Active"
    ),
}


def _query_csv(ctx, gpus, fields_spec, fmt):
    fields = [f.strip() for f in fields_spec.split(",") if f.strip()]
    bad = [f for f in fields if f not in QUERY_FIELDS]
    if bad:
        raise CommandError(f'Field "{bad[0]}" is not a valid field to query.', 2)
    fmt = fmt or ""
    noheader, nounits = "noheader" in fmt, "nounits" in fmt
    rows = []
    if not noheader:
        hdr = []
        for f in fields:
            sample = QUERY_FIELDS[f](ctx.node, ctx.node.gpus[0]) if ctx.node.gpus else ""
            unit = f" [{sample[1].strip()}]" if isinstance(sample, tuple) and not nounits else ""
            hdr.append(f + unit)
        rows.append(", ".join(hdr))
    for g in _visible(gpus):
        vals = []
        for f in fields:
            v = QUERY_FIELDS[f](ctx.node, g)
            if isinstance(v, tuple):
                v = f"{v[0]}" if nounits else f"{v[0]}{v[1]}"
            vals.append(str(v))
        rows.append(", ".join(vals))
    return "\n".join(_lost_msgs(ctx) + rows) + "\n"


# ---------------------------------------------------------------- reset / reboot


def _reset(ctx, gpus):
    out = []
    for g in gpus:
        if g.fallen_off_bus:
            raise CommandError(
                f"Unable to reset GPU {g.bus_id}: GPU is lost (fallen off the bus). "
                f"A node reboot / power cycle is required.",
                15,
            )
        if g.processes:
            raise CommandError(
                f"GPU {g.bus_id} is currently in use by another process.\n"
                f"1 device is currently being used by one or more other processes "
                f"(e.g., Fabric Manager, CUDA application, graphics application such as an X server, "
                f"or a monitoring application such as another instance of nvidia-smi). "
                f"Please first kill all processes using this device and all compute applications "
                f"running in the system.",
                255,
            )
    for g in gpus:
        _apply_pending(g)
        g.reset_required = False
        g.xid = None
        g.healthy = True
        g.ecc_volatile_sbe = 0
        g.ecc_volatile_dbe = 0
        g.remap_pending = False
        g.utilization = 0
        out.append(f"GPU {g.bus_id} was successfully reset.")
    return "\n".join(out + ["All done."]) + "\n"


def _apply_pending(g):
    if g.mig_pending is not None:
        g.mig_mode = g.mig_pending
        g.mig_pending = None
        if not g.mig_mode:
            g.gpu_instances = []
            g.compute_instances = []
    if g.ecc_pending:
        g.ecc_mode = not g.ecc_mode
        g.ecc_pending = False


def _on_reboot(ctx, node):
    for g in node.gpus:
        _apply_pending(g)
        g.fallen_off_bus = False
        g.reset_required = False
        g.xid = None
        g.healthy = not g.remap_failure
        g.ecc_volatile_sbe = 0
        g.ecc_volatile_dbe = 0
        g.remap_pending = False
        g.processes = []
        g.memory_used_mib = 0
        g.utilization = 0
        g.gpu_instances = []  # MIG instances are not persistent across reboot
        g.compute_instances = []
    node.driver_loaded = True
    for m in ("nvidia", "nvidia_uvm", "nvidia_modeset"):
        if m not in node.modules:
            node.modules.append(m)
    node.dmesg.append("[    0.000000] Linux version 5.15.0-1042-nvidia (reboot)")


HOOKS_REBOOT.append(_on_reboot)


# ---------------------------------------------------------------- MIG


def _mig(ctx, args):
    gpus = _selected(ctx, args)
    flag = next((a for a in args if a.startswith("-") and a not in ("-i", "--id")), None)
    if flag in ("-lgip", "--list-gpu-instance-profiles"):
        out = [
            "+-----------------------------------------------------------------------------+",
            "| GPU instance profiles:                                                      |",
            "| GPU   Name             ID    Instances   Memory     P2P    SM    DEC   ENC  |",
            "|                              Free/Total   GiB              CE    JPEG  OFA  |",
            "|=============================================================================|",
        ]
        for g in gpus:
            if not g.mig_mode:
                continue
            used_c = sum(MIG_PROFILES[i.profile][1] for i in g.gpu_instances)
            for name, (pid, cs, _ms, mx) in MIG_PROFILES.items():
                total = mx
                free = max(0, min(mx - sum(1 for i in g.gpu_instances if i.profile == name), (7 - used_c) // cs))
                out.append(
                    f"|   {g.index}  MIG {name:<12} {pid:>3}     {free}/{total}        {PROFILE_MEM_MIB[name] / 1024:5.2f}      No     "
                    f"{cs * 16 + (2 if cs == 7 else 0):<4}  |"
                )
        if len(out) == 5:
            raise CommandError(
                "No GPU instance profiles found: MIG mode is not enabled on the selected GPU(s). "
                "Enable it with: nvidia-smi -i <id> -mig 1",
                6,
            )
        return "\n".join(out) + "\n"
    if flag in ("-lgipp",):
        return (
            "".join(
                f"GPU  {g.index} Profile ID {MIG_PROFILES[p][0]} Placements: {{0,1,2,3,4,5,6}}:1\n"
                for g in gpus
                for p in MIG_PROFILES
            )
            + ""
        )
    if flag in ("-cgi", "--create-gpu-instance"):
        spec = _opt(args, "-cgi", "--create-gpu-instance")
        also_ci = "-C" in args or "--default-compute-instance" in args
        out = []
        for g in gpus:
            if not g.mig_mode:
                raise CommandError(
                    f"Unable to create GPU instance on GPU {g.index}: MIG mode is not enabled "
                    f"(Insufficient Permissions / Not Supported).",
                    6,
                )
            for tok in spec.split(","):
                tok = tok.strip()
                name = (
                    PROFILE_BY_ID.get(int(tok))
                    if tok.isdigit()
                    else (tok if tok in MIG_PROFILES else tok.removeprefix("MIG "))
                )
                if name not in MIG_PROFILES:
                    raise CommandError(f"Invalid GPU instance profile: {tok}", 2)
                pid, cs, ms, mx = MIG_PROFILES[name]
                used_c = sum(MIG_PROFILES[i.profile][1] for i in g.gpu_instances)
                used_m = sum(MIG_PROFILES[i.profile][2] for i in g.gpu_instances)
                same = sum(1 for i in g.gpu_instances if i.profile == name)
                if used_c + cs > 7 or used_m + ms > 8 or same >= mx:
                    raise CommandError(
                        f"Unable to create a GPU instance on GPU {g.index} using profile {pid}: Insufficient Resources",
                        6,
                    )
                gid = max([i.id for i in g.gpu_instances] + [0]) + 1
                g.gpu_instances.append({"id": gid, "profile": name, "profile_id": pid, "placement": used_c})
                out.append(
                    f"Successfully created GPU instance ID {gid:>2} on GPU {g.index:>2} using profile MIG {name} (ID {pid:>2})"
                )
                if also_ci:
                    g.compute_instances.append({"gi": gid, "id": 0, "profile": name})
                    out.append(
                        f"Successfully created compute instance ID  0 on GPU {g.index:>2} GPU instance ID {gid:>2} "
                        f"using profile MIG {name} (ID  {cs - 1})"
                    )
        return "\n".join(out) + "\n"
    if flag in ("-lgi", "--list-gpu-instances"):
        out = [
            "+-------------------------------------------------------+",
            "| GPU instances:                                        |",
            "| GPU   Name             Profile  Instance   Placement  |",
            "|                          ID       ID       Start:Size |",
            "|=======================================================|",
        ]
        for g in gpus:
            for i in g.gpu_instances:
                out.append(
                    f"|   {g.index}  MIG {i.profile:<12} {i.profile_id:>3}      {i.id:>3}         {i.placement}:{MIG_PROFILES[i.profile][1]}    |"
                )
        if len(out) == 5:
            raise CommandError("No GPU instances found: Not Found", 6)
        return "\n".join(out) + "\n"
    if flag in ("-cci", "--create-compute-instance"):
        gi_sel = _opt(args, "-gi", "--gpu-instance-id")
        out = []
        for g in gpus:
            for i in g.gpu_instances:
                if gi_sel and str(i.id) not in gi_sel.split(","):
                    continue
                if any(c.gi == i.id for c in g.compute_instances):
                    continue
                g.compute_instances.append({"gi": i.id, "id": 0, "profile": i.profile})
                out.append(
                    f"Successfully created compute instance ID  0 on GPU {g.index:>2} GPU instance ID {i.id:>2} using profile MIG {i.profile}"
                )
        return "\n".join(out) + "\n" if out else "No GPU instances need compute instances.\n"
    if flag in ("-lci", "--list-compute-instances"):
        out = [
            "| GPU     GPU       Name             Profile   Instance   Placement  |",
            "|       Instance                       ID        ID       Start:Size |",
        ]
        for g in gpus:
            for c in g.compute_instances:
                out.append(
                    f"|   {g.index}      {c.gi:>2}       MIG {c.profile:<12}  0         {c.id}         0:{MIG_PROFILES[c.profile][1]}    |"
                )
        if len(out) == 2:
            raise CommandError("No compute instances found: Not Found", 6)
        return "\n".join(out) + "\n"
    if flag in ("-dci", "--destroy-compute-instance"):
        out = []
        for g in gpus:
            if any(p for p in g.processes):
                raise CommandError(f"Unable to destroy compute instance on GPU {g.index}: In use by another client", 19)
            for c in list(g.compute_instances):
                out.append(
                    f"Successfully destroyed compute instance ID  {c.id} from GPU {g.index:>2} GPU instance ID {c.gi:>2}"
                )
            g.compute_instances = []
        return "\n".join(out) + "\n"
    if flag in ("-dgi", "--destroy-gpu-instance"):
        out = []
        for g in gpus:
            if g.compute_instances:
                raise CommandError(
                    f"Unable to destroy GPU instance on GPU {g.index}: In use by another client "
                    f"(destroy compute instances first with -dci)",
                    19,
                )
            for i in list(g.gpu_instances):
                out.append(f"Successfully destroyed GPU instance ID {i.id:>2} from GPU {g.index:>2}")
            g.gpu_instances = []
        return "\n".join(out) + "\n"
    raise CommandError(
        "usage: nvidia-smi mig [-i ID] -lgip | -cgi <profiles> [-C] | -lgi | -cci | -lci | -dci | -dgi", 2
    )


# ---------------------------------------------------------------- main entry


@command("nvidia-smi", help="GPU status/config: -q, -L, --query-gpu, -pm, -pl, -r, -mig, mig, topo -m, nvlink -s")
def nvidia_smi(ctx, args):
    _require_driver(ctx)
    if args and args[0] == "mig":
        return _mig(ctx, args[1:])
    if args and args[0] == "topo":
        return _topo(ctx)
    if args and args[0] == "nvlink":
        return _nvlink(ctx, args[1:])
    if args and args[0] == "dmon":
        return _dmon(ctx, args[1:])
    if not args:
        return _table(ctx)
    if "-h" in args or "--help" in args:
        return (
            "Usage: nvidia-smi [OPTION]...\n  -L  list GPUs\n  -q [-d SECTION]  query (MEMORY,ECC,TEMPERATURE,POWER,CLOCK,PERFORMANCE,ROW_REMAPPER,COMPUTE)\n"
            "  --query-gpu=FIELDS --format=csv[,noheader,nounits]\n  -i ID  select GPU\n  -pm 0|1  persistence mode\n"
            "  -pl WATTS  power limit\n  -r  reset GPU\n  -e 0|1  ECC mode\n  -c MODE  compute mode\n  -mig 0|1  MIG mode\n"
            "  -lgc MIN,MAX / -rgc  lock/reset GPU clocks\n  mig ...  MIG instance management\n  topo -m  topology\n  nvlink -s  NVLink status\n"
        )
    gpus = _selected(ctx, args)
    if "-L" in args or "--list-gpus" in args:
        out = []
        for g in _visible(gpus):
            out.append(f"GPU {g.index}: {g.name} (UUID: {g.uuid})")
            for gi in g.gpu_instances:
                out.append(
                    f"  MIG {gi.profile:<11} Device  {g.gpu_instances.index(gi)}: (UUID: MIG-{g.uuid[4:12]}-{gi.id:04d})"
                )
        return "\n".join(_lost_msgs(ctx) + out) + "\n"
    q = _opt(args, "--query-gpu")
    if q:
        return _query_csv(ctx, gpus, q, _opt(args, "--format"))
    if "-q" in args or "--query" in args:
        d = _opt(args, "-d", "--display")
        return _query_full(ctx, _visible(gpus), set(d.upper().split(",")) if d else set())
    pm = _opt(args, "-pm", "--persistence-mode")
    if pm is not None:
        on = pm in ("1", "ENABLED", "Enabled")
        out = []
        for g in _visible(gpus):
            g.persistence_mode = on
            out.append(f"{'Enabled' if on else 'Disabled'} persistence mode for GPU {g.bus_id}.")
        return "\n".join(out + ["All done."]) + "\n"
    pl = _opt(args, "-pl", "--power-limit")
    if pl is not None:
        watts = float(pl)
        out = []
        for g in _visible(gpus):
            if not (g.power_min_w <= watts <= g.power_max_w):
                raise CommandError(
                    f"Provided power limit {watts:.2f} W is not a valid power limit which should be "
                    f"between {g.power_min_w:.2f} W and {g.power_max_w:.2f} W for GPU {g.bus_id}",
                    2,
                )
            g.power_limit_w = int(watts)
            if watts >= g.power_default_w and "SW Power Cap" in g.throttle_reasons:
                g.throttle_reasons.remove("SW Power Cap")
            out.append(f"Power limit for GPU {g.bus_id} was set to {watts:.2f} W from {g.power_default_w:.2f} W.")
        return "\n".join(out + ["All done."]) + "\n"
    ecc = _opt(args, "-e", "--ecc-config")
    if ecc is not None:
        on = ecc == "1"
        out = []
        for g in _visible(gpus):
            g.ecc_pending = on != g.ecc_mode
            out.append(f"{'Enabled' if on else 'Disabled'} ECC support for GPU {g.bus_id}.")
        return "\n".join(out + ["All done.", "Reboot required."]) + "\n"
    cm = _opt(args, "-c", "--compute-mode")
    if cm is not None:
        modes = {
            "0": "Default",
            "DEFAULT": "Default",
            "2": "Prohibited",
            "PROHIBITED": "Prohibited",
            "3": "Exclusive_Process",
            "EXCLUSIVE_PROCESS": "Exclusive_Process",
        }
        mode = modes.get(cm.upper())
        if not mode:
            raise CommandError(f"Invalid compute mode: {cm}", 2)
        for g in _visible(gpus):
            g.compute_mode = mode
        return (
            "".join(f"Set compute mode to {mode.upper()} for GPU {g.bus_id}.\n" for g in _visible(gpus)) + "All done.\n"
        )
    mig = _opt(args, "-mig", "--multi-instance-gpu")
    if mig is not None:
        on = mig == "1"
        out = []
        for g in _visible(gpus):
            if not g.mig_capable:
                raise CommandError(f"Unable to enable MIG Mode for GPU {g.bus_id}: Not Supported", 3)
            if not on and (g.gpu_instances or g.compute_instances):
                raise CommandError(
                    f"Unable to disable MIG Mode for GPU {g.bus_id}: In use by another client. "
                    f"Destroy all compute and GPU instances first.",
                    19,
                )
            if g.processes:
                g.mig_pending = on
                out.append(
                    f"Warning: MIG mode is in pending {'enable' if on else 'disable'} state for GPU {g.bus_id}:In use by another client\n"
                    f"{g.bus_id} is currently being used by one or more other processes (e.g. CUDA application or a monitoring application "
                    f"such as another instance of nvidia-smi). Please first kill all processes using the device and retry the command "
                    f"or reboot the system to make MIG mode effective."
                )
            else:
                g.mig_mode = on
                g.mig_pending = None
                if not on:
                    g.gpu_instances, g.compute_instances = [], []
                out.append(f"{'Enabled' if on else 'Disabled'} MIG Mode for GPU {g.bus_id}")
        return "\n".join(out + ["All done."]) + "\n"
    if "-r" in args or "--gpu-reset" in args:
        if "-i" not in args and "--id" not in args and len(gpus) > 1:
            pass  # resetting all GPUs is allowed when nothing is using them
        return _reset(ctx, gpus)
    lgc = _opt(args, "-lgc", "--lock-gpu-clocks")
    if lgc is not None:
        lo, _, hi = lgc.partition(",")
        for g in _visible(gpus):
            g.sm_clock_mhz = int(hi or lo)
        return (
            "".join(
                f'GPU clocks set to "(gpuClkMin {lo}, gpuClkMax {hi or lo})" for GPU {g.bus_id}\n'
                for g in _visible(gpus)
            )
            + "All done.\n"
        )
    if "-rgc" in args or "--reset-gpu-clocks" in args:
        return "".join(f"All done. GPU clocks reset for GPU {g.bus_id}\n" for g in _visible(gpus))
    if "-i" in args or "--id" in args:
        return _table(ctx)
    raise CommandError("Invalid combination of input arguments. Please run 'nvidia-smi -h' for help.", 2)


def _topo(ctx):
    gpus = ctx.node.gpus
    hdr = (
        "\t"
        + "\t".join(f"GPU{g.index}" for g in gpus)
        + "\t"
        + "\t".join(f"NIC{i}" for i in range(len(ctx.node.ib)))
        + "\tCPU Affinity\tNUMA Affinity"
    )
    rows = [hdr]
    for g in gpus:
        cells = [
            "X"
            if h.index == g.index
            else ("NV18" if not (g.nvlink_down or h.nvlink_down) else "NV" + str(18 - len(g.nvlink_down)))
            for h in gpus
        ]
        nic = ["PXB" if i == g.index else "SYS" for i in range(len(ctx.node.ib))]
        numa = 0 if g.index < 4 else 1
        rows.append(
            f"GPU{g.index}\t"
            + "\t".join(cells)
            + "\t"
            + "\t".join(nic)
            + f"\t{numa * 56}-{numa * 56 + 55},{112 + numa * 56}-{112 + numa * 56 + 55}\t{numa}"
        )
    rows += [
        "",
        "Legend:",
        "  X    = Self",
        "  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes",
        "  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)",
        "  NV#  = Connection traversing a bonded set of # NVLinks",
    ]
    return "\n".join(rows) + "\n"


def _nvlink(ctx, args):
    gpus = _selected(ctx, args)
    out = []
    for g in gpus:
        out.append(f"GPU {g.index}: {g.name} (UUID: {g.uuid})")
        for l in range(g.nvlink_count):
            out.append(f"\t Link {l}: " + ("<inactive>" if l in g.nvlink_down else "26.562 GB/s"))
    return "\n".join(out) + "\n"


def _dmon(ctx, args=()):
    """One sample per GPU by default; ``-c N`` prints N samples (1 s apart on real hardware).

    If a training job with a modelled input pipeline runs on the node (see gds.py), SM%
    follows that pipeline: a starved job shows bursts of ~97% separated by near-idle samples.
    """
    from . import gds

    args = list(args)
    count = 1
    if "-c" in args:
        i = args.index("-c")
        try:
            count = max(1, min(int(args[i + 1]), 30))
        except (IndexError, ValueError):
            raise CommandError("Invalid count value. Must be a positive integer.", 2)
    sel = _opt(args, "-i", "--id")
    gpus = _visible(ctx.node.gpus)
    if sel is not None:
        ids = {s.strip() for s in str(sel).split(",")}
        gpus = [g for g in gpus if str(g.index) in ids]
    job = gds.running_job(ctx.node)
    pattern = gds.util_samples(job["ratio"], count) if job else None
    out = [
        "# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec   mclk   pclk",
        "# Idx      W      C      C      %      %      %      %    MHz    MHz",
    ]
    for k in range(count):
        for g in gpus:
            sm = g.utilization if pattern is None else pattern[k]  # DDP ranks stall together
            mem = 0 if pattern is None else int(sm * 0.42)
            pwr = int(g.power_draw_w) if pattern is None else int(120 + sm * 5.6)
            pclk = g.sm_clock_mhz if pattern is None else (1980 if sm > 50 else 1755)
            out.append(
                f"    {g.index:>1}  {pwr:>5}  {g.temp_c:>5}  {g.mem_temp_c:>5}  {sm:>5}  {mem:>5}      0      0  "
                f"{g.mem_clock_mhz:>5}  {pclk:>5}"
            )
    return "\n".join(out) + "\n"
