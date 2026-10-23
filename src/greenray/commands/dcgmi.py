"""dcgmi (NVIDIA DCGM CLI) emulation: discovery, group, diag, health, stats, dmon, policy.

dcgmi talks to the DCGM host engine (nv-hostengine, systemd unit ``nvidia-dcgm``) on the
current node, or on ``--host <node>``. If the unit is not active every command fails
with "Unable to connect to host engine".

State used (``state.dcgm``, keyed per host because each host engine is independent):
  dcgm.groups[host]          {gid: {"name", "gpus": [idx]}}        (user groups start at 2)
  dcgm.health_watches[host]  {gid: "pmint..."}                     (systems letters)
  dcgm.policies[host]        {gid: {"conditions": [...], "action", "validation"}}
  dcgm.diag_results[host]    {"level", "result": "Pass"/"Fail", "details": [...],
                              "failed_gpus": [...], "tests": {name: result}}
  dcgm.stats_enabled         bool; dcgm.stats_jobs[host] {job id: {...}}
  dcgm.injected[host]        {"<gpu>": {field id: value}}   values put in the host engine's cache by
                                                            `dcgmi test --inject` (the GPU itself is fine,
                                                            so nvidia-smi does not show them)
  dcgm.policy_violations[host]  [{"gid","gpu","condition","value","action","time"}] caught by a
                                registered policy listener (`dcgmi policy --reg`)
Groups, watches, policies, listeners and injected values are lost when the host engine restarts
or the node reboots, exactly like the real (non-persistent) DCGM state. Diag results are cleared on
reboot.
"""

from __future__ import annotations

import time

from ..core import live
from ..core.errors import CommandError
from ..core.state import AttrDict
from . import HOOKS_REBOOT, HOOKS_SERVICE, command
from .common import err as _err
from .common import opt as _opt

DCGM_VERSION = "3.3.6"
SERVICE = "nvidia-dcgm"

# field id -> (short name, unit/label, getter)
FIELDS = {
    100: ("SMCLK", "MHZ", lambda g: g.sm_clock_mhz),
    101: ("MMCLK", "MHZ", lambda g: g.mem_clock_mhz),
    140: ("MMTMP", "C", lambda g: g.mem_temp_c),
    150: ("GPUTMP", "C", lambda g: g.temp_c),
    155: ("POWER", "W", lambda g: f"{g.power_draw_w:.3f}"),
    156: ("TOTEC", "mJ", lambda g: int(g.power_draw_w * 3_600_000)),
    203: ("GPUTL", "%", lambda g: g.utilization),
    204: ("MCUTL", "%", lambda g: 0 if not g.utilization else 41),
    230: ("XIDERRS", "", lambda g: g.xid or 0),
    250: ("FBTTL", "MB", lambda g: g.memory_total_mib),
    251: ("FBFRE", "MB", lambda g: g.memory_total_mib - g.memory_used_mib),
    252: ("FBUSD", "MB", lambda g: g.memory_used_mib),
    310: ("ESVTL", "", lambda g: g.ecc_volatile_sbe),
    311: ("EDVTL", "", lambda g: g.ecc_volatile_dbe),
    312: ("ESATL", "", lambda g: g.ecc_aggregate_sbe),
    313: ("EDATL", "", lambda g: g.ecc_aggregate_dbe),
    393: ("RRUNC", "", lambda g: g.remapped_rows_uncorrectable),
    394: ("RRCOR", "", lambda g: g.remapped_rows_correctable),
    395: ("RRFAIL", "", lambda g: int(bool(g.remap_failure))),
    449: ("NVLTX", "", lambda g: 0),
    1001: ("GRACT", "", lambda g: f"{g.utilization / 100:.3f}"),
    1002: ("SMACT", "", lambda g: f"{g.utilization * 0.92 / 100:.3f}"),
    1003: ("SMOCC", "", lambda g: f"{g.utilization * 0.48 / 100:.3f}"),
    1004: ("TENSO", "", lambda g: f"{g.utilization * 0.61 / 100:.3f}"),
    1005: ("DRAMA", "", lambda g: f"{g.utilization * 0.37 / 100:.3f}"),
}
FIELD_LONG = {
    100: "DCGM_FI_DEV_SM_CLOCK",
    101: "DCGM_FI_DEV_MEM_CLOCK",
    140: "DCGM_FI_DEV_MEMORY_TEMP",
    150: "DCGM_FI_DEV_GPU_TEMP",
    155: "DCGM_FI_DEV_POWER_USAGE",
    156: "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
    203: "DCGM_FI_DEV_GPU_UTIL",
    204: "DCGM_FI_DEV_MEM_COPY_UTIL",
    230: "DCGM_FI_DEV_XID_ERRORS",
    250: "DCGM_FI_DEV_FB_TOTAL",
    251: "DCGM_FI_DEV_FB_FREE",
    252: "DCGM_FI_DEV_FB_USED",
    310: "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL",
    311: "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
    312: "DCGM_FI_DEV_ECC_SBE_AGG_TOTAL",
    313: "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL",
    393: "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS",
    394: "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS",
    395: "DCGM_FI_DEV_ROW_REMAP_FAILURE",
    449: "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL",
    1001: "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
    1002: "DCGM_FI_PROF_SM_ACTIVE",
    1003: "DCGM_FI_PROF_SM_OCCUPANCY",
    1004: "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    1005: "DCGM_FI_PROF_DRAM_ACTIVE",
}

HEALTH_SYSTEMS = {
    "p": "PCIe",
    "m": "Memory",
    "i": "InfoROM",
    "t": "Thermal and Power",
    "n": "NVLink",
    "d": "Driver",
    "s": "SM",
    "v": "NVSwitch",
}
LINE = "+---------------------------+------------------------------------------------+"


def _host(ctx, args):
    h = _opt(args, "--host")
    host = h.split(":")[0] if h else ctx.host
    if host in ("localhost", "127.0.0.1"):
        host = ctx.host
    node = ctx.state.nodes.get(host)
    if node is None or not node.get("up", True) or node.services.get(SERVICE) != "active":
        raise _err(
            f"Error: unable to establish a connection to the specified host: {h or 'localhost'}\n"
            f"Error: Unable to connect to host engine. Host engine connection invalid/disconnected.",
            255,
        )
    return host, node


def _dcgm(state):
    d = state.dcgm
    for k in ("groups", "health_watches", "policies", "diag_results", "stats_jobs", "injected", "policy_violations"):
        d.setdefault(k, {})
    d.setdefault("stats_enabled", False)
    return d


def _host_groups(state, host):
    d = _dcgm(state)
    g = d.groups.get(host)
    if not isinstance(g, dict) or any(not str(k).isdigit() for k in g):
        d.groups[host] = {}
    return d.groups[host]


def _visible(node):
    return [g for g in node.gpus if not g.fallen_off_bus]


# field id -> GPU attribute that `dcgmi test --inject` can override in the host engine's cache
INJECTABLE = {
    150: "temp_c",
    140: "mem_temp_c",
    155: "power_draw_w",
    230: "xid",
    310: "ecc_volatile_sbe",
    311: "ecc_volatile_dbe",
    312: "ecc_aggregate_sbe",
    313: "ecc_aggregate_dbe",
    393: "remapped_rows_uncorrectable",
}


def _view(state, host, g):
    """The GPU as the host engine sees it: hardware values overlaid with injected ones."""
    inj = (state.dcgm.get("injected") or {}).get(host, {}).get(str(g.index))
    if not inj:
        return g
    v = AttrDict(g)
    for fid, val in inj.items():
        attr = INJECTABLE.get(int(fid))
        if attr:
            v[attr] = val
    return v


def _group_gpus(state, host, node, args):
    gid = _opt(args, "-g", "--group")
    if gid is None or str(gid) == "0":
        return [g.index for g in _visible(node)], "0"
    if str(gid) == "1":
        return [], "1"
    groups = _host_groups(state, host)
    if str(gid) not in groups:
        raise _err(f"Error: The Group is not found.\nGroup ID {gid} not found.", 255)
    return list(groups[str(gid)]["gpus"]), str(gid)


def _box(title, rows):
    """rows: list of (left, right) or '-' separators."""
    out = [
        "+-------------------+----------------------------------------------------------+",
        f"| {title:<76} |",
        "+===================+==========================================================+",
    ]
    for r in rows:
        if r == "-":
            out.append("+-------------------+----------------------------------------------------------+")
        else:
            out.append(f"| {r[0]:<17} | {r[1]:<56} |")
    out.append("+-------------------+----------------------------------------------------------+")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- discovery


def _discovery(ctx, args):
    host, node = _host(ctx, args)
    vis = _visible(node)
    out = [
        f"{len(vis)} GPU{'s' if len(vis) != 1 else ''} found.",
        "+--------+----------------------------------------------------------------------+",
        "| GPU ID | Device Information                                                   |",
        "+--------+----------------------------------------------------------------------+",
    ]
    for g in vis:
        out += [
            f"| {g.index:<6} | Name: {g.name:<62} |",
            f"|        | PCI Bus ID: {g.bus_id:<56} |",
            f"|        | Device UUID: {g.uuid:<55} |",
            "+--------+----------------------------------------------------------------------+",
        ]
    ns = node.get("nvswitches", 0)
    out.append(f"{ns} NvSwitch{'es' if ns != 1 else ''} found.")
    if ns:
        out += ["+-----------+", "| Switch ID |", "+-----------+"]
        out += [f"| {i:<9} |" for i in range(ns)]
        out.append("+-----------+")
    out.append(f"{len(node.get('ib', []))} ConnectX found.")
    lost = [g for g in node.gpus if g.fallen_off_bus]
    for g in lost:
        out.append(f"Warning: GPU {g.index} ({g.bus_id}) is not visible to DCGM (inaccessible / fallen off the bus).")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- group


def _group(ctx, args):
    state = ctx.state
    host, node = _host(ctx, args)
    groups = _host_groups(state, host)
    name = _opt(args, "-c", "--create")
    if name is not None:
        gid = str(max([int(k) for k in groups] + [1]) + 1)
        gpus = []
        if "--default" in args:
            gpus = [g.index for g in _visible(node)]
        add = _opt(args, "-a", "--add")
        if add:
            gpus = _parse_entities(add, node)
        groups[gid] = {"name": name, "gpus": gpus}
        return f'Successfully created group "{name}" with a group ID of {gid}\n' + (
            "Specified entity ID is valid\nAdd to group operation successful.\n" if add else ""
        )
    if "-l" in args or "--list" in args:
        rows = [("Groups", "")]
        allg = [
            ("0", "DCGM_ALL_SUPPORTED_GPUS", [g.index for g in _visible(node)], "GPU"),
            ("1", "DCGM_ALL_SUPPORTED_NVSWITCHES", list(range(node.get("nvswitches", 0))), "NvSwitch"),
        ]
        allg += [(k, v["name"], list(v["gpus"]), "GPU") for k, v in sorted(groups.items(), key=lambda kv: int(kv[0]))]
        for gid, nm, ents, kind in allg:
            rows += [
                (f"-> {gid}", ""),
                ("   -> Group ID", gid),
                ("   -> Group Name", nm),
                ("   -> Entities", ", ".join(f"{kind} {e}" for e in ents) or "None"),
            ]
        return _box(f"GROUPS  ({len(allg)} groups found)", rows)
    gid = _opt(args, "-g", "--group")
    if "-d" in args or "--delete" in args:
        d = _opt(args, "-d", "--delete")
        if d not in groups:
            raise _err(f"Error: Cannot destroy group {d}. Return: Not Found", 255)
        del groups[d]
        _dcgm(state).health_watches.get(host, {}).pop(d, None)
        return f"Successfully removed group {d}\n"
    if gid is not None and ("-a" in args or "--add" in args or "-r" in args or "--remove" in args):
        if gid not in groups:
            raise _err(f"Error: The Group is not found.\nGroup ID {gid} not found.", 255)
        add = _opt(args, "-a", "--add")
        rem = _opt(args, "-r", "--remove")
        if add:
            for e in _parse_entities(add, node):
                if e not in groups[gid]["gpus"]:
                    groups[gid]["gpus"].append(e)
            return "Specified entity ID is valid\nAdd to group operation successful.\n"
        for e in _parse_entities(rem, node):
            if e in groups[gid]["gpus"]:
                groups[gid]["gpus"].remove(e)
        return "Remove from group operation successful.\n"
    if gid is not None and ("-i" in args or "--info" in args):
        if gid not in groups:
            raise _err(f"Error: The Group is not found.\nGroup ID {gid} not found.", 255)
        g = groups[gid]
        return _box(
            "GROUP INFO",
            [
                ("Group ID", gid),
                ("Group Name", g["name"]),
                ("Entities", ", ".join(f"GPU {e}" for e in g["gpus"]) or "None"),
            ],
        )
    raise _err(
        "Usage: dcgmi group -c <name> [--default] [-a <gpu ids>] | -l | -g <id> -a|-r <ids> | -d <id> | -g <id> -i", 2
    )


def _parse_entities(spec, node):
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip().lower().removeprefix("gpu:")
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif tok.isdigit():
            out.append(int(tok))
        elif tok:
            raise _err(f"Error: Invalid entity ID specified: {tok}", 2)
    for e in out:
        if e >= len(node.gpus):
            raise _err(f"Error: Invalid GPU ID {e}. Valid IDs are 0-{len(node.gpus) - 1}", 2)
    return out


# ---------------------------------------------------------------- diag

TESTS = [
    # (category, test name, minimum level)
    ("Deployment", "Software", 1),
    ("Integration", "PCIe", 2),
    ("Hardware", "GPU Memory", 2),
    ("Hardware", "Diagnostic", 3),
    ("Stress", "Targeted Stress", 3),
    ("Stress", "Targeted Power", 3),
    ("Stress", "Memory Bandwidth", 3),
    ("Hardware", "Memtest", 4),
    ("Hardware", "Pulse Test", 4),
]
DEPLOY_CHECKS = [
    "Denylist",
    "NVML Library",
    "CUDA Main Library",
    "Permissions and OS Blocks",
    "Persistence Mode",
    "Environment Variables",
    "Page Retirement/Row Remap",
    "Graphics Processes",
    "Inforom",
]


class _Result:
    def __init__(self):
        self.res = "Pass"
        self.msgs = []
        self.gpus = set()

    def fail(self, g, msg, sev="Fail"):
        if self.res != "Fail":
            self.res = sev
        self.msgs.append(msg)
        if g is not None and sev == "Fail":
            self.gpus.add(g)


def _evaluate(node, gpus, level):
    """Return (deploy{check: _Result}, tests{name: _Result}, failed gpu set)."""
    by_idx = {g.index: g for g in node.gpus}
    sel = [by_idx[i] for i in gpus if i in by_idx and not by_idx[i].fallen_off_bus]
    lost = [g for g in node.gpus if g.fallen_off_bus]
    deploy = {c: _Result() for c in DEPLOY_CHECKS}
    tests = {name: _Result() for cat, name, lvl in TESTS if lvl <= level and name != "Software"}
    for g in lost:
        deploy["NVML Library"].fail(
            g.index,
            f"GPU {g.index} ({g.bus_id}) is inaccessible: NVML returned "
            f"GPU_IS_LOST. The GPU has fallen off the bus (XID 79). "
            f"Reboot or power-cycle the node.",
        )
    for g in sel:
        if not g.persistence_mode:
            deploy["Persistence Mode"].fail(
                g.index,
                f"Persistence mode for GPU {g.index} is currently disabled. "
                f"The DCGM diagnostic requires persistence mode to be enabled. "
                f"Enable it with nvidia-smi -i {g.index} -pm 1 or nvidia-persistenced.",
            )
        if g.remap_failure:
            deploy["Page Retirement/Row Remap"].fail(
                g.index,
                f"GPU {g.index} had uncorrectable memory errors and row "
                f"remapping failed. Contact NVIDIA support (RMA).",
            )
        elif g.remap_pending:
            deploy["Page Retirement/Row Remap"].fail(
                g.index,
                f"GPU {g.index} had uncorrectable memory errors and row "
                f"remapping is pending. Drain the GPU and reset it or "
                f"reboot the node to resolve this issue.",
            )
        if g.retired_pages_pending:
            deploy["Page Retirement/Row Remap"].fail(
                g.index, f"GPU {g.index} has retired pages pending. Reset the GPU."
            )
        if g.reset_required and not g.remap_pending:
            deploy["NVML Library"].fail(
                g.index, f"GPU {g.index} requires a reset (nvidia-smi -r -i {g.index}) before it can be used."
            )
        if g.processes:
            deploy["Graphics Processes"].fail(
                g.index,
                f"GPU {g.index} has {len(g.processes)} process(es) running. "
                f"Stop GPU workloads before running the diagnostic.",
                sev="Warn",
            )
    if level >= 2:
        for g in lost:
            tests["PCIe"].fail(g.index, f"GPU {g.index} is inaccessible; PCIe/NVLink tests could not run on it.")
        for g in sel:
            for link in g.nvlink_down:
                tests["PCIe"].fail(
                    g.index,
                    f"GPU {g.index}'s NvLink link {link} is currently down. Check the "
                    f"system logs; if the problem persists after a reset, run the field "
                    f"diagnostic.",
                )
            if g.ecc_volatile_dbe:
                tests["GPU Memory"].fail(
                    g.index, f"Detected {g.ecc_volatile_dbe} volatile double-bit ECC error(s) in GPU {g.index}."
                )
            elif g.xid in (48, 63, 64, 94, 95):
                tests["GPU Memory"].fail(g.index, f"Detected XID {g.xid} for GPU {g.index} (memory error).")
            if g.remap_pending or g.reset_required:
                tests["GPU Memory"].fail(g.index, f"GPU {g.index} requires a reset; the memory test cannot run.")
            if g.xid and g.xid not in (48, 63, 64, 94, 95, 79):
                tests["PCIe"].fail(g.index, f"Detected XID {g.xid} for GPU {g.index}.")
    if level >= 3:
        for g in sel:
            if "HW Thermal Slowdown" in g.throttle_reasons or g.temp_c >= 87:
                tests["Targeted Stress"].fail(
                    g.index,
                    f"Detected clocks event due to thermal violation in GPU "
                    f"{g.index} (HW Thermal Slowdown). Check system cooling.",
                )
                tests["Targeted Power"].fail(
                    g.index, f"GPU {g.index} could not sustain the target power: thermal slowdown active."
                )
            if "HW Power Brake Slowdown" in g.throttle_reasons:
                tests["Targeted Power"].fail(
                    g.index, f"Detected clocks event due to HW Power Brake in GPU {g.index}. Check the power supplies."
                )
            if g.ecc_volatile_dbe or g.remap_pending:
                tests["Diagnostic"].fail(
                    g.index, f"GPU {g.index}: uncorrectable memory errors during the GEMM stress test."
                )
                tests["Memory Bandwidth"].fail(g.index, f"GPU {g.index}: test aborted after an ECC error.")
            if g.nvlink_down:
                tests["Diagnostic"].fail(
                    g.index,
                    f"GPU {g.index}: P2P bandwidth over NVLink below threshold "
                    f"({18 - len(g.nvlink_down)}/18 links active).",
                )
    if level >= 4:
        for g in sel:
            if g.ecc_aggregate_dbe >= 8 or g.remap_failure:
                tests["Memtest"].fail(
                    g.index,
                    f"GPU {g.index}: memtest found bad memory (aggregate DBE "
                    f"{g.ecc_aggregate_dbe}). Contact NVIDIA support.",
                )
    failed = set()
    for r in list(deploy.values()) + list(tests.values()):
        failed |= r.gpus
    return deploy, tests, failed


def _wrap(msg, width=44):
    chunks, cur = [], ""
    for w in msg.split():
        if cur and len(cur) + len(w) + 1 > width:
            chunks.append(cur)
            cur = ""
        cur += ("" if not cur else " ") + w
    chunks.append(cur)
    return chunks


def _diag(ctx, args):
    state = ctx.state
    host, node = _host(ctx, args)
    lvl = _opt(args, "-r", "--run")
    if lvl is None:
        raise _err("Usage: dcgmi diag -r <1|2|3|4|test name> [-g groupId] [-i gpuList] [-j]", 2)
    named = {
        "short": 1,
        "quick": 1,
        "medium": 2,
        "long": 3,
        "xlong": 4,
        "pcie": 2,
        "memory": 2,
        "diagnostic": 3,
        "targeted_stress": 3,
        "targeted_power": 3,
        "memory_bandwidth": 3,
        "memtest": 4,
    }
    if lvl.isdigit():
        level = int(lvl)
        if level not in (1, 2, 3, 4):
            raise _err(f"Error: Invalid run level {lvl}. Valid levels are 1, 2, 3 and 4.", 2)
    elif lvl.lower() in named:
        level = named[lvl.lower()]
    else:
        raise _err(f"Error: Invalid test name or run level: {lvl}", 2)
    gl = _opt(args, "-i", "--gpuList")
    if gl:
        gpus = _parse_entities(gl, node)
    else:
        gpus, _ = _group_gpus(state, host, node, args)
        gpus = (
            sorted(set(gpus) | {g.index for g in node.gpus if g.fallen_off_bus})
            if not _opt(args, "-g", "--group")
            else gpus
        )
    if not gpus:
        raise _err("Error: Diagnostic could not be run: the group has no GPUs.", 255)
    deploy, tests, failed = _evaluate(node, gpus, level)
    any_fail = bool(failed) or any(r.res == "Fail" for r in list(tests.values()) + list(deploy.values()))
    details = [m for r in list(deploy.values()) + list(tests.values()) for m in r.msgs]
    result = "Fail" if any_fail else "Pass"
    _dcgm(state).diag_results[host] = {
        "level": level,
        "result": result,
        "details": details,
        "failed_gpus": sorted(failed),
        "gpus": list(gpus),
        "tests": {k: v.res for k, v in {**deploy, **tests}.items()},
    }
    out = [
        "Successfully ran diagnostic for group.",
        LINE,
        "| Diagnostic                | Result                                         |",
        "+===========================+================================================+",
        "|-----  Metadata  ----------+------------------------------------------------|",
        f"| DCGM Version              | {DCGM_VERSION:<46} |",
        f"| Driver Version Detected   | {node.driver_version:<46} |",
        f"| GPU Device IDs Detected   | {','.join('2330' for g in gpus if not node.gpus[g].fallen_off_bus)[:46]:<46} |",
        "|-----  Deployment  --------+------------------------------------------------|",
    ]

    def detail(res, msgs):
        lines = []
        for m in msgs:
            prefix = "Warning: " if res == "Warn" else "Error: "
            lines += [f"| {'':<25} | {'  ' + c:<46} |" for c in _wrap(prefix + m)]
        return lines

    for c in DEPLOY_CHECKS:
        r = deploy[c]
        out.append(f"| {c:<25} | {r.res:<46} |")
        out += detail(r.res, r.msgs)
    cats = {}
    for cat, name, _lvl in TESTS:
        if name in tests:
            cats.setdefault(cat, []).append(name)
    for cat in ("Integration", "Hardware", "Stress"):
        if cat not in cats:
            continue
        out.append(f"+-----  {cat}  " + "-" * (18 - len(cat)) + "+------------------------------------------------+")
        for name in cats[cat]:
            r = tests[name]
            if r.res == "Pass":
                out.append(f"| {name:<25} | {'Pass - All':<46} |")
                continue
            ok = [i for i in gpus if i not in r.gpus]
            first = True
            if ok:
                out.append(f"| {name:<25} | {'Pass - GPUs: ' + ','.join(map(str, ok)):<46} |")
                first = False
            label = f"{r.res} - GPU{'s' if len(r.gpus) > 1 else ''}: " + ",".join(map(str, sorted(r.gpus)))
            out.append(f"| {name if first else '':<25} | {label:<46} |")
            out += detail(r.res, r.msgs)
    out.append(LINE)
    text = "\n".join(out) + "\n"
    if any_fail:
        raise CommandError(text.rstrip("\n"), 226)
    return text


# ---------------------------------------------------------------- health


def _health(ctx, args):
    state = ctx.state
    host, node = _host(ctx, args)
    d = _dcgm(state)
    watches = d.health_watches.setdefault(host, {})
    gpus, gid = _group_gpus(state, host, node, args)
    systems = _opt(args, "-s", "--set")
    if systems is not None:
        letters = "pmitndsv" if "a" in systems else systems
        bad = [c for c in letters if c not in HEALTH_SYSTEMS]
        if bad:
            raise _err(f"Error: Invalid health system '{bad[0]}'. Valid: a (all), p, m, i, t, n, d, s, v", 2)
        watches[gid] = "".join(sorted(set(letters), key="pmitndsv".index))
        return "Health monitor systems set successfully.\n"
    if "-f" in args or "--fetch" in args:
        w = watches.get(gid, "")
        rows = [(HEALTH_SYSTEMS[c], "On" if c in w else "Off") for c in "pmitnv"]
        return _box("Health Monitor Systems Status", rows)
    if "-c" in args or "--check" in args:
        w = watches.get(gid, "")
        problems = {}

        def add(g, sev, sysname, msg):
            problems.setdefault(g, []).append((sev, sysname, msg))

        by_idx = {g.index: _view(state, host, g) for g in node.gpus}
        targets = [by_idx[i] for i in gpus if i in by_idx] + [
            by_idx[g.index] for g in node.gpus if g.fallen_off_bus and g.index not in gpus
        ]
        for g in targets:
            if "p" in w and g.fallen_off_bus:
                add(g.index, "Failure", "PCIe", f"GPU {g.index} is inaccessible (fallen off the bus, XID 79).")
            if "m" in w:
                if g.ecc_volatile_dbe:
                    add(
                        g.index,
                        "Failure",
                        "Memory",
                        f"Detected {g.ecc_volatile_dbe} volatile double-bit ECC error(s) in GPU {g.index}.",
                    )
                if g.remap_failure:
                    add(g.index, "Failure", "Memory", f"GPU {g.index} had a row remapping failure.")
                elif g.remap_pending:
                    add(
                        g.index,
                        "Failure",
                        "Memory",
                        f"GPU {g.index} had uncorrectable memory errors and row remapping is pending. "
                        f"Drain the GPU and reset it or reboot the node.",
                    )
                if g.ecc_volatile_sbe >= 1000:
                    add(g.index, "Warning", "Memory", f"GPU {g.index} has {g.ecc_volatile_sbe} volatile SBEs.")
            if "n" in w and g.nvlink_down:
                for l in g.nvlink_down:
                    add(
                        g.index,
                        "Failure",
                        "NVLink",
                        f"GPU {g.index}'s NvLink link {l} is currently down. Run a field diagnostic on the GPU.",
                    )
            if "t" in w and ("HW Thermal Slowdown" in g.throttle_reasons or g.temp_c >= 87):
                add(
                    g.index,
                    "Warning",
                    "Thermal",
                    f"Clocks event (HW Thermal Slowdown) detected on GPU {g.index}; temperature {g.temp_c} C.",
                )
            if "t" in w and "HW Power Brake Slowdown" in g.throttle_reasons:
                add(g.index, "Warning", "Power", f"HW Power Brake slowdown detected on GPU {g.index}.")
            if "i" in w and g.get("inforom_corrupt"):
                add(g.index, "Warning", "InfoROM", f"A corrupt InfoROM has been detected in GPU {g.index}.")
            if ("p" in w or "m" in w) and g.xid and g.xid != 79:
                add(g.index, "Warning", "Driver", f"Detected XID {g.xid} for GPU {g.index}.")
        overall = "Healthy"
        if any(sev == "Failure" for ps in problems.values() for sev, _, _ in ps):
            overall = "Failure"
        elif problems:
            overall = "Warning"
        out = [
            "+---------------------------+----------------------------------------------------------+",
            "| Health Monitor Report                                                                |",
            "+===========================+==========================================================+",
            f"| Overall Health            | {overall:<56} |",
        ]
        if not w:
            out.append(f"| {'':<25} | {'(no health watches set for this group: dcgmi health -s)':<56} |")
        for gidx in sorted(problems):
            ps = problems[gidx]
            sev = "Failure" if any(s == "Failure" for s, _, _ in ps) else "Warning"
            out.append("+---------------------------+----------------------------------------------------------+")
            out.append(f"| GPU ID: {gidx:<17} | {sev:<56} |")
            for s_, sysname, msg in ps:
                out.append(f"| {'':<25} | {sysname + ' system: ' + s_:<56} |")
                words, cur, chunks = msg.split(), "", []
                for wd in words:
                    if len(cur) + len(wd) + 1 > 52:
                        chunks.append(cur)
                        cur = ""
                    cur += ("" if not cur else " ") + wd
                chunks.append(cur)
                out += [f"| {'':<25} |   {c:<54} |" for c in chunks]
        out.append("+---------------------------+----------------------------------------------------------+")
        d.setdefault("last_health", {})[host] = {
            "overall": overall,
            "gpus": sorted(problems),
            "watches": w,
            "gid": gid,
            "systems": sorted({sysname for ps in problems.values() for _, sysname, _ in ps}),
        }
        return "\n".join(out) + "\n"
    if "--clear" in args:
        watches.pop(gid, None)
        return "Health monitor systems cleared.\n"
    raise _err("Usage: dcgmi health [-g groupId] -s <a|pmitn...> | -f | -c", 2)


# ---------------------------------------------------------------- stats


def _stats(ctx, args):
    state = ctx.state
    host, node = _host(ctx, args)
    d = _dcgm(state)
    jobs = d.stats_jobs.setdefault(host, {})
    if "-e" in args or "--enable" in args:
        d.stats_enabled = True
        return "Successfully started process watches.\n"
    if "-d" in args or "--disable" in args:
        d.stats_enabled = False
        return "Successfully stopped process watches.\n"
    js = _opt(args, "-s", "--jstart")
    if js:
        if not d.stats_enabled:
            raise _err("Error: Job stats require process watches. Run 'dcgmi stats -e' first.", 255)
        jobs[js] = {"start": True, "stopped": False}
        return f"Successfully started recording stats for {js}.\n"
    je = _opt(args, "-x", "--jstop")
    if je:
        if je not in jobs:
            raise _err(f"Error: Job {je} not found.", 255)
        jobs[je]["stopped"] = True
        return f"Successfully stopped recording stats for {je}.\n"
    jid = _opt(args, "-j", "--job")
    if jid:
        rec = jobs.get(jid)
        sj = None
        for j in state.slurm.get("jobs", []):
            if str(j.get("id")) == jid and host in (j.get("nodelist") or []):
                sj = j
        if rec is None and not (sj is not None and d.stats_enabled):
            raise _err(
                f"Error: Unable to get job statistics. Return: No data is available\n"
                f"(start recording with 'dcgmi stats -g <gid> -s {jid}' or a Slurm prolog)",
                255,
            )
        gpus = (sj.get("alloc_gpus") or {}).get(host, []) if sj else [g.index for g in _visible(node)]
        out = [
            "Successfully retrieved statistics for job: " + jid + ". ",
            "+------------------------------------------------------------------------------+",
            "| Summary                                                                      |",
            "+====================================+=========================================+",
            "|-----  Execution Stats  ------------+-----------------------------------------|",
            "| Start Time                         | Mon Oct 19 09:00:00 2026                |",
            "| End Time                         * | Still Running                           |",
            "| Total Execution Time (sec)       * | Still Running                           |",
            f"| No. of Processes                   | {len(gpus):<39} |",
            "+-----  Performance Stats  ----------+-----------------------------------------+",
            "| Energy Consumed (Joules)           | 1243211                                 |",
            "| Power Usage (Watts)                | Avg: 612.4, Max: 698.1, Min: 71.2       |",
            "| Max GPU Memory Used (bytes)        | 64424509440                             |",
            "| SM Clock (MHz)                     | Avg: 1980, Max: 1980, Min: 345          |",
            "| GPU Utilization (%)                | Avg: 96, Max: 100, Min: 0               |",
            "+-----  Event Stats  ----------------+-----------------------------------------+",
            f"| Single Bit ECC Errors              | {sum(node.gpus[i].ecc_volatile_sbe for i in gpus if i < len(node.gpus)):<39} |",
            f"| Double Bit ECC Errors              | {sum(node.gpus[i].ecc_volatile_dbe for i in gpus if i < len(node.gpus)):<39} |",
            f"| Critical XID Errors                | {sum(1 for i in gpus if i < len(node.gpus) and node.gpus[i].xid):<39} |",
            "+-----  Slowdown Stats  -------------+-----------------------------------------+",
            "| Due to - Power (%)                 | 0                                       |",
            "|        - Thermal (%)               | 0                                       |",
            "+------------------------------------+-----------------------------------------+",
            "(*) Represents a process statistic. Otherwise device statistic.",
        ]
        return "\n".join(out) + "\n"
    raise _err("Usage: dcgmi stats [-g gid] -e | -d | -s <jobid> | -x <jobid> | -j <jobid>", 2)


# ---------------------------------------------------------------- dmon


def _dmon(ctx, args):
    host, node = _host(ctx, args)
    if "-l" in args or "--list" in args:
        out = [
            "___________________________________________________________________________________",
            "                       Long Name          Short Name       Field ID",
            "___________________________________________________________________________________",
        ]
        for fid, (short, _unit, _) in FIELDS.items():
            out.append(f"{FIELD_LONG[fid].lower().removeprefix('dcgm_fi_'):>32} {short:>19} {fid:>14}")
        return "\n".join(out) + "\n"
    spec = _opt(args, "-e", "--field-id")
    if not spec:
        raise _err("Usage: dcgmi dmon -e <fieldIds> [-c count] [-d delay_ms] [-i gpuIds] | dmon -l", 2)
    try:
        fids = [int(x) for x in spec.split(",")]
    except ValueError:
        raise _err(f"Error: Invalid field id list: {spec}", 2)
    for f in fids:
        if f not in FIELDS:
            raise _err(f"Error: Field ID {f} is not a valid field.", 2)
    sel = _opt(args, "-i", "--gpu-id")
    gpus = _visible(node)
    if sel:
        ids = _parse_entities(sel, node)
        gpus = [g for g in gpus if g.index in ids]
    count = int(_opt(args, "-c", "--count", default="1") or 1)
    hdr = "#Entity   " + "".join(f"{FIELDS[f][0]:<12}" for f in fids)
    units = "ID        " + "".join(f"{FIELDS[f][1]:<12}" for f in fids)
    count = max(1, min(count, 3))
    stream = live.Stream(ctx, header=[hdr.rstrip(), units.rstrip()], delay=int(_opt(args, "-d", default="1000")) / 1000)
    for k in range(count):
        rows = []
        for g in gpus:
            vals = []
            for f in fids:
                v = FIELDS[f][2](_view(ctx.state, host, g))
                if f >= 1001 and node.services.get(SERVICE) != "active":
                    v = "N/A"
                vals.append(f"{str(v):<12}")
            rows.append(f"GPU {g.index:<6}" + "".join(vals).rstrip())
        stream.sample(rows, last=k == count - 1)
    return stream.text()


# ---------------------------------------------------------------- policy

COND_FLAGS = {
    "-e": "DBE",
    "--eccerrors": "DBE",
    "-p": "PCIe",
    "--pcierrors": "PCIe",
    "-n": "NVLink",
    "--nvlinkerrors": "NVLink",
    "-x": "XID",
    "--xiderrors": "XID",
}
COND_VAL = {
    "-M": "MaxPages",
    "--maxpages": "MaxPages",
    "-T": "Thermal",
    "--maxtemp": "Thermal",
    "-P": "Power",
    "--maxpower": "Power",
}
COND_LABEL = {
    "DBE": "Double-bit ECC errors",
    "PCIe": "PCI errors and replays",
    "NVLink": "NVLink errors",
    "XID": "XID errors",
    "MaxPages": "Max retired pages / remapped rows",
    "Thermal": "Max temperature",
    "Power": "Max power",
}
ACTIONS = {0: "None", 1: "GPU Reset"}
VALIDATIONS = {
    0: "None",
    1: "System Validation (short)",
    2: "System Validation (medium)",
    3: "System Validation (long)",
}


def _stamp():
    return time.strftime("%a %b %d %H:%M:%S %Y", time.localtime(1792400000 + int(time.time()) % 86400))


def _violations(state, host, node, gid, pol):
    """[(gpu, condition, value, [message lines])] that currently violate ``pol``."""
    if gid == "0" or gid not in _host_groups(state, host):
        gpus = [g.index for g in _visible(node)]
    else:
        gpus = list(_host_groups(state, host)[gid]["gpus"])
    out = []
    by_idx = {g.index: g for g in node.gpus}
    for i in gpus:
        if i not in by_idx or by_idx[i].fallen_off_bus:
            continue
        g = _view(state, host, by_idx[i])
        for cond in pol["conditions"]:
            kind, _, limit = cond.partition(" ")
            if kind == "DBE" and g.ecc_volatile_dbe:
                out.append(
                    (
                        i,
                        "DBE",
                        g.ecc_volatile_dbe,
                        [
                            "A double-bit ECC error has violated policy manager values.",
                            f"DBE error count: {g.ecc_volatile_dbe}",
                        ],
                    )
                )
            elif kind == "XID" and g.xid:
                out.append((i, "XID", g.xid, ["An XID error has occurred.", f"XID: {g.xid}"]))
            elif kind == "NVLink" and g.nvlink_down:
                out.append(
                    (
                        i,
                        "NVLink",
                        len(g.nvlink_down),
                        ["An nvlink error has violated policy manager values.", f"Error count: {len(g.nvlink_down)}"],
                    )
                )
            elif kind == "Thermal" and limit and g.temp_c >= int(limit):
                out.append(
                    (i, "Thermal", g.temp_c, ["A thermal violation has occurred.", f"Temperature reached: {g.temp_c}"])
                )
            elif kind == "Power" and limit and float(g.power_draw_w) >= float(limit):
                out.append(
                    (
                        i,
                        "Power",
                        g.power_draw_w,
                        ["A power violation has occurred.", f"Power reached: {int(g.power_draw_w)}"],
                    )
                )
            elif kind == "MaxPages" and limit and g.remapped_rows_uncorrectable >= int(limit):
                out.append(
                    (
                        i,
                        "MaxPages",
                        g.remapped_rows_uncorrectable,
                        [
                            "Maximum number of retired pages has been violated.",
                            f"Pages retired: {g.remapped_rows_uncorrectable}",
                        ],
                    )
                )
    return out


def _notify(state, host, node):
    """Deliver new violations to registered listeners. Returns the lines a listener would print."""
    d = _dcgm(state)
    seen = d.policy_violations.setdefault(host, [])
    lines = []
    for gid, pol in sorted(d.policies.get(host, {}).items()):
        if not pol.get("registered"):
            continue
        for gpu, cond, value, msg in _violations(state, host, node, gid, pol):
            key = (gid, gpu, cond, value)
            if any((v["gid"], v["gpu"], v["condition"], v["value"]) == key for v in seen):
                continue
            seen.append(
                {"gid": gid, "gpu": gpu, "condition": cond, "value": value, "action": pol["action"], "time": _stamp()}
            )
            lines += [f"Timestamp: {_stamp()}", *msg, f"GPU ID: {gpu}  (group {gid}, action: {pol['action']})", ""]
    return lines


def _policy(ctx, args):
    state = ctx.state
    host, node = _host(ctx, args)
    d = _dcgm(state)
    pols = d.policies.setdefault(host, {})
    gpus, gid = _group_gpus(state, host, node, args)
    if "--get" in args:
        p = pols.get(gid)
        out = ["Policy information"]
        line = "+---------------------------+------------------------------------------------+"
        out += [line, f"| {'GPU group ID: ' + gid:<25} | {'Policy Information':<46} |", line.replace("-", "=")]
        if not p:
            out.append(f"| {'Violation conditions':<25} | {'None':<46} |")
        else:
            labels = []
            for c in p["conditions"]:
                kind, _, limit = c.partition(" ")
                unit = {"Thermal": " C", "Power": " W", "MaxPages": ""}.get(kind, "")
                labels.append(COND_LABEL[kind] + (f" - {limit}{unit}" if limit else ""))
            for k, lab in enumerate(labels):
                out.append(f"| {'Violation conditions' if k == 0 else '':<25} | {lab:<46} |")
            out += [
                f"| {'Isolation mode':<25} | {'Manual':<46} |",
                f"| {'Action on violation':<25} | {p['action']:<46} |",
                f"| {'Validation after action':<25} | {p['validation']:<46} |",
                f"| {'Validation failure action':<25} | {'None':<46} |",
            ]
        out.append(line)
        return "\n".join(out) + "\n"
    if "--clear" in args:
        pols.pop(gid, None)
        return "Policy successfully cleared.\n"
    st = _opt(args, "--set")
    if st is not None:
        try:
            act, val = (int(x) for x in st.split(","))
        except ValueError:
            raise _err("Error: --set expects <action>,<validation> e.g. 0,0", 2)
        if act not in ACTIONS or val not in VALIDATIONS:
            raise _err("Error: invalid --set value. action: 0 (none) | 1 (GPU reset); validation: 0-3", 2)
        conds = []
        for i, a in enumerate(args):
            if a in COND_FLAGS:
                conds.append(COND_FLAGS[a])
            elif a in COND_VAL:
                if i + 1 >= len(args) or not args[i + 1].replace(".", "", 1).isdigit():
                    raise _err(f"Error: {a} requires a numeric threshold", 2)
                conds.append(f"{COND_VAL[a]} {args[i + 1]}")
        if not conds:
            raise _err("Error: specify at least one condition (-e -p -n -x -M N -T C -P W)", 2)
        old = pols.get(gid) or {}
        pols[gid] = {
            "conditions": conds,
            "action": ACTIONS[act],
            "validation": VALIDATIONS[val],
            "registered": bool(old.get("registered")),
        }
        return "Policy successfully set.\n"
    if "--reg" in args:
        p = pols.get(gid)
        if not p:
            raise _err("Error: No policy is set for this group. Use --set first.", 255)
        p["registered"] = True
        out = ["Listening for violations.", ""]
        out += _notify(state, host, node)
        out.append(
            "(simulator: the listener keeps running for this lab session; violations caught later are "
            "printed when they happen and recorded)"
        )
        return "\n".join(out) + "\n"
    raise _err(
        "Usage: dcgmi policy [-g gid] --get | --set <act,val> [-e] [-p] [-n] [-x] [-M n] [-T C] [-P W] | --reg | --clear",
        2,
    )


# ---------------------------------------------------------------- test (error injection)


def _test(ctx, args):
    state = ctx.state
    host, node = _host(ctx, args)
    if "--inject" not in args:
        raise _err("Usage: dcgmi test --inject --gpuid <id> -f <fieldId> -v <value>", 2)
    gpu = _opt(args, "--gpuid", "-i")
    fid = _opt(args, "-f", "--field")
    val = _opt(args, "-v", "--value")
    if gpu is None or fid is None or val is None:
        raise _err("Error: --inject needs --gpuid <id>, -f <fieldId> and -v <value>", 2)
    if not gpu.isdigit() or int(gpu) >= len(node.gpus):
        raise _err(f"Error: Invalid GPU ID {gpu}", 2)
    try:
        fid_i = int(fid)
        value = float(val) if "." in val else int(val)
    except ValueError:
        raise _err(f"Error: invalid field id or value: {fid} {val}", 2)
    if fid_i not in INJECTABLE:
        raise _err(
            f"Error: field {fid_i} cannot be injected in this simulator. Supported: "
            + ", ".join(f"{k} ({FIELD_LONG[k]})" for k in sorted(INJECTABLE)),
            2,
        )
    _dcgm(state).injected.setdefault(host, {}).setdefault(str(int(gpu)), {})[fid_i] = value
    out = ["Successfully injected field info."]
    caught = _notify(state, host, node)
    if caught:
        out += ["", "[policy listener]", *caught]
    return "\n".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------- entry


@command(
    "dcgmi",
    help="DCGM CLI: discovery -l | group -c/-l/-a | diag -r 1-4 | health -s/-c | stats -e/-j | dmon -e | policy",
)
def dcgmi(ctx, args):
    if not args or args[0] in ("-h", "--help"):
        return (
            "Usage: dcgmi <subsystem> [--host <host>]\n  discovery -l          list GPUs/NvSwitches\n"
            "  group -c NAME [--default] | -l | -g ID -a IDS\n  diag -r 1|2|3|4 [-g ID] [-i GPUS]\n"
            "  health -g ID -s a|pmitn | -c | -f\n  stats -e | -s JOB | -x JOB | -j JOB\n"
            "  dmon -e FIELDS [-c N] | -l\n  policy -g ID --set A,V -e -x -T C -P W | --get | --reg\n"
            "  test --inject --gpuid N -f FIELD -v VALUE   inject a value into the host engine cache\n"
        )
    if args[0] in ("-v", "--version"):
        return f"Version : {DCGM_VERSION}\nBuild ID : 34\nBuild Date : 2024-05-28\nBuild Type : Release\n"
    sub, rest = args[0], args[1:]
    handlers = {
        "discovery": _discovery,
        "group": _group,
        "diag": _diag,
        "health": _health,
        "stats": _stats,
        "dmon": _dmon,
        "policy": _policy,
        "test": _test,
    }
    if sub not in handlers:
        raise _err(
            f"dcgmi: unknown subsystem '{sub}'. Try: discovery, group, diag, health, stats, dmon, policy, test", 2
        )
    if sub == "discovery" and not ({"-l", "--list", "-i", "--info", "-c", "--compute"} & set(rest)):
        raise _err("Usage: dcgmi discovery -l", 2)
    return handlers[sub](ctx, rest)


def _forget_host(state, host, diag=False):
    d = _dcgm(state)
    for k in ("groups", "health_watches", "policies", "injected"):
        d[k].pop(host, None)
    if diag:
        d.diag_results.pop(host, None)


def _service_hook(ctx, name, action):
    if name == SERVICE and action in ("restart", "stop", "start"):
        _forget_host(ctx.state, ctx.host)


def _reboot_hook(ctx, node):
    host = node.get("hostname", ctx.host)
    _forget_host(ctx.state, host, diag=True)


HOOKS_SERVICE.append(_service_hook)
HOOKS_REBOOT.append(_reboot_hook)
