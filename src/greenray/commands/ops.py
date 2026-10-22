"""Operations helpers: CUDA smoke test, Fabric Manager, support bundle, and lab-only
stand-ins for physical / out-of-band actions.

Real tools emulated here:
    deviceQuery            CUDA sample; fails with error 802 when Fabric Manager is down
    nv-fabricmanager       --version
    nvidia-bug-report.sh   collects /root/nvidia-bug-report.log.gz for NVIDIA Enterprise Support

Lab helpers (clearly NOT real commands; they stand in for things a person does):
    reseat-cable <node> <mlx5_N>                 a technician physically reseats an IB cable/transceiver
    hwticket create --node N --component C --summary "..."   opens a hardware RMA/support case
    hwticket list | show <id>
"""

from __future__ import annotations

from ..core.errors import CommandError
from . import HOOKS_REBOOT, HOOKS_SERVICE, command

FM = "nvidia-fabricmanager"


# ---------------------------------------------------------------- Fabric Manager


def fm_version(node):
    return node.get("fm_version") or node.driver_version


def _on_service(ctx, name, action):
    node = ctx.node
    if name != FM or action not in ("start", "restart"):
        return
    j = node.journal.setdefault(FM, [])
    j.append(f"Oct 19 10:20:01 {ctx.host} systemd[1]: Starting NVIDIA fabric manager service...")
    if not node.nvswitches:
        node.services[FM] = "failed"
        j.append(
            f"Oct 19 10:20:01 {ctx.host} nv-fabricmanager[5120]: request to query NVSwitch device information "
            "from NVSwitch driver failed with error:WARNING Nothing to do [NV_WARN_NOTHING_TO_DO]"
        )
        j.append(f"Oct 19 10:20:01 {ctx.host} systemd[1]: {FM}.service: Failed with result 'exit-code'.")
        return
    if not node.driver_loaded:
        node.services[FM] = "failed"
        j.append(
            f"Oct 19 10:20:01 {ctx.host} nv-fabricmanager[5120]: failed to open NVIDIA GPU driver interface; "
            "is the NVIDIA kernel module loaded?"
        )
        j.append(f"Oct 19 10:20:01 {ctx.host} systemd[1]: {FM}.service: Failed with result 'exit-code'.")
        return
    if fm_version(node) != node.driver_version:
        node.services[FM] = "failed"
        j.append(
            f"Oct 19 10:20:01 {ctx.host} nv-fabricmanager[5120]: fabric manager NVIDIA GPU driver interface version "
            f"{fm_version(node)} don't match with driver version {node.driver_version}. "
            "Please update with matching NVIDIA driver package."
        )
        j.append(
            f"Oct 19 10:20:01 {ctx.host} systemd[1]: {FM}.service: Control process exited, code=exited, status=1/FAILURE"
        )
        j.append(f"Oct 19 10:20:01 {ctx.host} systemd[1]: {FM}.service: Failed with result 'exit-code'.")
        j.append(f"Oct 19 10:20:01 {ctx.host} systemd[1]: Failed to start NVIDIA fabric manager service.")
        return
    j.append(f"Oct 19 10:20:02 {ctx.host} nv-fabricmanager[5120]: Connected to 1 node.")
    j.append(
        f"Oct 19 10:20:04 {ctx.host} nv-fabricmanager[5120]: Successfully configured all the available NVSwitches "
        "to route GPU NVLink traffic. NVLink Peer-to-Peer support will be enabled once the GPUs are successfully "
        "registered with the NVLink fabric."
    )
    j.append(f"Oct 19 10:20:04 {ctx.host} systemd[1]: Started NVIDIA fabric manager service.")


def _on_reboot(ctx, node):
    # a disabled unit does not come back after a reboot
    if FM in node.services and node.services_enabled.get(FM) is False:
        node.services[FM] = "inactive"


HOOKS_SERVICE.append(_on_service)
HOOKS_REBOOT.append(_on_reboot)


@command("nv-fabricmanager", help="nv-fabricmanager --version")
def nv_fabricmanager(ctx, args):
    if not ctx.node.nvswitches:
        raise CommandError("bash: nv-fabricmanager: command not found", 127)
    if "-v" in args or "--version" in args:
        return f"Fabric Manager version is : {fm_version(ctx.node)}\n"
    raise CommandError(
        "nv-fabricmanager runs as a service: use systemctl start nvidia-fabricmanager "
        "(config: /usr/share/nvidia/nvswitch/fabricmanager.cfg)"
    )


# ---------------------------------------------------------------- CUDA smoke test


@command(
    "deviceQuery",
    "./deviceQuery",
    "/usr/local/cuda/extras/demo_suite/deviceQuery",
    help="CUDA deviceQuery sample — quick check that CUDA can use the GPUs",
)
def device_query(ctx, args):
    node = ctx.node
    head = [
        f"{'./deviceQuery'} Starting...",
        "",
        " CUDA Device Query (Runtime API) version (CUDART static linking)",
        "",
    ]
    results = ctx.state.flags.setdefault("last_cuda", {})

    def fail(code, msg):
        results[ctx.host] = "FAIL"
        raise CommandError("\n".join(head + [f"cudaGetDeviceCount returned {code}", f"-> {msg}", "Result = FAIL"]), 1)

    if not node.gpus or not node.driver_loaded:
        fail(100, "no CUDA-capable device is detected")
    if node.nvswitches and node.services.get(FM) != "active":
        fail(802, "system not yet initialized")
    gpus = [g for g in node.gpus if not g.fallen_off_bus]
    out = head + [f"Detected {len(gpus)} CUDA Capable device(s)", ""]
    for g in gpus[:1]:
        out += [
            f'Device {g.index}: "{g.name}"',
            f"  CUDA Driver Version / Runtime Version          {node.cuda_version} / {node.cuda_version}",
            "  CUDA Capability Major/Minor version number:    9.0",
            f"  Total amount of global memory:                 {g.memory_total_mib} MBytes",
            "  (132) Multiprocessors, (128) CUDA Cores/MP:    16896 CUDA Cores",
            f"  GPU Max Clock rate:                            {g.max_sm_clock_mhz} MHz",
            "  Memory Bus Width:                              5120-bit",
            f"  Device PCI Domain ID / Bus ID / location ID:   0 / {int(g.bus_id[9:11], 16)} / 0",
            "",
            f"  ... ({len(gpus) - 1} more devices)",
            "",
        ]
    if len(gpus) > 1:
        out.append("> Peer access from NVIDIA H100 80GB HBM3 (GPU0) -> NVIDIA H100 80GB HBM3 (GPU1) : Yes")
    out += [
        f"deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = {node.cuda_version}, "
        f"CUDA Runtime Version = {node.cuda_version}, NumDevs = {len(gpus)}",
        "Result = PASS",
    ]
    results[ctx.host] = "PASS"
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- support bundle


@command("nvidia-bug-report.sh", help="collect nvidia-bug-report.log.gz for NVIDIA support")
def bug_report(ctx, args):
    if not ctx.node.gpus:
        raise CommandError("nvidia-bug-report.sh: no NVIDIA GPUs found on this host")
    path = "/root/nvidia-bug-report.log.gz"
    ctx.node.files[path] = f"(gzip) nvidia-bug-report for {ctx.host}: nvidia-smi -q, dmesg, SEL, journal...\n"
    ctx.state.flags.setdefault("bug_reports", []).append(ctx.host)
    return (
        "\nnvidia-bug-report.sh will now collect information about your\nsystem and create the file "
        "'nvidia-bug-report.log.gz' in the current\ndirectory.  It may take several seconds to run.\n\n"
        "Running nvidia-bug-report.sh... complete.\n\n"
        f"The file {path} has been created; please send this file, along\n"
        "with a description of your problem, to NVIDIA Enterprise Support.\n"
    )


# ---------------------------------------------------------------- lab helpers

LAB = "[LAB HELPER — stands in for a physical/out-of-band action; not a real Linux command]"


@command("reseat-cable", help="LAB HELPER: reseat-cable <node> <mlx5_N> — technician reseats the IB cable/transceiver")
def reseat_cable(ctx, args):
    if len(args) != 2:
        raise CommandError("usage: reseat-cable <node> <mlx5_N>   " + LAB)
    host, ca = args
    node = ctx.state.nodes.get(host)
    if node is None:
        raise CommandError(f"reseat-cable: unknown node {host}")
    port = next((p for p in node.ib if p.ca == ca), None)
    if port is None:
        raise CommandError(f"reseat-cable: {host} has no HCA {ca}")
    from .ib import _link_up

    port.cable_ok = True
    port.rate_gbps = port.expected_rate_gbps  # a reseated or replaced cable trains at full speed again
    was_active = port.state == "Active"
    up = _link_up(port) if node.up else False
    if up and not was_active:
        port.link_downed = port.get("link_downed", 0) + 1
    ctx.state.flags.setdefault("reseated", []).append(f"{host}/{ca}")
    msg = [
        LAB,
        f"A technician reseated the OSFP transceiver and fiber of {host} {ca} (HCA side and the "
        f"matching switch port), after inspecting/cleaning the connector.",
    ]
    if up:
        msg.append("The port LED went from off/amber to green after link training.")
    elif port.phys_state == "Disabled":
        msg.append("The port is administratively disabled, so it stays down (mlxlink --port_state UP).")
    return "\n".join(msg) + "\n"


@command("hwticket", help="LAB HELPER: hwticket create --node N --component C --summary TEXT | list | show ID")
def hwticket(ctx, args):
    tickets = ctx.state.flags.setdefault("hwtickets", [])
    if not args or args[0] == "list":
        if not tickets:
            return "No hardware tickets.\n"
        rows = [f"{'ID':<9} {'NODE':<9} {'COMPONENT':<10} {'STATUS':<8} SUMMARY"]
        rows += [f"{t.id:<9} {t.node:<9} {t.component:<10} {t.status:<8} {t.summary}" for t in tickets]
        return "\n".join(rows) + "\n"
    if args[0] == "show":
        t = next((t for t in tickets if len(args) > 1 and t.id == args[1]), None)
        if t is None:
            raise CommandError("hwticket: no such ticket")
        return "\n".join(f"{k:<10}: {v}" for k, v in t.items()) + "\n"
    if args[0] != "create":
        raise CommandError(
            "usage: hwticket create --node <node> --component <gpu|fan|psu|cable|hca|nvswitch|other> "
            '--summary "text" [--severity 1-4]'
        )
    opts = {"node": None, "component": None, "summary": "", "severity": "2"}
    it = iter(args[1:])
    for a in it:
        if a.startswith("--") and a[2:] in opts:
            opts[a[2:]] = next(it, "")
        elif a.startswith("--") and "=" in a and a[2:].split("=")[0] in opts:
            k, v = a[2:].split("=", 1)
            opts[k] = v
    if opts["node"] not in ctx.state.nodes:
        raise CommandError(f"hwticket: --node must be a cluster node (got {opts['node']})")
    comp = (opts["component"] or "").lower()
    if comp not in ("gpu", "fan", "psu", "cable", "hca", "nvswitch", "memory", "other"):
        raise CommandError("hwticket: --component must be one of gpu, fan, psu, cable, hca, nvswitch, memory, other")
    if len(opts["summary"]) < 10:
        raise CommandError("hwticket: write a useful --summary (symptom + evidence), at least 10 characters")
    tid = f"HW-{1041 + len(tickets)}"
    bundle = "/root/nvidia-bug-report.log.gz" in ctx.state.nodes[opts["node"]].files
    tickets.append(
        {
            "id": tid,
            "node": opts["node"],
            "component": comp,
            "summary": opts["summary"],
            "severity": opts["severity"],
            "status": "OPEN",
            "bug_report_attached": bundle,
        }
    )
    note = (
        ""
        if bundle
        else (
            "\nNote: no nvidia-bug-report.log.gz found on the node. NVIDIA Enterprise Support "
            "will ask for one (run nvidia-bug-report.sh on the node)."
        )
    )
    return f"{LAB}\nCreated {tid} for {opts['node']} ({comp}), severity {opts['severity']}.{note}\n"
