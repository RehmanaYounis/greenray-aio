"""BCM monitoring: measurables, actions and triggers, as cmsh's ``monitoring`` mode shows them.

The model lives in ``state.bcm.monitoring``. Measurables are read from the simulated nodes at
evaluation time (GPU temperature from the GPUs, health checks from services and dmesg), so a
trigger reacts to whatever the rest of the simulator does to a node. Triggers are evaluated after
every command and at the end of a monitoring commit; a firing runs the enter actions once, when
the condition becomes true, and the leave actions once, when it becomes false again.
"""

from __future__ import annotations

import re

from ...core.state import HEAD
from .. import HOOKS_STATE, HOOKS_TICK
from ..common import dashed_table
from ..slurm.nodes import _derive_state, _set_drain

TRIGGER_PROPS = ("name", "expression", "enteractions", "leaveactions", "severity", "disabled")
SEVERITIES = ("info", "warning", "error")
OPERATORS = (">=", "<=", "==", "!=", ">", "<")

# name -> (type, class, unit)
MEASURABLES = {
    "gpu_temp": ("metric", "GPU", "C"),
    "gpu_mem_temp": ("metric", "GPU", "C"),
    "gpu_power": ("metric", "GPU", "W"),
    "gpu_util": ("metric", "GPU", "%"),
    "gpu_ecc_dbe": ("metric", "GPU", "count"),
    "gpu_sm_clock": ("metric", "GPU", "MHz"),
    "LoadOne": ("metric", "OS", "load"),
    "MemoryUsed": ("metric", "Memory", "B"),
    "FreeSpace": ("metric", "Disk", "B"),
    "ManagedServicesOk": ("healthcheck", "Internal", ""),
    "ssh2node": ("healthcheck", "Network", ""),
    "dmesg": ("healthcheck", "OS", ""),
    "GPUHealth": ("healthcheck", "GPU", ""),
    "diskspace": ("healthcheck", "Disk", ""),
}
MANAGED = ("slurmd", "nvidia-dcgm", "nvidia-fabricmanager", "nvidia-persistenced", "munge")
BUILTIN_ACTIONS = {
    "drain": "Drain the node in the workload manager",
    "undrain": "Undrain the node in the workload manager",
    "poweroff": "Power the node off",
    "poweron": "Power the node on",
    "reboot": "Reboot the node",
    "killprocess": "Kill the process the health check named",
    "Send e-mail to administrators": "Mail the administrators",
    "Event": "Record an event",
}


def monitoring_init(state) -> None:
    bcm = state.bcm
    if "monitoring" not in bcm:
        bcm.monitoring = {
            "actions": {
                name: {"name": name, "builtin": True, "description": desc, "script": ""}
                for name, desc in BUILTIN_ACTIONS.items()
            },
            "triggers": {},
            "fired": {},  # trigger name -> {node: True while the condition holds}
        }


HOOKS_STATE.append(monitoring_init)


# ---------------------------------------------------------------- readings


def _healthcheck(state, host: str, name: str) -> str:
    node = state.nodes[host]
    if not node.get("up", True) or node.get("power", "on") != "on":
        return "FAIL"
    if name == "ssh2node":
        return "PASS"
    if name == "ManagedServicesOk":
        bad = [s for s in MANAGED if s in node.services and node.services[s] != "active"]
        return "FAIL" if bad else "PASS"
    if name == "dmesg":
        return "FAIL" if any("Xid" in line or "fallen off the bus" in line for line in node.dmesg) else "PASS"
    if name == "GPUHealth":
        gpus = node.gpus
        if not node.get("driver_loaded", True) and gpus:
            return "FAIL"
        return "FAIL" if any(g.fallen_off_bus or g.reset_required or g.ecc_volatile_dbe for g in gpus) else "PASS"
    if name == "diskspace":
        return "PASS"
    return "UNKNOWN"


def reading(state, host: str, name: str):
    """Current value of a measurable on a node; GPU metrics report the worst GPU."""
    node = state.nodes.get(host)
    if node is None:
        return None
    kind = MEASURABLES[name][0]
    if kind == "healthcheck":
        return _healthcheck(state, host, name)
    gpus = node.gpus
    if name == "gpu_temp":
        return max((g.temp_c for g in gpus), default=0)
    if name == "gpu_mem_temp":
        return max((g.mem_temp_c for g in gpus), default=0)
    if name == "gpu_power":
        return max((g.power_draw_w for g in gpus), default=0)
    if name == "gpu_util":
        return max((g.utilization for g in gpus), default=0)
    if name == "gpu_ecc_dbe":
        return sum(g.ecc_volatile_dbe for g in gpus)
    if name == "gpu_sm_clock":
        return min((g.sm_clock_mhz for g in gpus), default=0)
    if name == "LoadOne":
        return round(0.3 + 0.8 * sum(1 for g in gpus if g.utilization > 50), 2)
    if name == "MemoryUsed":
        return 48 * 1024**3 + sum(g.memory_used_mib for g in gpus) * 1024**2
    if name == "FreeSpace":
        return 12 * 1024**4 - sum(len(c) for p, c in node.files.items() if p.startswith("/raid/"))
    return None


def _fmt(name: str, value) -> str:
    unit = MEASURABLES[name][2]
    if isinstance(value, str):
        return value
    if unit == "B":
        for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024 or suffix == "TiB":
                return f"{value:.1f} {suffix}" if suffix != "B" else f"{value} B"
            value /= 1024
    return f"{value} {unit}".strip()


def latest(state, hosts: list[str], kind: str) -> str:
    """Rows for ``latestmetricdata`` (metrics) or ``latesthealthdata`` (health checks)."""
    rows = []
    for host in hosts:
        for name, (mtype, _cls, _unit) in MEASURABLES.items():
            if mtype != kind:
                continue
            value = reading(state, host, name)
            rows.append((host, name, _fmt(name, value), "" if value is None else "now"))
    return dashed_table(["Device", "Measurable", "Value", "Age"], rows, [12, 24, 20, 8])


# ---------------------------------------------------------------- triggers


def parse_expression(text: str) -> dict:
    """``gpu_temp > 85`` -> {measurable, operator, value}; raises ValueError with cmsh's wording."""
    m = re.fullmatch(r"\s*(\w+)\s*(>=|<=|==|!=|>|<)\s*(\S+)\s*", text)
    if not m:
        raise ValueError("Error: expression must look like '<measurable> <operator> <value>', e.g. gpu_temp > 85")
    name, op, raw = m.groups()
    if name not in MEASURABLES:
        raise ValueError(f"Error: unknown measurable '{name}' (see monitoring; measurable; list)")
    if MEASURABLES[name][0] == "healthcheck":
        if op not in ("==", "!=") or raw.upper() not in ("PASS", "FAIL"):
            raise ValueError(f"Error: {name} is a health check; compare it with == PASS or == FAIL")
        return {"measurable": name, "operator": op, "value": raw.upper()}
    try:
        value: float | int = int(raw) if raw.lstrip("-").isdigit() else float(raw)
    except ValueError:
        raise ValueError(f"Error: '{raw}' is not a number") from None
    return {"measurable": name, "operator": op, "value": value}


def _holds(expr: dict, value) -> bool:
    if value is None:
        return False
    op, want = expr["operator"], expr["value"]
    if isinstance(want, str):
        return (value == want) if op == "==" else (value != want)
    return {
        ">": value > want,
        "<": value < want,
        ">=": value >= want,
        "<=": value <= want,
        "==": value == want,
        "!=": value != want,
    }[op]


def _run_action(state, host: str, action: str, trigger: str) -> str:
    from ..slurm.scheduler import _ensure

    sn = None
    if "slurm" in state:
        _ensure(state)
        sn = state.slurm.nodes.get(host)
    node = state.nodes[host]
    if action == "drain" and sn is not None:
        _set_drain(state, sn, f"BCM trigger {trigger}", user="cmd")
        _derive_state(state, sn)
        return "drained"
    if action == "undrain" and sn is not None:
        if sn.drain and sn.reason.startswith("BCM trigger"):
            sn.drain, sn.reason, sn.reason_user = False, "", ""
            _derive_state(state, sn)
            return "undrained"
        return "left alone (not drained by a trigger)"
    if action == "poweroff":
        node.power, node.up = "off", False
        return "powered off"
    if action == "poweron":
        node.power, node.up = "on", True
        return "powered on"
    if action in ("reboot", "killprocess"):
        return f"{action} requested"
    if action == "Send e-mail to administrators":
        return "mail sent to root"
    return "ran"


def evaluate(state) -> list[str]:
    """Evaluate every trigger on every compute node; returns the event lines written."""
    if "bcm" not in state or "monitoring" not in state.bcm:
        return []
    from .model import NOW

    mon = state.bcm.monitoring
    written = []
    for name, trig in mon.triggers.items():
        if trig.get("disabled") or not trig.get("expression"):
            continue
        expr = trig.expression
        active = mon.fired.setdefault(name, {})
        for host, dev in state.bcm.devices.items():
            if dev.type == "HeadNode" or host not in state.nodes:
                continue
            value = reading(state, host, expr.measurable)
            now = _holds(expr, value)
            was = bool(active.get(host))
            if now == was:
                continue
            active[host] = now
            actions = trig.get("enteractions" if now else "leaveactions", [])
            results = [f"{a}: {_run_action(state, host, a, name)}" for a in actions]
            shown = _fmt(expr.measurable, value)
            line = (
                f"{NOW} [{trig.get('severity', 'warning')}] {HEAD}: trigger {name} "
                f"{'fired' if now else 'cleared'} for {host}: {expr.measurable} {shown} "
                f"{'' if now else 'no longer '}{expr.operator} {expr.value}"
                + (f"; actions: {', '.join(results)}" if results else "")
            )
            state.bcm.setdefault("events", []).append(line)
            state.nodes[HEAD].journal.setdefault("cmd", []).append(
                f"{NOW} {HEAD} cmd[1127]: Info: monitoring: {line.split(': ', 1)[1]}"
            )
            written.append(line)
    return written


HOOKS_TICK.append(evaluate)


# ---------------------------------------------------------------- cmsh helpers


def validate_trigger(prop: str, value: str, actions: dict, err) -> object:
    if prop == "expression":
        try:
            return parse_expression(value)
        except ValueError as e:
            err(str(e))
    if prop in ("enteractions", "leaveactions"):
        items = [x.strip() for x in value.split(",") if x.strip()]
        for a in items:
            if a not in actions:
                err(f"Error: unknown action '{a}' (see monitoring; action; list)")
        return items
    if prop == "severity":
        if value.lower() not in SEVERITIES:
            err(f"Error: severity must be one of {', '.join(SEVERITIES)}")
        return value.lower()
    if prop == "disabled":
        if value.lower() not in ("yes", "no", "true", "false"):
            err("Error: disabled must be yes or no")
        return value.lower() in ("yes", "true")
    if prop == "name":
        err("Error: property 'name' is read-only")
    return value


def expression_text(expr) -> str:
    return f"{expr.measurable} {expr.operator} {expr.value}" if expr else ""
