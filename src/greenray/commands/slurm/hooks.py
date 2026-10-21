"""systemctl and reboot hooks: how slurmd, slurmctld and munge service changes reach the simulated controller."""

from __future__ import annotations

import datetime as _dt

from .. import HOOKS_REBOOT, HOOKS_SERVICE
from .common import _GUARD, BASE_TIME, HEAD, MUNGE_KEY, _ts
from .munge import _munge_ok
from .nodes import _derive_state, _finish, _refresh, _register, apply_config
from .scheduler import _ensure, schedule


def _service_hook(ctx, name, action):
    state = ctx.state
    if "slurm" not in state:
        return
    _ensure(state)
    host = ctx.host
    if name == "munge":
        node = ctx.node
        if action in ("start", "restart", "reload"):
            key = node.files.get(MUNGE_KEY)
            if key is None:
                ts = _ts(state).replace("T", " ")
                node.journal.setdefault("munge", []).append(
                    f'{ts} {host} munged[1187]: munged: Error: Failed to find keyfile "{MUNGE_KEY}": No such file or directory'
                )
                node.services["munge"] = "failed"
                return
            node["munge_key_loaded"] = key
            ts = _ts(state).replace("T", " ")
            node.journal.setdefault("munge", []).append(
                f"{ts} {host} systemd[1]: Started MUNGE authentication service."
            )
        _GUARD["depth"] += 1
        try:
            if host != HEAD and host in state.slurm.nodes and _munge_ok(state, host):
                if node.services.get("slurmd") == "active" and state.slurm.nodes[host].not_responding:
                    _register(state, host, log=False)
            _refresh(state)
        finally:
            _GUARD["depth"] -= 1
        return
    if name == "slurmd" and host in state.slurm.nodes:
        if action in ("start", "restart", "reload"):
            ok = _register(state, host)
            if ok is False and ctx.node.services.get("slurmd") == "failed":
                return
            _GUARD["depth"] += 1
            try:
                _refresh(state)
            finally:
                _GUARD["depth"] -= 1
        elif action == "stop":
            state.slurm.nodes[host].not_responding = True
            _derive_state(state, state.slurm.nodes[host])
    elif name == "slurmctld" and host == HEAD:
        if action in ("start", "restart", "reload"):
            state.slurm.controller_up = True
            apply_config(state)
            schedule(state)
        elif action == "stop":
            state.slurm.controller_up = False


def _reboot_hook(ctx, node):
    state = ctx.state
    if "slurm" not in state:
        return
    _ensure(state)
    name = node.hostname if "hostname" in node else ctx.host
    if name == HEAD:
        return
    sn = state.slurm.nodes.get(name)
    if sn is None:
        return
    for j in state.slurm.jobs:
        if j.state == "RUNNING" and name in j.nodelist:
            _finish(state, j, "NODE_FAIL", "0:0")
    sn.boot_time = _ts(state)
    expected = sn.get("_expected_reboot")
    if not expected and not sn.drain and not sn.down:
        sn.down = True
        sn.reason = f"Node unexpectedly rebooted boot_time={int((BASE_TIME - _dt.datetime(1970, 1, 1)).total_seconds()) + state.slurm.clock}"
        sn.reason_user = "slurm"
        sn.reason_time = _ts(state)
    node.services["slurmd"] = "active"
    if "munge" in node.services:
        if MUNGE_KEY in node.files:
            node.services["munge"] = "active"
            node["munge_key_loaded"] = node.files[MUNGE_KEY]
    _register(state, name, after_reboot=True)
    sn["_reregister"] = True  # confirm against the real post-boot GPU state on next use


HOOKS_SERVICE.append(_service_hook)
HOOKS_REBOOT.append(_reboot_hook)
