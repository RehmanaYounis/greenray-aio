"""Audit log of destructive operations, so scenarios can grade how a fix was done, not only that it was.

Commands append an event when they do something that costs users work while it matters
(a reboot with jobs running, another user's job cancelled). Checks read it with ``audit()``.
"""

from __future__ import annotations

REBOOT_WITH_JOBS = "reboot_with_jobs"
POWER_OFF_WITH_JOBS = "power_off_with_jobs"
NODE_DOWN_WITH_JOBS = "node_down_with_jobs"
CANCEL_USER_JOB = "cancel_user_job"
DELETE_RUNNING_POD = "delete_running_pod"
RUNTIME_RESTART_WITH_CONTAINERS = "runtime_restart_with_containers"


def record(state, kind: str, host: str, command: str, detail: str = "") -> None:
    state.setdefault("audit", []).append({"kind": kind, "host": host, "command": command, "detail": detail})


def events(state, kind: str | None = None) -> list:
    return [e for e in state.get("audit", []) if kind is None or e["kind"] == kind]


def running_jobs_on(state, host: str) -> list:
    return [j for j in state.slurm.jobs if j.state == "RUNNING" and host in j.nodelist]
