"""Slurm emulation: sinfo, squeue, scontrol, sbatch, srun, salloc, scancel, sacct,
sacctmgr, sprio, sshare, slurmd -C, plus a tiny scheduler.

State lives in ``state.slurm`` (see greenray/state.py). Extra fields this module adds
lazily (scenario setups may set them directly):

  slurm.config        parsed *running* slurm.conf: {"params": {lowercase key: value},
                      "names": {lowercase: OriginalKey}, "nodes": {...}, "partitions": {...}}
                      Re-read only on `scontrol reconfigure` or slurmctld restart.
  slurm.clock         simulated seconds since 2026-10-19 09:00:00; every Slurm command
                      advances it by 30 s, which is how batch jobs progress and finish.
  slurm.nodes[n]      + drain, down, invalid_reg, not_responding (flags; `state` is derived
                      from them: IDLE / MIXED / ALLOCATED / DRAIN / DRAINING / INVAL / DOWN),
                      reason_user, reason_time, gres_reported, drain_log, last_resume_diag
  slurm.partitions[p] + qos, max_nodes, default_time
  slurm.qos[q]        + max_tres_per_job, max_jobs_per_user, max_wall, grp_tres, flags
  slurm.users[u]      + qos (allowed list), accounts

Job fields (``slurm.jobs`` entries; setups may give only some of them):
  id, name, user, account, partition, qos, state (PENDING/RUNNING/COMPLETED/FAILED/
  CANCELLED/NODE_FAIL/TIMEOUT), reason, num_nodes, ntasks_per_node, cpus_per_task,
  gpus_per_node, gpu_type, time_limit (minutes, None = partition limit), req_nodes,
  exc_nodes, nodelist (list), alloc_gpus {node: [idx]}, submit_time, start_time,
  end_time, duration (seconds; None = runs until cancelled), command, script, work_dir,
  stdout, container_image, exit_code ("0:0"), priority, batch, held, dependency,
  exclusive, restarts, steps
"""

from __future__ import annotations

# Importing the command modules runs their @command registrations and installs the hooks.
from . import accounting, ctl, hooks, jobs, munge, sacctmgr, status, users  # noqa: F401
from .affinity import os_cpus, task_binding
from .common import (
    _TASK,
    _USER,
    BASE_TIME,
    HEAD,
    SLURM_VERSION,
    TICK,
    compress_hostlist,
    expand_hostlist,
    fmt_elapsed,
    fmt_limit,
    fmt_limit_upper,
    parse_gres_spec,
    parse_time,
    parse_tres,
)
from .conf import parse_slurm_conf
from .jobs import parse_job_opts
from .nodes import apply_config
from .scheduler import schedule, tick

__all__ = [
    "BASE_TIME",
    "HEAD",
    "SLURM_VERSION",
    "TICK",
    "_TASK",
    "_USER",
    "apply_config",
    "compress_hostlist",
    "expand_hostlist",
    "fmt_elapsed",
    "fmt_limit",
    "fmt_limit_upper",
    "os_cpus",
    "parse_gres_spec",
    "parse_job_opts",
    "parse_slurm_conf",
    "parse_time",
    "parse_tres",
    "schedule",
    "task_binding",
    "tick",
]
