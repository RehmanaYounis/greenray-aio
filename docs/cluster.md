---
title: The simulated DGX cluster
description: "The simulated four-node DGX H100 cluster behind the NCP-AIO labs: Base Command Manager, Slurm, Kubernetes with the GPU Operator, and the tools available on each node."
---

# The simulated cluster

Every lab starts from the same healthy cluster, then the scenario breaks something in it. It is a small slice of a DGX SuperPOD managed by Base Command Manager.

| Host | Role |
| --- | --- |
| `bcm-head` | BCM head node (`cmsh`), Slurm controller, Kubernetes control plane, NGC CLI |
| `dgx-01` to `dgx-04` | DGX H100: 8 H100 80GB GPUs, NVSwitch, 8 ConnectX-7 NDR 400G ports, `slurmd`, `kubelet`, DCGM |

You start on `bcm-head` as root. `ssh dgx-01` moves your shell to a compute node and `exit` brings it back. The prompt, the login banner, `uptime` and the timestamps in `dmesg -T` follow a simulated clock that advances as you work.

## Tools

| Area | Commands |
| --- | --- |
| GPU | `nvidia-smi` (status, `-q -d`, `--query-gpu`, power, persistence, ECC, reset, MIG, `topo -m`, `nvlink -s`, `dmon`) |
| Health | `dcgmi` (discovery, groups, health, policy, diag, stats, dmon, nvlink, topo, fieldgroup, profile), `nvsm` (show health, versions, alerts, gpus, storage, power, fans; dump health; update firmware), `dmesg`, `journalctl`, `systemctl`, `ipmitool` (sel, sdr, power, chassis status) |
| Slurm | `sinfo`, `squeue`, `scontrol` (nodes, partitions, jobs, reservations, suspend, resume, top, reconfigure with the controller's checks), `sbatch` (job arrays, `--begin`, `--mem`, `--dependency`, `--test-only`, `--wait`, scripts on stdin), `srun`, `salloc` (an allocation shell that `exit` releases; pyxis), `sacct`, `sacctmgr`, `sprio`, `sshare`, `scancel`. Time limits end jobs with `TIMEOUT`; a plain user cannot cancel other users' jobs or change nodes, and `cmsh` needs an administrator profile |
| Kubernetes | `kubectl` (get, describe, apply, delete, logs, exec, events, taint, label, rollout with history and undo, set image, expose, scale, explain, config, replica sets, CRDs), `helm` (GPU Operator, Network Operator), the Run:ai CLI (installed only in the Run:ai scenarios) |
| Containers | `docker` (images, containers, `system`, `volume`, `network`), `docker compose`, `enroot`, `ngc`, `nvidia-ctk`, `crictl` |
| BCM | `cmsh` in one-shot mode (device, category, softwareimage with its `kernelmodules` submode, `foreach` and `range` over nodes, partition, user, `wlm` with its job queues, `configurationoverlay`, `monitoring` with its measurables, actions and triggers, `latestmetricdata` and `latesthealthdata`, power, imageupdate, grabimage, `main; events`), `cm-wlm-setup` and `cm-kubernetes-setup` (explain what the wizards did), `cmha` |
| Fabric | `ibstat`, `ibdev2netdev`, `ibdiagnet`, `mlxlink`, `perfquery`, NCCL tests (`all_reduce_perf` through `mpirun`), GPUDirect RDMA and GPUDirect Storage checks |
| DPU | BlueField tools (`mlxconfig`, `bfb-install`, `rshim`, DOCA services) |
| Shell | `cd` (and `cd -`), `pwd`, `ls`, `cat`, `less`, `grep` (with its usual options), `sed` (substitutions, deletes, line ranges, `-i`, `-n`, `-E`), `awk` (field printing with guards, `-F`, `NR`, `END`), `cut`, `sort`, `uniq`, `tr`, `head`, `tail`, `wc`, `find`, `mv`, `cp`, `diff`, `tar`, `jq`, `date`, `xargs`, `test`, `bash script.sh`, `watch`, `clear`, `pdsh`, `sudo`, `su`, `env`, `export`; variables (`X=1`, `$X`, `${X:-default}`, `$((2+3))`, `$HOME`, `$USER`, `$?`), globbing (`*`, `?`, `[...]`), heredocs, pipes, `&&`, `||`, `;`, redirects (`>`, `>>`, `<`, `2>`, `2>&1`, `&>`), `$(...)`; `nano`, `vi` and `vim` open your real editor; `--help` and `man` work as on a real node |
| Node diagnostics | `top`, `htop`, `ps`, `pgrep`, `kill`, `lsof`, `fuser`, `free`, `vmstat`, `uptime`, `w`, `last`, `hostnamectl`, `timedatectl`, `lscpu`, `lsblk`, `numactl`, `lsmod`, `modinfo`, `ss`, `ip`, `ifconfig`, `ethtool`, `ping`, `nslookup`, `traceroute`, `curl`, `shutdown`; `/etc/hosts`, `/etc/os-release`, `/proc/uptime`, `/proc/driver/nvidia/version` and the `/dev/nvidia*` device nodes follow the node's state |

`commands` inside a lab prints the full list for the current version, and `explain <command>` describes what a tool is for and what to look for in its output.

## Slurm configuration

The cluster's `slurm.conf` is on the head node at `/etc/slurm/slurm.conf`, and the simulator follows what it says.

Memory is a tracked resource (`SelectTypeParameters=CR_Core_Memory`) with a default of 8 GB per CPU (`DefMemPerCPU=8192`). A job that names its memory (`--mem`, `--mem-per-cpu`, `--mem-per-gpu`) is allocated that much. A job that names none is allocated the default for its CPUs, so a one-CPU job holds 8 GB and several jobs can share a node. `sacct` (`ReqMem`, `ReqTRES`, `AllocTRES`), `scontrol show job`, `scontrol show node` (`AllocMem`, `AllocTRES`) and `scontrol show partition` report the same figures, and a job keeps what it was given when it started even if `slurm.conf` changes later.

Remove the default and the cluster behaves as a real one does without it: a job that names no memory asks for 0 and is given all of the node's `RealMemory`, and a second job cannot share that node until the first ends (reason `Resources`). `DefMemPerNode` and `DefMemPerGPU` are honoured in its place.

## Limits

Outputs are realistic but simplified. Only the options that matter for the exam are implemented, and a command that does not parse usually works in its canonical form. The shell runs one command line at a time: `for` and `while` loops, `if`, functions and background jobs (`&`) are not supported, and `cmsh` keeps its staged changes only until the end of the `cmsh -c` invocation, so `commit` belongs in the same script as the `set`. A tool's `--help` page lists exactly the options the simulator honours: the test-suite runs every one of them and fails if an option changes nothing. A `--long` option the tool does not know is refused in the tool's own words instead of being ignored, an unknown subcommand of `ipmitool` or `systemctl` likewise, and `ssh <host> <command>` ends with the exit status of the remote command, so `&&` and `||` behave as on a real node. The grader reads the end state of the cluster, so what matters is what you changed, not the exact flags you used to change it.

If a real tool behaves differently from the simulator, trust the real one and please [open an issue](https://github.com/RehmanaYounis/greenray-aio/issues/new?template=tool_difference.md) with the command and what the real tool does.
