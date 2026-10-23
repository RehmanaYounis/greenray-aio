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
| Health | `dcgmi` (discovery, groups, health, policy, diag, stats, dmon), `dmesg`, `journalctl`, `systemctl`, `ipmitool` |
| Slurm | `sinfo`, `squeue`, `scontrol`, `sbatch`, `srun`, `salloc` (with pyxis), `sacct`, `sacctmgr`, `sprio`, `sshare`, `scancel` |
| Kubernetes | `kubectl` (get, describe, apply, delete, logs, exec, events, taint, label, rollout), `helm` (GPU Operator, Network Operator), the Run:ai CLI |
| Containers | `docker`, `docker compose`, `enroot`, `ngc`, `nvidia-ctk`, `crictl` |
| BCM | `cmsh` in one-shot mode (device, category, softwareimage, partition, user, power, imageupdate, grabimage), `cmha` |
| Fabric | `ibstat`, `ibdiagnet`, `mlxlink`, `perfquery`, NCCL tests (`all_reduce_perf` through `mpirun`), GPUDirect RDMA and GPUDirect Storage checks |
| DPU | BlueField tools (`mlxconfig`, `bfb-install`, `rshim`, DOCA services) |
| Shell | `cd`, `pwd`, `ls`, `cat`, `less`, `grep`, `sed -i`, `awk`, `cut`, `sort`, `uniq`, `tr`, `head`, `tail`, `wc`, `find`, `mv`, `cp`, `diff`, `tar`, `date`, `xargs`, `test`, `bash script.sh`, `watch`, `clear`, heredocs, pipes, `&&`, redirects, `$(...)`; `nano`, `vi` and `vim` open your real editor; `--help` and `man` work as on a real node |
| Node diagnostics | `top`, `htop`, `ps`, `pgrep`, `kill`, `lsof`, `fuser`, `free`, `vmstat`, `uptime`, `w`, `hostnamectl`, `timedatectl`, `lscpu`, `numactl`, `lsmod`, `modinfo`, `ss`, `ip`, `ifconfig`, `ethtool`, `ping`, `nslookup`, `traceroute`, `curl`, `shutdown` |

`commands` inside a lab prints the full list for the current version, and `explain <command>` describes what a tool is for and what to look for in its output.

## Limits

Outputs are realistic but simplified. Only the options that matter for the exam are implemented, and a command that does not parse usually works in its canonical form. A tool's `--help` page lists exactly the options the simulator honours: the test-suite runs every one of them and fails if an option changes nothing. The grader reads the end state of the cluster, so what matters is what you changed, not the exact flags you used to change it.

If a real tool behaves differently from the simulator, trust the real one and please [open an issue](https://github.com/RehmanaYounis/greenray-aio/issues/new?template=tool_difference.md) with the command and what the real tool does.
