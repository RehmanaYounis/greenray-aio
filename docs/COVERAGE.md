# Exam coverage map

This maps every topic in the NCP-AIO blueprint to the hands-on scenarios (`aiolab lab <id>`) and the question-bank tags that practise it. `pytest` checks that every scenario listed here exists, so the map can't silently go stale.

**Exam format.** The NVIDIA certification page, as reported by search results in October 2026, describes 30 multiple-choice questions plus 3 hands-on lab exercises in one 120-minute, remotely proctored session. The labs run on a Linux command line against live Slurm, Kubernetes and Base Command Manager clusters. `aiolab mock` reproduces that shape. Re-check the official page before your exam.

**Topic sources.** The domain names and weights come from the official page (Installation and Deployment 31%, Administration 23%, Workload Management 23%, Troubleshooting and Optimization 23%). The topic bullets are paraphrased from public summaries of the blueprint. Official materials win if they differ.

## Installation and Deployment (31%)

| Topic | Hands-on scenarios |
| --- | --- |
| Install and configure Base Command Manager; add and provision nodes | `ins-bcm-add-node-category`, `ins-bcm-softwareimage-update` |
| Set up a workload manager through BCM (Slurm) | `ins-slurm-gres-config` |
| Initialise Kubernetes on NVIDIA hosts; GPU Operator | `ins-gpu-operator-preinstalled-driver` |
| Network Operator, RDMA for pods | `ins-network-operator-rdma` |
| Container runtime for GPUs (NVIDIA Container Toolkit) | `ins-container-toolkit-docker` |
| Deploy containers from NGC (and cloud VMI images) | `ins-ngc-registry-auth` |
| Storage requirements: shared filesystems for the cluster | `ins-storage-shared-mounts` |
| Deploy DOCA services on BlueField DPUs | `ins-dpu-doca-service` |
| NVIDIA Mission Control, cloud VMIs, firmware | question bank only (concepts) |

## Administration (23%)

| Topic | Hands-on scenarios |
| --- | --- |
| BCM monitoring, node status, power control | `adm-bcm-power-and-status` |
| BCM patching and image synchronisation (grabimage / imageupdate) | `ins-bcm-softwareimage-update`, `adm-bcm-grabimage-sync` |
| BCM users and roles | `adm-bcm-user-role` |
| Slurm partitions, QOS and limits | `adm-slurm-partition-qos` |
| Slurm accounts and fairshare | `adm-slurm-fairshare-accounts` |
| MIG configuration | `adm-mig-h100-1g` |
| MIG in Kubernetes | `adm-k8s-mig-mixed-strategy` |
| GPU sharing in Kubernetes (time-slicing) | `adm-k8s-time-slicing` |
| Run:ai projects, quotas and resource allocation between teams | `adm-runai-project-quota` |
| DCGM health watches and policies | `adm-dcgm-health-policy` |
| Power and persistence settings | `tsh-power-cap-persistence` |

## Workload Management (23%)

| Topic | Hands-on scenarios |
| --- | --- |
| Slurm job submission and pending reasons | `wkl-slurm-pending-reasons` |
| Multi-node Slurm GPU jobs | `wkl-slurm-multinode-sbatch` |
| Containers in Slurm (pyxis/enroot) | `wkl-pyxis-container-job` |
| Kubernetes GPU pods and scheduling | `wkl-k8s-pod-pending-gpu` |
| Taints, tolerations and node selection | `wkl-k8s-taint-toleration` |
| Run:ai workload submission and fractional GPUs | `wkl-runai-fractional-gpu` |

## Troubleshooting and Optimization (23%)

| Topic | Hands-on scenarios |
| --- | --- |
| XID errors and GPUs falling off the bus | `tsh-xid79-drained-node` |
| ECC and row remapping | `tsh-ecc-row-remap` |
| DCGM diagnostics | `tsh-dcgm-diag-failure` |
| Thermal and power throttling | `tsh-thermal-throttle`, `tsh-power-cap-persistence` |
| Fabric Manager / NVSwitch | `tsh-fabricmanager-down` |
| InfiniBand link problems | `tsh-ib-port-down` |
| NCCL and multi-node performance | `tsh-nccl-ib-hca` |
| GPUDirect RDMA (Magnum IO) | `tsh-gpudirect-peermem` |
| Storage performance (Magnum IO / GPUDirect Storage) | `tsh-storage-io-bottleneck` |
| CPU/GPU NUMA placement | `tsh-numa-affinity` |
| Docker / container GPU access | `tsh-docker-gpu-runtime` |
| Kubernetes GPU stack failures | `tsh-device-plugin-crashloop` |
| Slurm node DOWN (munge, slurmd) | `tsh-slurm-node-down-munge` |
