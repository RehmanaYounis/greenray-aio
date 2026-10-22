# Exam coverage map

This maps every topic in the NCP-AIO blueprint to the hands-on scenarios (`greenray lab <id>`) and the question-bank tags that practise it. `pytest` checks that every scenario listed here exists, so the map can't silently go stale.

**Exam format.** The NVIDIA certification page, as reported by search results in October 2026, describes 30 multiple-choice questions plus 3 hands-on lab exercises in one 120-minute, remotely proctored session. The labs run on a Linux command line against live Slurm, Kubernetes and Base Command Manager clusters. `greenray mock` reproduces that shape. Re-check the official page before your exam.

**Topic sources.** The domain names and weights come from the official page (Installation and Deployment 31%, Administration 23%, Workload Management 23%, Troubleshooting and Optimization 23%). The topic bullets are paraphrased from public summaries of the blueprint. Official materials win if they differ.

## On-ramp (start here)

Six short scenarios, ten minutes each, for the first week. They practise the Linux and cluster basics every other scenario assumes: reading logs, editing a configuration file and applying it, submitting a job, reading pod events, reading BCM settings, and naming a GPU three different ways.

| Skill | Scenario |
| --- | --- |
| Reading Slurm accounting and node logs (`sacct`, `journalctl`, `dmesg`) | `tsh-first-look-at-logs` |
| Editing `slurm.conf` and making slurmctld re-read it | `adm-slurm-partition-time-limit` |
| Writing and submitting a GPU batch script (`sbatch`, `squeue`, `sacct`) | `wkl-slurm-first-gpu-job` |
| Reading pod events and node labels (`kubectl describe`, `nodeSelector`) | `wkl-k8s-pod-pending-nodeselector` |
| Reading device, category and image settings in `cmsh` | `adm-bcm-cmsh-inventory` |
| Mapping PCI bus id, index, UUID and serial of a GPU (`nvidia-smi`) | `tsh-which-gpu-is-which` |

## Cross-system incidents

Three hard scenarios where the symptom sits in one system and the cause in another, as it does on the exam's live clusters. Each has three variants and belongs to the mock-exam pool.

| Symptom | Cause | Scenario |
| --- | --- | --- |
| A DGX node is DOWN in Slurm with no NVIDIA driver after a reboot | Its BCM category was changed, so the node-installer provisioned the wrong image | `tsh-node-lost-driver-after-category-change` |
| A two-node training run is eight times slower | The updated software image lost the GPUDirect RDMA module, and one IB link trained at 100G | `tsh-nccl-slow-after-image-update` |
| A Deployment replica stays Pending on a labelled node pool | One pool node was switched to MIG and no longer advertises whole GPUs | `wkl-k8s-replica-pending-after-mig` |

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
| BCM monitoring, node status, power control | `adm-bcm-power-and-status`, `adm-bcm-cmsh-inventory` |
| BCM patching and image synchronisation (grabimage / imageupdate) | `ins-bcm-softwareimage-update`, `adm-bcm-grabimage-sync`, `tsh-node-lost-driver-after-category-change` |
| BCM users and roles | `adm-bcm-user-role` |
| Slurm partitions, QOS and limits | `adm-slurm-partition-qos`, `adm-slurm-partition-time-limit` |
| Slurm accounts and fairshare | `adm-slurm-fairshare-accounts` |
| MIG configuration | `adm-mig-h100-1g` |
| MIG in Kubernetes | `adm-k8s-mig-mixed-strategy`, `wkl-k8s-replica-pending-after-mig` |
| GPU sharing in Kubernetes (time-slicing) | `adm-k8s-time-slicing` |
| Run:ai projects, quotas and resource allocation between teams | `adm-runai-project-quota` |
| DCGM health watches and policies | `adm-dcgm-health-policy` |
| Power and persistence settings | `tsh-power-cap-persistence` |

## Workload Management (23%)

| Topic | Hands-on scenarios |
| --- | --- |
| Slurm job submission and pending reasons | `wkl-slurm-first-gpu-job`, `wkl-slurm-pending-reasons` |
| Multi-node Slurm GPU jobs | `wkl-slurm-multinode-sbatch` |
| Containers in Slurm (pyxis/enroot) | `wkl-pyxis-container-job` |
| Kubernetes GPU pods and scheduling | `wkl-k8s-pod-pending-nodeselector`, `wkl-k8s-pod-pending-gpu` |
| Taints, tolerations and node selection | `wkl-k8s-taint-toleration` |
| Run:ai workload submission and fractional GPUs | `wkl-runai-fractional-gpu` |

## Troubleshooting and Optimization (23%)

| Topic | Hands-on scenarios |
| --- | --- |
| GPU identification (bus id, index, UUID, serial) | `tsh-which-gpu-is-which` |
| XID errors and GPUs falling off the bus | `tsh-xid79-drained-node` |
| ECC and row remapping | `tsh-ecc-row-remap` |
| DCGM diagnostics | `tsh-dcgm-diag-failure` |
| Thermal and power throttling | `tsh-thermal-throttle`, `tsh-power-cap-persistence` |
| Fabric Manager / NVSwitch | `tsh-fabricmanager-down` |
| InfiniBand link problems | `tsh-ib-port-down` |
| NCCL and multi-node performance | `tsh-nccl-ib-hca`, `tsh-nccl-slow-after-image-update` |
| GPUDirect RDMA (Magnum IO) | `tsh-gpudirect-peermem` |
| Storage performance (Magnum IO / GPUDirect Storage) | `tsh-storage-io-bottleneck` |
| CPU/GPU NUMA placement | `tsh-numa-affinity` |
| Docker / container GPU access | `tsh-docker-gpu-runtime` |
| Kubernetes GPU stack failures | `tsh-device-plugin-crashloop` |
| Slurm node DOWN (munge, slurmd) | `tsh-slurm-node-down-munge` |
| Reading logs: job failures, OOM | `tsh-first-look-at-logs` |
