# Lab 03: Solution notes

**Setup question.** No. `nvidia.com/gpu` is an *extended resource*. It appears only after the **NVIDIA device plugin** (a DaemonSet) registers with the kubelet and advertises how many GPUs the node has. `minikube --gpus=all` enables the plugin as an addon. On a production cluster the **GPU Operator** deploys it, together with the container toolkit, GPU Feature Discovery labels, the DCGM exporter, the MIG manager and validators.

**Scenario 1.**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: cuda-check
spec:
  restartPolicy: Never
  containers:
  - name: cuda
    image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
    command: ["nvidia-smi", "-L"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

```bash
kubectl apply -f gpu-pod.yaml
kubectl get pod cuda-check -w
kubectl logs cuda-check
```

**Scenario 2.** The pod stays `Pending`. `kubectl describe pod <name>` shows a `FailedScheduling` event like `0/1 nodes are available: 1 Insufficient nvidia.com/gpu.` The scheduler lists the reason each node was rejected, counted per reason. On a real cluster it may read `0/5 nodes are available: 1 node(s) had untolerated taint {...}, 4 Insufficient nvidia.com/gpu`. Fix: request what one node can provide (1 here), or run on a node with more GPUs. A pod can never span nodes; multi-node training uses several pods (for example via MPIJob, PyTorchJob or Run:ai).

**Scenario 3.** For extended resources, `limits` is what matters. If you set only `requests`, the API server rejects the pod because extended resources must have limits. If you set both, they must be equal. Extended resources must be integers, so `0.5` is rejected. Sharing a GPU needs **time-slicing** (device plugin config: replicas, with no isolation), **MPS**, or **MIG** (hardware isolation, data-centre GPUs only). Run:ai's fractional GPUs are a scheduler-level feature built on top of these.

## Exam traps

- Pending plus `Insufficient nvidia.com/gpu` while GPUs are idle often means the **device plugin isn't running** on that node (operator pods failing, or a node label or taint keeping them off). Check `kubectl get pods -n gpu-operator -o wide` first.
- With MIG, the resource names depend on the strategy: `single` advertises `nvidia.com/gpu`, and `mixed` advertises `nvidia.com/mig-1g.10gb` and similar. Simulator: `aiolab lab adm-k8s-mig-mixed-strategy`.
- On DGX nodes the driver is preinstalled, so install the GPU Operator with `--set driver.enabled=false`. Simulator: `aiolab lab ins-gpu-operator-preinstalled-driver`.
