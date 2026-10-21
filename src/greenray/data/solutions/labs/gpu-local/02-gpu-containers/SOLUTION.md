# Lab 02: Solution notes

**Scenario 1.** Without `--gpus`, Docker uses the default `runc` path: no GPU device nodes or driver libraries are injected, so `nvidia-smi` is "executable file not found" or fails to talk to the driver. Fix:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

`--gpus` asks the NVIDIA Container Toolkit (`nvidia-container-runtime` / `nvidia-container-cli`) to inject the devices and the driver's user-space libraries **from the host**. That's why the image never contains a driver, only CUDA user-space. Docker Desktop on Windows ships this integration. On a Linux server (DGX) you install `nvidia-container-toolkit` and run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

**Scenario 2.**

```bash
docker run --rm --gpus '"device=0"' nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
docker run --rm --gpus '"device=GPU-xxxxxxxx-...."' nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L
```

The value is parsed as CSV, so the inner double quotes keep `device=0,1` as one field. The outer single quotes stop the shell from removing them. The environment-variable equivalent is `-e NVIDIA_VISIBLE_DEVICES=0` with the nvidia runtime. Indexes can change between boots, while UUIDs are stable, so automation should use UUIDs. MIG devices are selected by their `MIG-...` UUIDs.

**Scenario 3.** An RTX 4060 Laptop typically reaches tens of TFLOPS in fp16. The laptop's power and thermal limits mean `SW Power Cap` or thermal reasons often appear, a smaller version of the same diagnosis you do on a DGX.

**Scenario 4.** The username is the literal string `$oauthtoken`, and the password is the NGC API key. Many NGC images are public, but private, org/team or early-access content and the NGC CLI need the key. The key belongs only in Docker's credential store, `~/.ngc/config` or Kubernetes Secrets (an `imagePullSecret` of type `kubernetes.io/dockerconfigjson`), never in Git. Simulator lab: `aiolab lab ins-ngc-registry-auth`.

## Exam traps

- **Driver inside the image:** wrong. The driver always comes from the host. The image only needs a CUDA version the host driver supports.
- **Pyxis syntax:** Slurm's pyxis writes images as `nvcr.io#nvidia/pytorch:24.05-py3` (`#` after the registry), not `/`. Practise in `aiolab lab wkl-pyxis-container-job`.
- **Containers that can't find the runtime:** the runtime isn't configured. On containerd/Kubernetes nodes run `nvidia-ctk runtime configure --runtime=containerd` (the GPU Operator's toolkit container does this for you).
