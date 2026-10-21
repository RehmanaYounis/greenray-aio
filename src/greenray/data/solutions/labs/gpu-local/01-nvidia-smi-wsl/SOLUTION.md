# Lab 01: Solution notes

Try the lab first. Your exact numbers will differ; what matters is the reasoning.

**Task 1.** "CUDA Version" in the header is the **highest CUDA version the installed driver supports**. It is not a CUDA toolkit installed on the system. A container built for CUDA 12.4 runs on any driver that reports 12.4 or newer. The WDDM driver model means Windows owns the GPU, and WSL reaches it through a paravirtualised interface.

**Task 2.**

```bash
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm --format=csv,noheader,nounits
```

`nvidia-smi --help-query-gpu` lists every field. This is how monitoring scripts and Slurm health checks read GPU state reliably, without parsing the table.

**Task 3.** Under load, `sm` (utilisation), `pwr`, `pclk` and `mclk` rise, and the P-state drops towards P0/P2. `-s pucm` selects power/temperature, utilisation, clocks and memory.

**Task 4.** At idle the GPU reduces clocks on purpose, so "GPU Idle" is active and that is healthy. On a busy data-centre GPU, look for `SW Power Cap` (a power limit), `HW Slowdown` or `HW Thermal Slowdown` (cooling or power-delivery hardware), and `SW Thermal Slowdown`. Practise the diagnosis in `aiolab lab tsh-power-cap-persistence` and `tsh-thermal-throttle`.

**Task 5 (typical results on GeForce + WSL2):**

| Command | Typical result | Meaning |
| --- | --- | --- |
| `-pm 1` | Not supported in WSL / Windows | Persistence mode is a Linux-native-driver feature. On DGX, prefer the `nvidia-persistenced` daemon |
| `-mig 1` | "Not Supported" | MIG exists only on data-centre GPUs: A100, A30, H100, H200, B200 and similar |
| `-q -d ECC` | ECC "N/A" | GeForce memory has no ECC reporting. Data-centre GPUs show volatile/aggregate counters and remapped rows |
| `-pl 60` | "Insufficient permissions" or "not supported" | Laptops and WDDM lock power management. On Linux data-centre GPUs, root can set any value between the min and max limits |

**Task 6.** "Not Supported" or "N/A" means the capability doesn't exist on this GPU or driver model, which is not a fault. "Insufficient Permissions" means you need root or the driver model forbids it. An actual fault looks like `Unable to determine the device handle`, `ERR!` in a field, or XID messages in `dmesg`.

## Exam traps

- Don't confuse the driver's CUDA version with the toolkit version (`nvcc --version`).
- `nvidia-smi` working inside WSL proves only that the Windows driver works. It says nothing about the Container Toolkit (Lab 02).
- On a DGX, settings changed with `nvidia-smi` (power limit, application clocks) are **not persistent** across reboots unless a service or the cluster manager re-applies them.
