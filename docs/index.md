# GreenRay AIO

Hands-on lab simulator and mock exam for the NVIDIA-Certified Professional: AI Operations (NCP-AIO) exam.

GreenRay AIO drops you into a broken or half-configured four-node DGX H100 cluster managed by Base Command Manager and asks you to find out what is wrong and fix it, with the commands you would use on real systems: `nvidia-smi`, `dcgmi`, Slurm, `kubectl` and `helm` with the GPU Operator, `cmsh`, InfiniBand tools, Docker and NGC. Nothing is real: no GPU, no network, no account. Everything runs in one Python process on your laptop.

<video controls preload="metadata" style="width: 100%; max-width: 900px; border-radius: 6px;">
  <source src="https://github.com/user-attachments/assets/16234b2c-1e5a-4718-abc8-5ab954ff79cd" type="video/mp4">
</video>

A ninety-second lab, start to finish: install, ask what to practise, diagnose a slow node, fix it, read the debrief.

```
root@bcm-head:~# ssh dgx-03
root@dgx-03:~# dmesg | grep -i xid
[ 8812.402113] NVRM: Xid (PCI:0000:5d:00): 79, pid=0, GPU has fallen off the bus.
root@dgx-03:~# nvidia-smi
Unable to determine the device handle for GPU00000000:5D:00.0: Unknown Error
root@dgx-03:~# check
  PASS  you read the kernel log on dgx-03
  ....  2 more success criteria not met yet
```

## Where to go

| | |
| --- | --- |
| [Getting started](getting-started.md) | Install or update in one command and finish your first lab in ten minutes |
| [Using the lab](usage.md) | Every `greenray` command, the lab meta-commands, and what the grader does and does not judge |
| [Study path](study-path.md) | The on-ramp, `greenray next`, the four-week plan, the mock exam and its report card |
| [The simulated cluster](cluster.md) | Hosts, tools and known limits |
| [Exam coverage](coverage.md) | Every blueprint topic mapped to the scenarios that practise it |
| [Writing scenarios](scenario-format.md) | The scenario file format, for contributors |

## What it is not

GreenRay AIO is an independent study tool. It is not affiliated with, endorsed by or sponsored by NVIDIA. NVIDIA, DGX, Base Command Manager, NCP-AIO and related names are trademarks of NVIDIA Corporation and are used only to describe the certification this tool helps you prepare for. The questions and scenarios are original work written from public documentation; nothing is taken from the exam.
