# GreenRay AIO

[![CI](https://github.com/RehmanaYounis/greenray-aio/actions/workflows/ci.yml/badge.svg)](https://github.com/RehmanaYounis/greenray-aio/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/greenray-aio)](https://pypi.org/project/greenray-aio/)
[![Python](https://img.shields.io/pypi/pyversions/greenray-aio)](https://pypi.org/project/greenray-aio/)
[![Docs](https://img.shields.io/badge/docs-rehmanayounis.github.io-green)](https://rehmanayounis.github.io/greenray-aio/)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)

Hands-on lab simulator and mock exam for the NVIDIA-Certified Professional: AI Operations (NCP-AIO) exam.

GreenRay AIO drops you into a broken or half-configured four-node DGX H100 cluster managed by Base Command Manager and asks you to find out what is wrong and fix it, with the commands you would use on real systems: `nvidia-smi`, `dcgmi`, Slurm, `kubectl` and `helm` with the GPU Operator, `cmsh`, InfiniBand tools, Docker and NGC. It grades the end state of the cluster, not the commands you typed. No GPU, no cloud account, nothing but Python.

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

## Install

```bash
uv tool install greenray-aio        # or: pipx install greenray-aio
greenray next
```

Or with Docker, keeping your progress in a named volume:

```bash
docker run -it --rm -v greenray:/home/lab/.greenray ghcr.io/rehmanayounis/greenray-aio
```

Python 3.10 or newer on Linux, WSL, macOS or Windows. See [Getting started](https://rehmanayounis.github.io/greenray-aio/getting-started/) for the details and your first lab.

## How it works

- **45 scenarios** cover every topic of the exam blueprint ([coverage map](https://rehmanayounis.github.io/greenray-aio/coverage/)), from a six-lab on-ramp for the basics to multi-node incidents.
- **Guided or incident mode.** Guided shows the objectives and hints. Incident mode shows only the ticket; the root cause stays hidden until the debrief, which then gives you the reference solution, the edge cases and your own commands side by side.
- **A grader that reads state.** Any valid fix passes. Evidence checks confirm you diagnosed before you changed things.
- **A study path.** `greenray next` says what to practise, `greenray plan` lays out four weeks, and every mock exam ends with a report card.
- **A mock exam in the current format**: 30 questions plus 3 labs on Slurm, Kubernetes and BCM in 120 minutes, and a labs-only mode for pacing.
- **A terminal that behaves like a node**: `--help` and `man` pages, your real editor for `nano` and `vim`, tab completion, history, `watch`, live `dmon` samples.

## Documentation

The full manual is at [rehmanayounis.github.io/greenray-aio](https://rehmanayounis.github.io/greenray-aio/):

- [Getting started](https://rehmanayounis.github.io/greenray-aio/getting-started/): install and your first ten minutes
- [Using the lab](https://rehmanayounis.github.io/greenray-aio/usage/): every command, the lab meta-commands, what the grader judges, settings
- [Study path](https://rehmanayounis.github.io/greenray-aio/study-path/): the on-ramp, `next` and `plan`, the mock exam and its report card
- [The simulated cluster](https://rehmanayounis.github.io/greenray-aio/cluster/): hosts, tools and limits
- [Writing scenarios](docs/scenario-format.md), [Architecture](ARCHITECTURE.md), [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), [Changelog](CHANGELOG.md)

## Licence and trademarks

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

GreenRay AIO is an independent study tool. It is not affiliated with, endorsed by or sponsored by NVIDIA. NVIDIA, DGX, Base Command Manager, NCP-AIO and related names are trademarks of NVIDIA Corporation, used only to describe the certification this tool helps you prepare for. The scenarios and questions are original work written from public documentation; nothing is taken from the exam.
