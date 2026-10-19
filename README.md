# aiolab: NCP-AIO cluster simulator and mock exam

> **Simulated environment.** Every node, GPU, switch and log in this tool is software. Results you get here count as *simulated* evidence in [PROGRESS.md](../PROGRESS.md), not real-hardware evidence.

`aiolab` gives you a broken (or half-configured) **4-node DGX H100 cluster managed by NVIDIA Base Command Manager**. You fix it by typing the same commands you would use on a real system. When you think you're done, `check` grades the end state of the cluster, not whether you typed a particular command.

The real exam is **30 multiple-choice questions plus 3 hands-on labs in 120 minutes**. The labs run on live Slurm, Kubernetes and BCM clusters. `aiolab mock` reproduces that shape, and every blueprint topic has a lab. See [docs/COVERAGE.md](docs/COVERAGE.md).

```
root@bcm-head:~# ssh dgx-03
root@dgx-03:~# dmesg | grep -i xid
[ 8812.402113] NVRM: Xid (PCI:0000:5d:00): 79, pid=0, GPU has fallen off the bus.
root@dgx-03:~# nvidia-smi
Unable to determine the device handle for GPU00000000:5D:00.0: Unknown Error
...
root@dgx-03:~# check
  PASS  you checked the kernel log before acting
  FAIL  dgx-03 is back in service (IDLE)
```

## Install

The project is managed with [uv](https://docs.astral.sh/uv/). uv installs the right Python version and the dependencies for you. A GPU is **not** required.

**WSL / Linux (recommended)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once
cd ~/NVIDIA-NCP-AIO-Prep/simulator
uv sync                                             # creates .venv from pyproject.toml + uv.lock
uv run aiolab
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # once
cd NVIDIA-NCP-AIO-Prep\simulator
uv sync
uv run aiolab
```

Use WSL for the best experience: you get arrow-key history and tab completion there.

### Solutions (private)

Reference solutions, explanations and exam answer keys are **not** in this repository. They live in the private repository `NVIDIA-NCP-AIO-Prep-solutions`. Clone it next to this one and the simulator finds it automatically:

```bash
cd ~ && git clone https://github.com/RehmanaYounis/NVIDIA-NCP-AIO-Prep-solutions.git
```

You can point at another location with `AIOLAB_SOLUTIONS=/path`. Without it, labs still run and grade your work. `solution`, the debrief and question scoring need it.

Developer commands:

```bash
uv run pytest            # test suite
uv run ruff check .      # lint
uv run ruff format .     # format
```

## Use

Run these as `uv run aiolab …` from the `simulator/` folder. If you activate the environment (`source .venv/bin/activate`), plain `aiolab …` works too.

| Command | What it does |
| --- | --- |
| `aiolab` | Interactive menu |
| `aiolab list` | All scenarios by exam domain, with your pass marks |
| `aiolab lab <id>` | Start a hands-on scenario |
| `aiolab sandbox` | A healthy cluster to explore freely |
| `aiolab mock` | **Full mock exam:** 30 questions + 3 labs (one Slurm, one Kubernetes, one BCM) in 120 minutes, with no hints. Graded at the end |
| `aiolab exam` | Question drill: 60 questions in 120 minutes, weighted 31/23/23/23 like the blueprint |
| `aiolab exam -n 20 -m 40 -d "Administration"` | Domain drill |
| `aiolab progress` | Passed labs, exam history, weakest domain |
| `aiolab validate` | Self-test: every scenario is solvable by its reference solution |

Inside a lab, these meta-commands are available alongside the simulated cluster commands:

| Command | What it does |
| --- | --- |
| `task` | Show the briefing again |
| `check` | Grade your work |
| `hint` | Show the next hint (hints mean the pass isn't "clean") |
| `solution` | Give up and see the answer, explanation and edge cases |
| `reset` | Start the scenario over |
| `commands` | List the simulated tools |
| `quit` | Leave the lab |

When you pass, you get the **debrief**: the reference solution, why it works, and the edge cases and exam traps around it. Read it even when you passed.

Progress is stored in `~/.aiolab/progress.json` on your machine (set `AIOLAB_HOME` to move it).

## The simulated cluster

| Host | Role |
| --- | --- |
| `bcm-head` | BCM head node (`cmsh`), Slurm controller, Kubernetes control plane, NGC CLI |
| `dgx-01` … `dgx-04` | DGX H100: 8x H100 80GB, NVSwitch, 8x ConnectX-7 NDR 400G, `slurmd`, `kubelet`, DCGM |

Simulated tools:

- **GPU:** `nvidia-smi` (queries, `-q -d`, power, persistence, ECC, reset, MIG, topology, NVLink)
- **Health:** `dcgmi`, `dmesg`, `journalctl`, `systemctl`, `ipmitool`
- **Slurm:** `sinfo`, `squeue`, `scontrol`, `sbatch`, `srun`, `salloc` (with pyxis), `sacct`, `sacctmgr`, `sprio`, `sshare`
- **Kubernetes:** `kubectl`, `helm` (GPU Operator, Network Operator), Run:ai CLI
- **Containers:** `docker`, `enroot`, `ngc`, `nvidia-ctk`, `crictl`
- **BCM:** `cmsh` (device, category, softwareimage, user, power, imageupdate, grabimage)
- **Fabric and Magnum IO:** `ibstat`, `ibdiagnet`, `mlxlink`, `perfquery`, NCCL tests (`all_reduce_perf`, `mpirun`), GPUDirect RDMA/Storage checks
- **DPU:** BlueField tools (`mlxconfig`, `bfb-install`, DOCA services)
- **Shell basics:** `cat`, `sed -i`, `grep`, heredocs, pipes and redirects for editing config files

`commands` inside a lab prints the full list.

**Limits:** outputs are realistic but simplified, and only the options that matter for the exam are implemented. If something doesn't parse, try the canonical form of the command. If a real command behaves differently from the simulator, trust the real one, and please open an issue.

## Code layout

```
src/aiolab/
  cli.py            command-line entry point (argparse)
  core/             cluster state model, bash-like shell, errors
  commands/         one module per tool family (nvidia_smi, slurm, kubectl, cmsh, ib, ...)
  content/          loaders: scenarios, questions, private solutions, validation, domains
  app/              interactive layer: lab runner, question drill, mock exam, progress, UI
  data/scenarios/   public scenario files (problem only)
  data/questions/   public question bank (questions only)
tests/              pytest suite; solution-replay tests skip when the private repo is absent
docs/               SCENARIO_FORMAT.md, COVERAGE.md
```

## Adding scenarios and questions

- Scenarios: the public problem goes in `src/aiolab/data/scenarios/`, and the solution goes in the private repo. See [docs/SCENARIO_FORMAT.md](docs/SCENARIO_FORMAT.md). Add every new scenario to [docs/COVERAGE.md](docs/COVERAGE.md).
- Questions: these go in `src/aiolab/data/questions/`, with the answer key in the private repo under `questions/<same file>`. The format is documented in `src/aiolab/content/questions.py`.
- Run `uv run aiolab validate` and `uv run pytest` before committing. CI runs lint and tests on every push.

All content is original and written from public documentation concepts. It is not copied from the exam.
