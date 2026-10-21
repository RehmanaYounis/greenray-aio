# GreenRay AIO

**Hands-on lab simulator and mock exam for the NVIDIA-Certified Professional: AI Operations (NCP-AIO) exam.**

GreenRay AIO drops you into a broken or half-configured **4-node DGX H100 cluster managed by Base Command Manager (BCM)**. You investigate and fix it with the same commands you'd use on real systems:
- `nvidia-smi` and `dcgmi`
- Slurm
- `kubectl` and `helm` with the GPU Operator
- `cmsh`
- InfiniBand tools and NCCL tests
- Docker and NGC containers

`check` grades the end state of the cluster, not whether you typed a particular command.

- **39 incident and task scenarios** covering every topic of the exam blueprint ([docs/COVERAGE.md](docs/COVERAGE.md))
- **Incident mode**: only the incident report is shown, and the root cause stays hidden until the debrief
- **`explain <command>`** for learning tools as you need them, plus progressive hints and a debrief after every lab
- **Feels like a real node**: `--help` and `man` pages, `nano`/`vim` (your real editor), `cd` with tab completion of paths, command history, `watch`, live `dmon` samples, a coloured prompt and real-tool colours
- **Full mock exam** in the current format: 30 questions plus 3 hands-on labs in 120 minutes (per NVIDIA's exam page; re-check before booking)
- Runs anywhere Python runs: Linux, WSL, macOS or Windows. **No GPU needed.**

```
root@bcm-head:~# ssh dgx-03
root@dgx-03:~# dmesg | grep -i xid
[ 8812.402113] NVRM: Xid (PCI:0000:5d:00): 79, pid=0, GPU has fallen off the bus.
root@dgx-03:~# nvidia-smi
Unable to determine the device handle for GPU00000000:5D:00.0: Unknown Error
...
root@dgx-03:~# check
  PASS  you read the kernel log on dgx-03
  ....  2 more success criteria not met yet
```

> GreenRay AIO is an independent study tool. It is **not affiliated with, endorsed by or sponsored by NVIDIA**. NVIDIA, DGX, NCP-AIO and related names are trademarks of NVIDIA Corporation, used here only to describe the exam this tool helps you prepare for. Every node, GPU and log is simulated.

## Install

Pick one. A GPU is **not** required.

**With uv or pipx** (any OS with Python 3.10+):

```bash
uv tool install greenray-aio      # or: pipx install greenray-aio
greenray
```

**With Docker** (nothing to install):

```bash
docker run -it --rm -v greenray:/home/lab/.greenray ghcr.io/rehmanayounis/greenray-aio
```

The named volume keeps your progress between runs.

**From source** (to contribute):

```bash
git clone https://github.com/RehmanaYounis/greenray-aio.git
cd greenray-aio && uv sync && uv run greenray
```

Use WSL rather than PowerShell on Windows for the best terminal experience. `greenray doctor` prints your environment when something looks wrong.

Developer commands (see [CONTRIBUTING.md](CONTRIBUTING.md)):

```bash
uv run pytest            # test suite (add --cov for coverage)
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type checks
uv build                 # wheel + sdist
docker build -t greenray-aio .
```

## Use

Installed with uv tool, pipx or Docker, the command is `greenray …`. From a source checkout use `uv run greenray …`.

| Command | What it does |
| --- | --- |
| `greenray` | Interactive menu (`--no-color` for plain output) |
| `greenray list` | All scenarios by exam domain, with your pass marks |
| `greenray incident [-d DOMAIN] [-t TOPIC]` | **Investigate a random incident.** Only the incident report is shown; the scenario's id, title and success criteria stay hidden until the debrief. `-t` takes a scenario tag such as `slurm`, `kubernetes`, `bcm`, `mig` or `infiniband` |
| `greenray incident --id <id>` | Retry a specific scenario as an incident (for a fresh independent attempt) |
| `greenray lab <id>` | Start a hands-on scenario in guided mode (title, objectives and hints visible) |
| `greenray sandbox` | A healthy cluster to explore freely |
| `greenray mock` | **Full mock exam:** 30 questions + 3 labs (one Slurm, one Kubernetes, one BCM) in 120 minutes, with no hints. Graded at the end |
| `greenray mock --labs-only` | The 3 labs alone in 40 minutes, for exam-day pacing. Every mock ends with a report card: per-domain bars over questions and labs, and what to practise next |
| `greenray exam` | Question drill: 60 questions in 120 minutes, weighted 31/23/23/23 like the blueprint |
| `greenray exam -n 20 -m 40 -d "Administration"` | Domain drill |
| `greenray progress` | Passed labs, exam history, weakest domain |
| `greenray next` | What to practise next: the on-ramp first, then your weakest domain, then guided passes to redo as incidents |
| `greenray plan` | A four-week schedule that covers every scenario, with your passes ticked off |
| `greenray validate` | Self-test: every scenario is solvable by its reference solution |
| `greenray doctor` | Version, Python, terminal, editor, solutions source and progress path |

Inside a lab, these meta-commands are available alongside the simulated cluster commands:

| Command | What it does |
| --- | --- |
| `task` | Show the briefing again |
| `check` | Grade your work. In incident mode, unmet criteria are counted but not named |
| `explain <cmd>` | What a command is for, its useful options, what to look for in its output, and common errors. It never describes a scenario's fix |
| `hint` | Show the next hint. Hints get more specific, and the last one asks for confirmation because it is close to the answer |
| `objectives` | Incident mode only: reveal the success criteria (this attempt then counts as guided) |
| `retry` | Incident mode only: start a fresh attempt (cluster, timer, hints and objectives reset) |
| `solution` | Give up and see the answer, explanation and edge cases |
| `reset` | Put the cluster back to the broken state. Same attempt: hints used and the timer carry over |
| `commands` | List the simulated tools |
| `quit` | Leave the lab |

When you pass, you get the **debrief**: the reference solution, why it works, the edge cases and exam traps, **your own commands** for comparison, and a related scenario to practise next. Read it even when you passed. Ctrl-C cancels the current line, as in bash.

### Guided or independent

`greenray progress` shows three counts per domain:
- **passed**
- **without hints**
- **as an incident, unguided**: passed in incident mode with no hints and no revealed objectives

A good routine is to learn a topic in guided mode first, then prove it with `greenray incident -t <topic>`.

### What the grader can and cannot judge

- It grades the **end state** of the cluster, so any valid fix passes. For example, `nvidia-smi -pm 1` and `systemctl enable --now nvidia-persistenced` are both accepted for persistence mode.
- Evidence checks confirm that you **ran** the diagnostic commands. Except where a check says otherwise, they do **not** check that you ran them before changing anything. Diagnose first anyway; the debrief reminds you.
- It does not generally detect unnecessary changes. Only some scenarios guard against specific harmful ones, such as raising power limits to hide a thermal fault, or killing a user's job.
- If a command ever crashes inside the simulator (a bug), the cluster is restored to its state before that command and you are told so.

Progress is stored in `~/.greenray/progress.json` on your machine (set `GREENRAY_HOME` to move it). Progress from the older `~/.aiolab` folder is copied over automatically the first time you run it.

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
- **Shell basics:** `cd`, `pwd`, `ls`, `cat`, `grep`, `sed -i`, heredocs, pipes and redirects; `nano`, `vi` and `vim` open your real editor; `--help` and `man` work as on a real node

`commands` inside a lab prints the full list.

**Limits:** outputs are realistic but simplified, and only the options that matter for the exam are implemented. If something doesn't parse, try the canonical form of the command. If a real command behaves differently from the simulator, trust the real one, and please open an issue.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Trust model

Scenario files are **code**: `checks` are Python expressions and `setup` may contain Python, both run on your machine. They come only from this package and, if you use one, your own solutions directory. Commands you type in a lab never reach `eval`/`exec`; they are parsed by the simulated tools. Only install scenario files you trust.

## Code layout

```
src/greenray/
  cli.py            command-line entry point (argparse)
  core/             cluster state model, bash-like shell, errors
  commands/         one module per tool family (nvidia_smi, slurm, kubectl, cmsh, ib, ...)
  content/          loaders: scenarios, questions, solutions, command guides, validation, domains
  app/              interactive layer: lab runner, question drill, mock exam, progress, UI
  data/scenarios/   scenario files (problem only)
  data/questions/   question bank (questions only)
  data/solutions/   reference solutions, answer keys and lab solutions
  data/command_guides.yaml   scenario-agnostic guides for `explain`
tests/              pytest suite, including a replay of every reference solution
docs/               SCENARIO_FORMAT.md, COVERAGE.md
```

## Adding scenarios and questions

- Scenarios: the problem goes in `src/greenray/data/scenarios/` and the solution in `src/greenray/data/solutions/scenarios/`. See [docs/SCENARIO_FORMAT.md](docs/SCENARIO_FORMAT.md). Add every new scenario to [docs/COVERAGE.md](docs/COVERAGE.md).
- Questions: these go in `src/greenray/data/questions/`, with the answer key in `src/greenray/data/solutions/questions/<same file>`. The format is documented in `src/greenray/content/questions.py`.
- Run `uv run greenray validate` and `uv run pytest` before committing. CI runs lint and tests on every push.

All content is original and written from public documentation concepts. It is not copied from the exam.

## Related

- Real-hardware practice that complements the simulator: the RTX 4060 / WSL2 labs and the rented-GPU cloud session kit live in the companion study repository, NVIDIA-NCP-AIO-Prep (`labs/gpu-local`, `labs/cloud-gpu`).
