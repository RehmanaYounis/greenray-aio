# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/).

## [1.3.1] - 2026-10-26

### Fixed
- The mock exam menu can be left without grading (`x`), says what it does with input it does not understand, and the questions say what `s`, `f` and `q` do. Ctrl-C inside a question returns to the exam menu, and at the menu it offers to leave, instead of ending the program without a record.

## [1.3.0] - 2026-10-23

### Changed
- Merging into `main` is the release: the workflow tags the version in `pyproject.toml` and publishes it, with that version's changelog section as the release notes. Tags are no longer pushed by hand.

### Added
- The everyday Linux tools on every node: `less`, `cut`, `uniq`, `tr`, `printf`, `find`, `mv`, `ln`, `rmdir`, `which`, `file`, `stat`, `du`, `basename`, `dirname`, `realpath`, `date`, `sleep`, `seq`, `xargs`, `test`, `expr`, `diff`, `cmp`, `history`, `bash script.sh`, `sudo`, `tar`, `gzip`, `jq`, `nproc`, `htop`, `pgrep`, `killall`, `w`, `who`, `lsof`, `fuser`, `vmstat`, `hostnamectl`, `timedatectl`, `modinfo`, `ss`, `netstat`, `ifconfig`, `ethtool`, `nslookup`, `dig`, `traceroute`, `curl`, `wget` and `shutdown`. They read the simulated node, so `ss -tlnp` lists the services that are running, `curl` reaches the API server only when it is up, `lsof /dev/nvidia0` names the job's processes and `shutdown` powers the node off with the same audit record as a power-off from BCM.

## [1.2.2] - 2026-10-23

### Fixed
- `srun --pty bash` inside an existing allocation opened no shell and reported "execve(): bash: No such file"; it now moves you onto the allocated node, and `exit` returns to the allocation without ending it.
- `salloc` accepted `--pty`, which is an `srun` option; it is rejected as the real tool does.

### Added
- `top`, as a one-shot snapshot of the node's load and processes.

## [1.2.1] - 2026-10-23

### Changed
- The container tools and `cmsh` are packages with one module per concern, like Slurm and kubectl, and the option, error and table helpers that four modules each carried are one shared toolkit. No behaviour changes; the golden-output and option suites were the check.

## [1.2.0] - 2026-10-23

### Added
- A test that runs every option listed on every help page and fails when one changes neither the output nor the cluster. The help pages already promised to simulate the options they list; the suite now holds the simulator to it, with the handful of genuine no-ops listed with their reasons.
- An issue template for reporting a simulated command that differs from the real tool.
- `mpirun` honours `--map-by`, `--rank-by`, `--bind-to`, `--display-map`, `--report-bindings`, `--tag-output`, `--oversubscribe`, `--prefix` and `-wdir`.
- `sbatch`, `srun` and `salloc` keep and validate the rest of a submission: `--comment`, `--constraint` against node features, `--reservation`, `--hint`, `--distribution`, `--mem-bind`, `--mpi`, `--signal`, `--mem`, `--time-min`, `--sockets-per-node`, `--cpus-per-gpu`, `--ntasks-per-gpu`, `--requeue` and the pyxis flags; `-v` prints the defined options and `salloc -Q` is quiet. `scontrol show job` shows the comment, features, reservation, memory, minimum time and requeue setting. `srun -o` writes the step's output to the file.

### Fixed
- `kubectl scale` with no arguments crashed; `-f=`, `-n=` and the other short options with `=` were parsed wrongly; `patch --type=json` with an object patch crashed; a name together with `--all` or a selector was accepted by `delete`, `wait`, `label` and `taint`; `logs` and `exec` accepted a container name the pod does not have; `run --rm` without `-i` was accepted.
- `helm --kube-context` with an unknown context was ignored.
- `sacct` showed every user's jobs without `-a`.
- `uname` printed everything without `-a`; `mkdir` created missing parents without `-p`; `ping -c` was ignored; `modprobe --remove`, `runuser --user`, `taskset --pid` and `--cpu-list`, `mlxconfig -e` on ConnectX-7, `munge -n`, `numactl -N`/`-m`/`-C`/`-i`/`-p` with a space and `showmount` without `-e` were not honoured.

## [1.1.0] - 2026-10-23

### Added
- Short lab codes. Every scenario has a code such as `tsh-1` next to its id, shown by `greenray list`, `next`, `plan` and the report card, and accepted wherever an id is: `greenray lab tsh-3`, `greenray incident --id adm-2`, the menu. Codes are fixed for the life of a scenario; `greenray validate` checks them.
- The Labs menu starts a lab: after the list, type a code or id.
- Golden-output tests for `nvidia-smi`, `sinfo`, `squeue`, `sacct`, `kubectl get`, `cmsh device list` and `ibstat` on the healthy cluster, so a change to a tool's layout is a reviewed decision rather than an accident.

### Fixed
- `nvidia-smi -i N` printed every GPU in the status table; it now shows only the selected ones, including `-i 2,5`, and the MIG and process sections follow the selection.
- The `nvidia-smi` status table was ragged: the per-GPU rows were shorter than the borders. Every line is now the same width, and the MIG devices table has the real tool's columns.

## [1.0.1] - 2026-10-23

### Fixed
- The documentation site serves the demo recording itself; the previous source only played on github.com.

## [1.0.0] - 2026-10-23

The first stable release. The scenario format is now a commitment: a change that breaks existing scenario files bumps the major version.

What 1.0 contains, for someone arriving fresh: a simulated four-node DGX H100 cluster managed by Base Command Manager with emulated `nvidia-smi`, DCGM, Slurm, Kubernetes with the GPU Operator, `cmsh`, InfiniBand tools, NCCL tests, Docker and NGC; 48 scenarios with reference solutions covering every topic of the NCP-AIO blueprint, from a six-lab on-ramp to cross-system incidents, most with variants; a grader that reads the cluster's end state, the evidence gathered and the safety of the work; guided and incident modes; a study path (`next`, `plan`), a mock exam in the current format with a report card, and reflection notes; installation from PyPI or as a Docker image; a documentation site.

### Added
- A demo recording on the README and the documentation home page.

## [0.9.3] - 2026-10-22

### Changed
- After every objective is met, the lab offers the debrief and then the reflection instead of printing both at once, so the result stays on screen. Enter accepts, `n` skips.
- Documentation reviewed against the current behaviour: updating instructions, the menu, the offered debrief, safe practice and variants in the README and the architecture page.

## [0.9.2] - 2026-10-22

### Fixed
- A lab started without `--variant` showed the unfilled briefing, objectives and hints (`{node}`) while grading a concrete variant. The lab now shows the variant it grades, and `retry` shows the next one.
- Scenario titles no longer carry placeholders, so `greenray list` and `greenray next` read correctly.
- The test-suite drives every variant of every scenario through the lab interface, in guided and incident mode, exactly as a learner types it, and fails on any placeholder that reaches the screen.

## [0.9.1] - 2026-10-22

### Fixed
- The interactive menu answers unknown input instead of silently reprinting itself, accepts `help`, the entry names and commands pasted from the docs (`greenray next`, `lab <id>`), and tells a first-time user where to start.

## [0.9.0] - 2026-10-22

Where to look, not only what to fix.

### Added
- Three cross-system incidents, each with three variants and in the mock-exam pool: a DGX node without a driver after a reboot because its BCM category changed; a two-node training run eight times slower because the updated software image lost the GPUDirect RDMA module and one InfiniBand link trained at 100G; a Deployment replica Pending because a pool node was switched to MIG.
- Reflection after a passed lab: three ungraded questions (the evidence, the choice of fix, the verification) whose answers are kept in `~/.greenray/notes/<scenario-id>.md`. `greenray progress` counts them.

### Changed
- Reseating a cable restores a link that trained at reduced speed, as a replaced cable does.

## [0.8.0] - 2026-10-22

Tests judgment, not memory.

### Added
- Scenario variants: a scenario can list variants that change the node, GPU, job or user. A lab picks one at random, `greenray lab <id> --variant N` chooses, `retry` moves to the next, and `greenray validate` replays every variant. Ten scenarios have three variants each.
- Safe-practice grading: the simulator records destructive operations that cost users work (a reboot or power-off with jobs on the node, a node set DOWN with jobs, another user's job cancelled, a running pod deleted, a container runtime restarted with containers up). Checks read that log with `audit()` and require order of work with `ran_before()`. Checks carry a kind (outcome, evidence or safety) that results display. Seven scenarios use them.
- A strictness table in the test-suite: invocations the real tools reject must fail in the simulator with the same wording.

### Fixed
- `scontrol update partitionname=... maxtime=` rejects malformed time limits; `sacct -j` rejects non-numeric job ids.
- `nvidia-smi -pm` rejects values other than 0/1 and `-lgc` rejects clocks outside the supported range.
- `modprobe` fails for modules the DGX image does not ship; `rm` fails for a missing file unless `-f` is given.
- `docker pull`, `docker run` and `enroot import` fail for Docker Hub repositories that do not exist.

## [0.7.0] - 2026-10-22

Ready for a public audience.

### Added
- A documentation site built with MkDocs and published to GitHub Pages: getting started, using the lab, the study path, the simulated cluster, exam coverage, the scenario format, architecture, contributing, security and this changelog.
- CI builds the site on every push; Dependabot groups GitHub Actions updates into one pull request.

### Changed
- The README is a front page: what the tool is, how to install it, how it teaches, and links into the documentation. The manual moved to the site.
- `docs/COVERAGE.md` and `docs/SCENARIO_FORMAT.md` are now `docs/coverage.md` and `docs/scenario-format.md`.

### Removed
- Compatibility with the project's former name: the `AIOLAB_*` environment variables, the `~/.aiolab` progress migration and the old solutions-directory names.
- Bundled solutions for real-hardware labs that are not part of this repository.
- The code of conduct file; its content rule lives in CONTRIBUTING.

## [0.6.0] - 2026-10-21

Teaches better.

### Added
- Six on-ramp scenarios for the first week: reading Slurm accounting and node logs, editing slurm.conf and reconfiguring, a first GPU batch job, a Pending pod with a wrong node selector, a read-only cmsh inventory, and mapping a GPU's bus id to its index, UUID and serial.
- `greenray next` recommends what to practise from your progress; `greenray plan` prints a four-week schedule covering every scenario, with passed labs ticked.
- `greenray mock --labs-only`: the three labs alone in 40 minutes. Every mock now ends with a report card: per-domain bars over questions and labs, and a practise-next list.
- A code of conduct and Dependabot configuration.

### Changed
- Hints name the tool and the setting instead of quoting the fix command; `greenray validate` and the test-suite reject a hint that quotes a state-changing solution step.
- On-ramp scenarios are never picked as mock-exam labs.
- An external solutions directory overrides the bundled solutions file by file.
- `nvidia-smi -q` prints the PCI block (bus id) and the minor number.

## [0.5.1] - 2026-10-21

### Fixed
- The release workflow pushed the Docker image under a mixed-case name, which GHCR rejects.

## [0.5.0] - 2026-10-21

Installs anywhere and ships complete.

### Added
- Reference solutions and exam answer keys ship with the package; an external solutions directory still overrides them.
- Apache-2.0 licence and a NOTICE with the NVIDIA non-affiliation statement.
- PyPI publishing through trusted publishing, and a multi-arch Docker image on GHCR, both built by the release workflow.
- `greenray doctor` prints the version, environment, editor and solutions source.

## [0.4.0] - 2026-10-21

Feels like a real terminal.

### Added
- Line editing on every OS (prompt_toolkit): persistent history, Ctrl-R search, and tab completion of commands, hosts and the simulated node's paths relative to the working directory.
- A simulated clock: `uptime`, `dmesg -T` and the login banner show the current date and time, and time passes as you work.
- `watch -n N <command>` refreshes on a terminal, `clear`, and `nvidia-smi dmon` / `dcgmi dmon` print samples live with real delays.
- `--no-color` flag; the mock exam shows time left, questions answered and the current lab in the prompt.
- `ARCHITECTURE.md`.

### Changed
- `commands/slurm.py` and `commands/kubectl.py` are now packages split by concern; their public names are unchanged.
- Exam answers are kept out of the persistent history.

## [0.3.0] - 2026-10-21

Renamed to **GreenRay AIO** (formerly `aiolab`). Adds editors, realistic help, terminal realism and CI.

### Added
- `greenray` command and package (the `AIOLAB_*` environment variables and `~/.aiolab` progress still work).
- Text editors `nano`, `vi` and `vim`: your real editor opens on a terminal, with a line editor as fallback.
- Realistic `--help`, `help <sub>` and `man` pages for 94 commands, respecting each tool's own meaning of `-h`.
- Working `cd`/`pwd` with relative paths, an Ubuntu-style coloured prompt and a DGX-style login banner.
- Real-tool colours on the terminal (`systemctl`, `grep`, `ls`, `ip -br`, `dmesg`).
- Polished screens: banner, menu, incident ticket card, results table, progress bars, debrief panels.
- CI on Linux and Windows (Python 3.10 and 3.13) with lint, type checks, tests, coverage and a build.

## [0.2.0]

### Added
- Incident mode, `explain <command>`, guided vs independent progress, the full mock exam, and the reliability fixes (command rollback, realistic errors, Ctrl-C handling).
- 39 scenarios covering every NCP-AIO blueprint topic, and a 166-question bank.

## [0.1.0]

### Added
- First simulator core (as `aiolab`): cluster state model, bash-like shell, `nvidia-smi` with MIG, scenario engine with grading, timed question drills and progress tracking.
