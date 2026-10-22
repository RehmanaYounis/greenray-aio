# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/).

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
