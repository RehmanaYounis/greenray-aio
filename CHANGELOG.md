# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/).

## [0.4.0] - unreleased

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
