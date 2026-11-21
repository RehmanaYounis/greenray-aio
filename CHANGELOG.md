# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/).

## [1.7.0] - 2026-11-22

### Changed
- The mock exam follows the real sequence reported by recent candidates: 35 questions in a first section, a submit step after which the questions cannot be reopened, then the three labs on the same 120-minute clock. The question section can be submitted with unanswered questions after a confirmation that says how many count as wrong; the lab section keeps its menu, `f` to finish and `x` to leave. `greenray mock --labs-only` is unchanged. The CLI, menu and documentation describe the new shape.

## [1.6.1] - 2026-11-22

### Fixed
- Lab `tsh-1` (`tsh-first-look-at-logs`): the accounting criterion is met only when a successful `sacct` or `scontrol show job` query actually returns the incident job, so an empty query such as `sacct -j 999999` earns no credit; the ticket criterion no longer names the cause in its wording before the lab is solved. Scenario checks can now use `printed(pattern, cmd=...)` to look at what a command printed, not only that it ran.
- `sacct -j` and `sacct --jobs` without a job id report that the option requires an argument instead of listing every job; `journalctl -u`, `--unit` and `-xeu` without a unit do the same instead of printing the whole journal.

## [1.6.0] - 2026-11-22

The shell and Slurm behave like the real ones in the places a candidate notices in the first minutes of a lab.

### Added
- Shell: variables (`X=1`, `export X`, `$X`, `${X:-default}`, `${X:=default}`, `$((2 + 3))`, `$HOME`, `$USER`, `$HOSTNAME`, `$PWD`, `$OLDPWD`), `$?` with the exit status of the last pipeline, `cd -`, `env VAR=x cmd`, globbing with `*`, `?` and `[...]` (an unmatched pattern stays as typed, a quoted one is never expanded), redirections `2>`, `2>>`, `2>&1`, `&>`, `&>>` and `<`, quoted heredoc markers with a redirect after them (`cat <<'EOF' > file`), `sudo -u USER cmd`, and `ssh NODE 'cmd > file'` running the whole quoted line on the node.
- Slurm: job arrays (`--array=1-4%2`, `0,3,5`, `1-10:2`; `1002_3` ids in `squeue`, `sacct`, `scontrol show job` and `scancel`; `squeue -r`; `%A` and `%a` in output names; `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT`); `--begin` with the `BeginTime` reason and `EligibleTime`; `--mem`, `--mem-per-cpu` and `--mem-per-gpu` checked against the nodes and accounted when placing jobs (`ReqTRES mem=`, `MinMemoryNode`, `sacct ReqMem`); `--test-only`; `--wait`; `sbatch` reading the script from stdin; a job that reaches its time limit ends as `TIMEOUT` with the controller's journal line, and `--wrap='sleep N'` runs for N seconds; `salloc` opens an allocation shell with `SLURM_*` set that `exit` relinquishes, or runs the command it is given and releases; `--dependency` on an unknown job and `-x` leaving fewer nodes than `-N` are refused as slurmctld refuses them, `singleton` works; `--mail-type`, `--mail-user`, `--tasks-per-node`, `--ntasks-per-core`, `--threads-per-core`, `--oversubscribe`, `--spread-job`, `--gpu-bind`, `--open-mode`, `--priority` (root) and `--kill-on-invalid-dep` are accepted and shown.
- Reservations: `scontrol create|update|delete reservation` and `show reservation`; a node in an active `MAINT` reservation shows `maint` in `sinfo` and holds other jobs with `ReqNodeNotAvail, Reserved for maintenance`; `scontrol suspend`, `resume` and `top`.
- Permissions: a plain user is refused `scancel` on another user's job, `scontrol update|create|delete|reconfigure|reboot`, `sacctmgr add|modify|delete` and `cmsh` (a BCM user with the `admin` profile may run it); drains and `sacct` records carry the user who acted.
- `scontrol reconfigure` rejects a `slurm.conf` with an unknown key or a partition of unknown nodes the way slurmctld does, and keeps the previous configuration.
- Commands: `pdsh` (`-w`, `-x`, `-g category`, `-a`), `ibdev2netdev`, `lsblk`, `last`, `ipmitool chassis status`; `kubectl get rs|crd`, `kubectl explain`, `kubectl config get-contexts|current-context|view`, `kubectl set image`, `kubectl expose`, `kubectl rollout history|undo` with revisions kept per deployment, services as real objects; `nvidia-smi mig -lcip`; `dcgmi nvlink -s|-e`, `topo`, `fieldgroup -l|-c|-d|-i`, `profile -l|--pause|--resume`; `nvsm show gpus|storage|power|fans|version`; `docker system df|prune`, `docker volume ls|create|rm|inspect|prune`, `docker network ls|inspect`.
- Node files: `/etc/hosts`, `/etc/os-release`, `/proc/uptime`, `/proc/loadavg`, and `/proc/driver/nvidia/version` with the `/dev/nvidia*` device nodes, present only while the driver is loaded.

### Changed
- Words on a command line are no longer rewritten into paths unless the command takes files or the word looks like a path, so `jq .`, `echo data` and `kubectl get ns slurm` mean what they say.
- Error text is stderr: it reaches the terminal, not a pipe or a redirected file, unless `2>&1` or `2>` says so.
- `cmsh -c` discards what was staged and not committed when the invocation ends, with a warning, as the real client does; `commit` belongs in the same script as the change.
- A job without a memory request takes no memory from the node's accounting; jobs with one are placed against `RealMemory`.

## [1.5.1] - 2026-11-18

Three read-only audits drove the simulator the way a learner would and compared it with the real tools; about ninety findings were reproduced and are fixed here, each with a test.

### Fixed
- Inputs that crashed the simulator now get the tool's own error: malformed numbers and times for `sbatch`, `srun`, `salloc`, `scontrol update`, `sacctmgr` (a bad QOS value used to be stored and then crash every later `sbatch`), cmsh `bmcsettings`, `kubectl create ns`, `--field-selector`, `jsonpath`, `scale --replicas`, `helm rollback`, `dcgmi dmon`, `head`, `tail`, `sed`; cmsh `pexec`; removing a category or image that devices still use is refused; the GPU Operator DaemonSets print as YAML.
- Scenario grading: `tsh-first-look-at-logs` no longer passes a ticket that says "room"; `sacct -j` shows other users' jobs; `tsh-which-gpu-is-which` accepts the CSV report its own hint suggests; state names in drain and resume checks are case-insensitive and every reset path counts; evidence checks that any `cat`, a bare `nvidia-smi` on the head or a `kubectl` in the wrong namespace used to satisfy now need the real evidence; a `show` before a `set` inside one `cmsh -c` script counts as reading first; `#SBATCH -G 1`, several options on one `#SBATCH` line, any key order in a `nodeSelector`, `nvidia-smi -q -i N` and `dpkg -l | grep` are accepted as the correct alternatives they are; "slurm.conf was not edited by hand" compares the file with what cmd generates; two checks that were true before any work was done now need the work.
- Text: stray quotes and a wrong pronoun in four briefings, job times in `tsh-xid79-drained-node`, a hint in `wkl-slurm-pending-reasons` that said to raise a limit the check needs lowered, four last hints that handed over the full fix command; the xid79 solution collects `nvidia-bug-report.sh` before the reboot, as question tsh-001 keys.
- Questions: eight `lab:` links pointed at scenarios that do not teach the point; five near-duplicates removed (181 questions); tsh-015 no longer gives away tsh-017.
- App: Ctrl-C or end of input at a yes/no prompt in a lab, the mock or the menu declines instead of leaving the program and losing the attempt; a mock that runs out of input ends; a drill with nothing answered records nothing; `-n` and `-m` must be positive; a labs-only mock says it has no questions instead of recommending `q`; old or odd `progress.json` files are read without an "Unknown scenario" message; the debrief no longer promises `retry` in guided mode and describes the ordering checks correctly; checks that were already true when a lab opened are shown dim as "true from the start".
- One clock: `date`, `uptime`, the login banner and `docker ps` now sit on the same day as Slurm, kubectl, cmsh and the scenario journals (19 October 2026).
- Slurm output: `scontrol -o`, options anywhere on the line, `OverSubscribe` and `AllowGroups` on partitions; `sacct -b`, `-l`, `-S`, `-E`, the `NCPUS`, `ReqMem` and `ReqNodes` fields, and root sees every user as the help says; `squeue -O`; `%N` in output names is the node the job ran on; `srun -c N nproc` prints N and `bash -c` steps see `SLURM_PROCID`, `SLURM_LOCALID` and `CUDA_VISIBLE_DEVICES`; a missing command in a batch script fails like bash with exit code 127; `scancel -s SIGNAL` and `--me`; `srun --mpi=list`; `sacctmgr` options anywhere, `-P` without a trailing bar, `show tres`, the job ids that block a QOS delete.
- cmsh: IP and MAC addresses are validated and duplicate IPs refused; `set hostname` renames the node; `drain --reason=x` and a real drain timestamp; `slurmsubmit` and `slurmaccounting` roles; `cmsh -h` and `man cmsh` list every mode; DGX nodes list `nvidia-fabricmanager`, `datacenter-gpu-manager` and `nvsm` in `dpkg -l`; `kubectl -n NS VERB` works.
- Text tools honour their options: `grep -o -n -l -w -H -h -e -F` and `-c` exits 1 without a match, `sort -n -u -k -t`, `wc -w -c`, `cat -n`, `ls -l -a` with bare names in the current directory and dotfiles hidden, `awk` guards, `-F`, `$NF`, `NR`, `BEGIN`/`END` and sums, `sed` with POSIX groups, escaped delimiters, `-n`, `-E`, line addresses and ranges, `tail -n +N`, `xargs -n1` and `-I`, old-style `tar czf`, `diff` change letters and `-` for stdin, `jq` pipes, `length`, `keys`, `.[]`, `select` and `-r`, and each tool's own error prefix.
- `journalctl -u -k -b -n -p --since -r` and `dmesg -l` filter; a healthy unit shows its start in the journal; `modprobe -r` refuses a module another loaded module uses.
- GPU tools: `nvidia-smi -rgc` resets the clocks, `-ac`/`-rac`, `pmon`, `--query-compute-apps`, `nvlink -e` and `-c`, `-e` validated, MIG placements as on an H100 with the successes printed before a failure; `dcgmi stats -j` after `-x` shows the end time, the built-in groups cannot be modified, `discovery -c`.
- Containers and network: `docker ps -q` and `--format`, `inspect` ids match `ps`, CREATED from the clock; `ngc registry image pull`, `--format_type json|csv`, the right wording for an unsupported entity; `ip route` and `ip link show DEV`; `helm list -o json|yaml` and a stable LAST DEPLOYED; `kubectl` no longer offers `go-template`.
- Help and man pages: `grep --help` lists what works; `man scontrol`, `man slurm.conf`, `man sacct` and `man salloc` exist; `nvidia-smi -h` lists the new options.

### Added
- Solutions may carry `expect:` lines naming what key steps must print; `greenray validate` and the test suite replay them, so a tool that goes quiet on a solution step is caught.

## [1.5.0] - 2026-11-17

### Added
- BCM monitoring in `cmsh`: `monitoring` mode with the `measurable`, `action` and `trigger` submodes. Measurables are read from the simulated nodes (GPU temperature, power, utilisation, ECC errors, load, memory, free space, and the PASS/FAIL health checks for managed services, dmesg, GPU health, disk space); the built-in actions are `drain`, `undrain`, `poweroff`, `poweron`, `reboot`, `killprocess` and mail, and script actions can be added. A trigger (`set expression 'gpu_temp > 85'`, `set enteractions drain`, `set leaveactions undrain`) is evaluated after every command and on commit: its enter actions run once when a node crosses the threshold and its leave actions once when it recovers, each firing recorded in the BCM event log and the cmd journal. Device mode gains `latestmetricdata` and `latesthealthdata`.
- `nvsm` on the DGX nodes: `show health`, `show versions`, `show alerts`, `dump health` and `update firmware --type nvswitch|bmc|bios|all`, with the firmware set modelled per node. An inconsistent NVSwitch firmware set takes half the NVLinks down, which `nvidia-smi nvlink -s` and the NCCL tests already show; an update is staged and applied by a reboot, and is refused while jobs run on the node unless forced (an audit event).
- Two scenarios: `adm-bcm-monitoring-hot-node` (adm-15, two variants: make BCM drain a hot node and undrain it when it cools) and `tsh-nvswitch-firmware-partial-update` (tsh-19, hard, two variants: one DGX runs NCCL at half speed after an interrupted firmware update). 53 scenarios in all.
- Eight questions on triggers, their actions and where firings are logged, `latestmetricdata`, script actions, and `nvsm`. 186 questions in all.

## [1.4.0] - 2026-11-17

### Added
- BCM-managed Slurm in `cmsh`: `wlm` mode with the Slurm instance and its `jobqueue` submode (`list`, `show`, `add`, `set nodes|maxtime|defaulttime|default|state|qos|allowqos`, `remove`, `commit`), with the checks the real tool makes on node lists, time limits and states. A commit regenerates the `NodeName` and `PartitionName` lines of `/etc/slurm/slurm.conf` from the model and reconfigures `slurmctld`, and so does `systemctl restart cmd` on the head node, so a partition added to the file by hand disappears the way it does on a real cluster.
- `configurationoverlay` mode: the five overlays `cm-wlm-setup` creates (`slurm-server`, `slurm-client`, `slurm-client-gpu`, `slurm-submit`, `slurm-accounting`) with `list`, `show`, `set priority`, `append`/`removefrom categories|nodes` and `commit`. A node is a Slurm client when an overlay with the `slurmclient` role covers it or its category; only `slurm-client-gpu` puts `Gres=gpu:h100:8` on its `NodeName` line.
- Bulk operations in device and category mode: `foreach -c <categories> (...)`, `foreach -n <hostlist> (...)` and `range -n <hostlist>` followed by `get`, `set`, `clear` or `commit` on the selection.
- `kernelmodules` submode of a software image (`list`, `add`, `remove`, `commit`); a commit rebuilds the image's initial ramdisk and the modules load when a node boots from the image, not before.
- `cm-wlm-setup` and `cm-kubernetes-setup` on the head node explain what the interactive wizards produced and where it lives in `cmsh`; `cm-kubernetes-setup --add-user` and `--list-users` work. `cmsh -c "main; events"` prints the BCM event log.
- Three scenarios: `adm-bcm-wlm-default-queue` (adm-13, a hand-added partition vanished when cmd restarted; two variants), `adm-bcm-wlm-new-queue` (adm-14) and `ins-bcm-overlay-new-category` (ins-10, a node moved to a new category is absent from `sinfo`). 51 scenarios in all.
- Twelve questions on BCM-managed Slurm, configuration overlays, the setup wizards, `foreach`, `kernelmodules` and node-level image overrides. 178 questions in all.

## [1.3.2] - 2026-10-26

### Changed
- One colour code for labs everywhere: easy green, medium yellow, hard red, in the lab list, `next` and the mock-exam menu. In the exam menu the questions entry is cyan and the two ways out are blue; a question is bold above its plain choices; what you type at an exam prompt is cyan.

## [1.3.1] - 2026-10-26

### Fixed
- The mock exam menu can be left without grading (`x`), says what it does with input it does not understand, and the questions say what `s`, `f` and `q` do. Ctrl-C inside a question returns to the exam menu, and at the menu it offers to leave, instead of ending the program without a record.
- A mock-exam lab that was never opened scores nothing and is reported as not attempted; before, its invariant checks ("defq is still up") counted as work.
- In the question review the whole correct and chosen lines are coloured, not only the mark, and explanations are blue rather than dim.

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
