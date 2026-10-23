# Using the lab

Installed with uv tool, pipx or Docker, the command is `greenray`. From a source checkout use `uv run greenray`. `greenray` on its own opens a menu: type the number of an entry or its name (`next`, `plan`, `lab <id>`, ...), and `help` explains the commands. `--no-color` (or the `NO_COLOR` variable) switches to plain output.

## Commands

| Command | What it does |
| --- | --- |
| `greenray next` | What to practise next: the on-ramp first, then the easiest unpassed lab in your weakest exam domain, then guided passes to redo as unguided incidents |
| `greenray plan` | A four-week schedule covering every scenario, with your passes ticked off |
| `greenray list` | All scenarios by exam domain, with your pass marks |
| `greenray lab <id>` | A scenario in guided mode: title, objectives and hints are available |
| `greenray lab <id> --variant N` | A specific variant of a scenario that has several (another node, GPU or job); without the flag one is picked at random |
| `greenray incident [-d DOMAIN] [-t TOPIC]` | A random scenario as an incident: only the incident report is shown, and the id, title and success criteria stay hidden until the debrief. `-t` is a scenario tag such as `slurm`, `kubernetes`, `bcm`, `mig` or `infiniband` |
| `greenray incident --id <id>` | A specific scenario as an incident, for a fresh independent attempt |
| `greenray sandbox` | A healthy cluster to explore; nothing is graded |
| `greenray mock` | The full mock exam: 30 questions and 3 labs (one Slurm, one Kubernetes, one BCM) in 120 minutes, graded at the end |
| `greenray mock --labs-only` | The three labs alone in 40 minutes, for exam-day pacing |
| `greenray exam [-n N] [-m MIN] [-d DOMAIN]` | A question drill: 60 questions in 120 minutes by default, weighted like the blueprint; `-d` restricts it to one domain |
| `greenray progress` | Passed labs per domain, exam history and your weakest domain |
| `greenray validate` | Self-test: every scenario starts broken, its reference solution fixes it, and no hint quotes a fix command |
| `greenray doctor` | Version, Python, terminal, editor, solutions source and progress path |

## Inside a lab

These meta-commands sit alongside the simulated cluster commands:

| Command | What it does |
| --- | --- |
| `task` | Show the briefing again |
| `check` | Grade your work. In incident mode, unmet criteria are counted but not named |
| `explain <cmd>` | What a command is for, its useful options, what to look for in its output, and common errors. It never describes a scenario's fix |
| `hint` | The next hint. Hints get more specific, and the last one asks for confirmation because it is close to the answer |
| `objectives` | Incident mode only: reveal the success criteria. The attempt then counts as guided |
| `retry` | Incident mode only: a fresh attempt. Cluster, timer, hints and objectives all reset |
| `solution` | Give up and see the answer, the explanation and the edge cases |
| `reset` | Put the cluster back to the broken state within the same attempt; hints used and the timer carry over |
| `commands` | List the simulated tools |
| `quit` | Leave the lab |

Ctrl-C cancels the current line, as in bash. Tab completes commands, hostnames and paths on the simulated node; the up arrow and Ctrl-R search your history.

When you pass, the lab offers the **debrief**: the reference solution, why it works, the edge cases and exam traps, your own commands for comparison, and a related scenario to practise next. Enter shows it; `n` skips it. Read it even when you passed.

After the debrief the lab offers three questions (Enter accepts, `n` skips): which evidence identified the cause, why your fix was safer than the alternative, and what you would verify before declaring recovery. They are not graded. Answer in a line or two, or press Enter to skip. Answers are appended to `~/.greenray/notes/<scenario-id>.md`, one file per scenario, so your own reasoning is there to reread before the exam.

## Guided or independent

`greenray progress` keeps three counts per domain:

- **passed**: all objectives met, in any mode;
- **without hints**: passed with no hints and no revealed objectives;
- **as an incident, unguided**: passed in incident mode with no hints and no revealed objectives.

A good routine is to learn a topic in guided mode first, then prove it with `greenray incident -t <topic>`.

## What the grader can and cannot judge

- It grades the **end state** of the cluster, so any valid fix passes. `nvidia-smi -pm 1` and `systemctl enable --now nvidia-persistenced` are both accepted for persistence mode.
- **Evidence** checks confirm that you ran the diagnostic commands, and where the order matters, that you ran them before changing anything.
- **Safe practice** checks catch the shortcuts an operations team would not accept: a reboot or power-off with a user's job still on the node, a node set DOWN instead of drained, another user's job cancelled, a running pod deleted, a power limit raised to hide a thermal fault. Results tag these rows so you can see what each criterion rewards. They count towards passing like any other objective.
- If a command crashes inside the simulator, which is a bug, the cluster is restored to its state before that command and you are told so. Please report it.

## Files and settings

Progress lives in `~/.greenray/progress.json`, command history in `~/.greenray/history`. The mock exam never writes answers to the history.

| Variable | Effect |
| --- | --- |
| `GREENRAY_HOME` | Directory for progress, history and notes instead of `~/.greenray` |
| `GREENRAY_SOLUTIONS` | A directory of reference solutions that overrides the bundled ones, file by file |
| `GREENRAY_DEBUG` | Show the Python error behind a failed check in `check` output |
| `NO_COLOR` | Plain output, the same as `--no-color` |
| `VISUAL`, `EDITOR` | The editor `nano`, `vi` and `vim` open inside a lab; otherwise the first of nano, vim, vi on your PATH, else a built-in line editor |
