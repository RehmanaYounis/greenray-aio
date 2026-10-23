# Architecture

GreenRay AIO is a single Python process: a simulated cluster, a shell that runs
emulated tools against it, scenario files that break and grade it, and a terminal app
on top. Nothing talks to real hardware or the network.

```
 learner ──▶ app (menus, labs, exams, debrief; rich + prompt_toolkit)
               │  one line at a time
               ▼
           core/shell.py   bash-like parsing: pipes, &&, redirects, heredocs, $(…), ~
               │  argv
               ▼
           commands/*      one module (or package) per tool family, registered with @command
               │  read / mutate
               ▼
           core/state.py   the cluster: nodes → GPUs, IB ports, services, files; slurm, k8s, bcm …
               ▲
           content/        scenarios (break the state, grade it), questions, help pages, guides
```

## The state tree
`core/state.py` builds a healthy 4-node DGX H100 cluster as one tree of dict-like
`AttrDict`s. Everything a tool can show or change lives there, so a command is just a
function of `(ctx, args)` over that tree. The tree is small enough to pickle per command;
the shell uses that to roll back a crashed emulator without ending the lab.

## Commands
`commands/__init__.py` holds the registry. A tool raises `CommandError` with the real
tool's wording for user mistakes; any other exception is a simulator bug and is rolled
back and reported. Tools only colour output when `ctx.colorize` is true (straight to a
colour terminal), and only stream samples when `ctx.streaming` is true, so pipes, tests
and files get plain text. Service and reboot hooks let one tool react to another
(`systemctl restart slurmd` re-reads `gres.conf`).

Large families (`slurm`, `kubectl`, `container`, `cmsh`) are packages with one module per
concern and an `__init__` that re-exports what other tools use. `commands/common.py` is the
toolkit they share: option lookup, errors and the two plain table layouts.

## Scenarios and grading
A scenario is a YAML file: `setup` ops applied to the healthy tree, `checks` as Python
expressions over the resulting tree, hints, and a reference solution kept separately.
Checks grade the end state, so any valid fix passes; `ran()` requires that evidence was
gathered, `ran_before()` that it came before a change, and `audit()` reads the log of
destructive operations that commands record in `core/audit.py` (a reboot with jobs on
the node, another user's job cancelled). A scenario may list variants; placeholders in
the problem and the solution are filled per variant. `content/validate.py` proves every
variant starts broken and that its solution fixes it, and that no hint quotes a fix.

## Trust boundary
Scenario files are code (`eval` for checks, `exec` for `python:` setup). They come only
from this repository and the solutions source. What the learner types never reaches
`eval`/`exec`; it is parsed by the emulated tools.

## App
`app/lab.py` runs an attempt (guided or incident mode), the debrief and the offer to reflect;
`app/reflect.py` keeps those answers as notes. `app/exam.py` and
`app/mock.py` run question drills and the mock exam (full or labs-only) and print the report
card. `app/study.py` turns progress into a recommendation (`next`) and a four-week schedule
(`plan`); `app/progress.py` stores results in `~/.greenray`. `app/ui.py` renders panels and
tables with rich and reads lines through `app/console.py` (history, tab completion of
commands, hosts and simulated paths).
