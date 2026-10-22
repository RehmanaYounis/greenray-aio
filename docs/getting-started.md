# Getting started

## Install

Pick one. A GPU is not required, and nothing is installed system-wide.

=== "uv or pipx"

    Any OS with Python 3.10 or newer.

    ```bash
    uv tool install greenray-aio      # or: pipx install greenray-aio
    greenray
    ```

=== "Docker"

    Nothing to install. The named volume keeps your progress between runs.

    ```bash
    docker run -it --rm -v greenray:/home/lab/.greenray ghcr.io/rehmanayounis/greenray-aio
    ```

=== "From source"

    For contributors.

    ```bash
    git clone https://github.com/RehmanaYounis/greenray-aio.git
    cd greenray-aio
    uv sync
    uv run greenray
    ```

On Windows, use WSL rather than PowerShell for the best terminal experience. `greenray doctor` prints the version, Python, terminal, editor and progress path when something looks wrong; paste it into bug reports.

## Your first ten minutes

Ask what to do first:

```
$ greenray next
```

On a fresh profile that is the first on-ramp lab, where a user's job disappeared overnight and you find out why from Slurm accounting and the node's logs. Start it:

```
$ greenray lab tsh-first-look-at-logs
```

You land on the BCM head node as root. The briefing is a ticket; the objectives say what "done" means. A few things to know:

- Everything you type runs in the simulated cluster. `ssh dgx-02` moves you to a compute node, `exit` brings you back.
- `check` grades your work at any time. Check as often as you like.
- `hint` gives the next hint. `explain sacct` tells you what a tool is for and what to look for in its output.
- `help` lists the lab commands; `commands` lists the simulated tools.

When every objective is met you get the debrief: the reference solution, why it works, the edge cases, and your own commands next to it for comparison. Then run `greenray next` again.

## What to read next

- [Using the lab](usage.md) for every command and how grading works.
- [Study path](study-path.md) for the four-week route from the on-ramp to the mock exam.
