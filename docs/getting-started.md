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

## Updating

```bash
uv tool upgrade --refresh greenray-aio     # or: pipx upgrade greenray-aio
docker pull ghcr.io/rehmanayounis/greenray-aio    # Docker
```

`--refresh` makes uv ask PyPI again instead of reusing its cached package index, which can lag a new release by a while. `greenray --version` shows what you have.

## Your first ten minutes

Ask what to do first:

```
$ greenray next
```

On a fresh profile that is the first on-ramp lab, where a user's job disappeared overnight and you find out why from Slurm accounting and the node's logs. Start it:

```
$ greenray lab tsh-first-look-at-logs
```

`greenray` on its own opens a menu instead: type the number of an entry, or its name, and `help` explains the commands. Either way you end up in the same labs.

You land on the BCM head node as root. The briefing is a ticket; the objectives say what "done" means. A few things to know:

- Everything you type runs in the simulated cluster. `ssh dgx-02` moves you to a compute node, `exit` brings you back.
- `check` grades your work at any time. Check as often as you like.
- `hint` gives the next hint. `explain sacct` tells you what a tool is for and what to look for in its output.
- `help` lists the lab commands; `commands` lists the simulated tools.

When every objective is met, the lab offers the debrief: the reference solution, why it works, the edge cases, and your own commands next to it for comparison. Press Enter to read it, and once more to answer three short questions about what you did; they are not graded, and your answers are kept as notes. Then run `greenray next` again.

## What to read next

- [Using the lab](usage.md) for every command and how grading works.
- [Study path](study-path.md) for the four-week route from the on-ramp to the mock exam.
