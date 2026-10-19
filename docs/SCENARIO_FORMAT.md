# Scenario format

A scenario has two files with the same name:

| File | Repository | Contents |
| --- | --- | --- |
| `src/aiolab/data/scenarios/<id>.yaml` | public (this repo) | Problem: briefing, objectives, setup (fault injection), checks (grading), hints |
| `scenarios/<id>.yaml` | private `NVIDIA-NCP-AIO-Prep-solutions` | Answer: reference solution, explanation, edge cases |

The loader refuses a public file that contains `solution`, `explanation` or `edge_cases`, so answers can't leak into the public repo by accident.

Prefix ids with the domain: `ins-`, `adm-`, `wkl-`, `tsh-`. The file name must equal the id.

## Public file

```yaml
id: tsh-example
title: Short, symptom-style title
domain: Troubleshooting and Optimization   # one of the 4 blueprint domains, spelled exactly
difficulty: medium                          # easy | medium | hard
time_limit_min: 15
tags: [slurm, xid]                          # `slurm`, `kubernetes` or `bcm` makes it eligible for `aiolab mock`
briefing: |
  What the user or ticket says. Describe symptoms, not the cause.
objectives:
  - What "done" means, in words (shown to the learner)
setup:                       # faults applied to a fresh healthy cluster, in order
  - set: nodes.dgx-03.gpus.5.xid          # dotted path; list indexes allowed
    value: 79
  - append: nodes.dgx-03.dmesg
    value: "[ 8812.402113] NVRM: Xid (PCI:0000:5d:00): 79, pid=0, GPU has fallen off the bus."
  - merge: slurm.nodes.dgx-03
    value: {state: DRAIN, reason: "healthcheck: GPU missing"}
  - file: /etc/slurm/gres.conf
    host: dgx-03
    content: "Name=gpu File=/dev/nvidia[0-3]\n"
  - python: |                 # escape hatch for complex setups; `state` is in scope
      for g in state.nodes['dgx-03'].gpus: g.temp_c = 80
checks:                      # Python expressions; all must be true to pass
  - desc: dgx-03 is back in service
    expr: slurm.nodes['dgx-03'].state == 'IDLE'
  - desc: you checked the kernel log first          # use ran() only for diagnosis evidence
    expr: ran(r'^dmesg', 'dgx-03')
hints: [gentle, more specific, nearly the answer]
references:
  - Public doc titles to read next
```

## Private file

```yaml
solution:                    # replayed by the test-suite; must make every check pass
  - ssh dgx-03
  - dmesg | grep -i xid
  - |                        # multi-line step: first line is the command, the rest is heredoc input
    cat > /etc/slurm/gres.conf <<EOF
    Name=gpu File=/dev/nvidia[0-7]
    EOF
explanation: |
  Why this is the fix, and what each piece of evidence meant.
edge_cases:
  - Exam traps, variations, and what changes on other hardware.
```

## Writing checks

These names are available in `checks` expressions: `state`, `nodes`, `slurm`, `k8s`, `bcm`, `ngc`, `dcgm`, `flags`. Helpers: `ran(regex, host=None)`, `file(host, path)`, `pods(namespace=None, prefix='')`, `release(name)`, `jobs(**fields)` and `re`.

- Grade the **end state**, so that any valid solution passes. Use `ran()` only to require that the learner gathered evidence before acting.
- State objects are dicts with attribute access. A key that clashes with a dict method (`values`, `items`, `keys`, `get`, `update`, `pop`) must be read with `obj['values']`, not `obj.values`. For Helm releases, use `release('x').computed`, which holds the chart defaults merged with user values.
- The state layout is documented in `src/aiolab/core/state.py`.

## Checklist for a new scenario

1. Add the id to `docs/COVERAGE.md` under its blueprint topic.
2. `uv run aiolab validate` should show `ok` for it.
3. `uv run pytest` should pass.
