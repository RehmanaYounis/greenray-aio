# Scenario format

A scenario has two files with the same name:

| File | Repository | Contents |
| --- | --- | --- |
| `src/greenray/data/scenarios/<id>.yaml` | problem | Problem: briefing, objectives, setup (fault injection), checks (grading), hints |
| `src/greenray/data/solutions/scenarios/<id>.yaml` | answer | Answer: reference solution, explanation, edge cases |

The loader refuses a problem file that contains `solution`, `explanation` or `edge_cases`, so the lab UI can always keep the answer hidden until it is asked for.

Prefix ids with the domain: `ins-`, `adm-`, `wkl-`, `tsh-`. The file name must equal the id.

Each scenario also carries a `code`, the short handle learners type (`tsh-3`). A new scenario takes the next free number in its domain; `greenray validate` rejects a duplicate, a gap or a prefix that differs from the id's. Codes are never reassigned, so a lab someone wrote down keeps meaning the same thing.

## Problem file

```yaml
id: tsh-example
code: tsh-19                                # next free number in the domain; never reused
title: Short, symptom-style title
domain: Troubleshooting and Optimization   # one of the 4 blueprint domains, spelled exactly
difficulty: medium                          # easy | medium | hard
time_limit_min: 15
tags: [slurm, xid]                          # `slurm`, `kubernetes` or `bcm` makes it eligible for `greenray mock`
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
  - desc: you checked the kernel log                 # evidence: ran() does not know order
    expr: ran(r'^dmesg', 'dgx-03')
  - desc: you read the log before rebooting          # ran_before() does
    kind: evidence
    expr: ran_before(r'^dmesg', r'^reboot', 'dgx-03')
  - desc: nothing was rebooted with a job still on it
    kind: safety
    expr: not audit('reboot_with_jobs')
hints: [gentle, more specific, nearly the answer]
references:
  - Public doc titles to read next
```

## Answer file

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

## Variants

A scenario may carry a `variants:` list so that a repeat is a different problem: another node, GPU, job id or user.

```yaml
variants:
  - {node: dgx-02, gpu: 3, bus: 5D, job: 1500, user: alice}
  - {node: dgx-04, gpu: 6, bus: BA, job: 2317, user: dave}
```

Every variant must have the same keys. `{node}`, `{gpu}` and the other names are substituted in the title, briefing, objectives, setup, checks and hints, and in the solution file's solution, explanation and edge cases. Only names that exist in the variant are replaced, so Python dict literals and regex quantifiers such as `\D{0,4}` in checks are left alone. A plain scalar in YAML may not start with `{`, so quote values that begin with a placeholder.

`greenray lab <id>` picks a variant at random, `--variant N` chooses one, and `retry` in incident mode moves to the next. `greenray validate` and the test-suite replay every variant.

## How scenarios are presented

- `greenray lab <id>` (guided) shows the title, briefing and objectives.
- `greenray incident` shows only the briefing, so write the **briefing as an incident report**: symptoms, impact, and every constraint the learner must respect (e.g. "don't kill bob's job"). It must never name the root cause.
- Objectives and check descriptions may name the cause. In incident mode they stay hidden until solved, or until the learner chooses `objectives`.
- Hints should go from a pointer to an area, to a diagnostic command, to (only in the last hint) something close to the fix. The last hint is shown only after confirmation. A hint may name the tool and the setting, but never quotes a fix command from the solution verbatim; `greenray validate` and the test-suite check this by replaying the solution and comparing every state-changing step with every hint.

## Writing checks

These names are available in `checks` expressions: `state`, `nodes`, `slurm`, `k8s`, `bcm`, `ngc`, `dcgm`, `flags`. Helpers: `ran(regex, host=None)`, `file(host, path)`, `pods(namespace=None, prefix='')`, `release(name)`, `jobs(**fields)` and `re`.

- Grade the **end state**, so that any valid solution passes, and accept every legitimate way to reach it.
- Use `ran()` to require evidence gathering. It does not know the order of commands; `ran_before(earlier, later, host=None)` does: both regexes must have matched, the first match of `earlier` before the first of `later`.
- `audit(kind)` lists destructive operations the simulator recorded while they cost users work: `reboot_with_jobs`, `power_off_with_jobs`, `node_down_with_jobs` (a node set DOWN, not drained, with jobs on it), `cancel_user_job` (scancel of another user's job), `delete_running_pod` and `runtime_restart_with_containers`. A check such as `not audit('reboot_with_jobs')` rewards draining first.
- A check may declare `kind: safety` or `kind: evidence`; a check whose expression uses `ran(` is evidence by default, everything else is an outcome. Results show the tag, so learners see what each criterion rewards. Every kind counts the same towards passing.
- A check that raises an exception counts as failed. The error text is shown only by `greenray validate` or with `GREENRAY_DEBUG=1`.
- Scenario files are trusted code (`eval` for checks, `exec` for `python:` setup). Never load them from untrusted sources.
- State objects are dicts with attribute access. A key that clashes with a dict method (`values`, `items`, `keys`, `get`, `update`, `pop`) must be read with `obj['values']`, not `obj.values`. For Helm releases, use `release('x').computed`, which holds the chart defaults merged with user values.
- The state layout is documented in `src/greenray/core/state.py`.

## Checklist for a new scenario

1. Add the id to `docs/coverage.md` under its blueprint topic.
2. `uv run greenray validate` should show `ok` for it.
3. `uv run pytest` should pass.
