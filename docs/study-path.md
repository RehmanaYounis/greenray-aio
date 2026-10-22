# Study path

The exam tests whether you can operate a GPU cluster, not whether you can recite its documentation. GreenRay AIO is built around that: you learn each topic by fixing it, then prove it with the cause hidden, then rehearse the exam format against the clock.

## The on-ramp

Six ten-minute labs for the first week. They practise the basics every other scenario assumes, in this order:

1. `tsh-first-look-at-logs`: a job disappeared overnight. Slurm accounting and the node's `journalctl` and `dmesg` say why.
2. `tsh-which-gpu-is-which`: the BMC names a PCI bus id; hardware support wants an index, a UUID and a serial.
3. `adm-bcm-cmsh-inventory`: read a node's MAC, category, software image and kernel out of `cmsh`.
4. `wkl-slurm-first-gpu-job`: write, submit and read the output of a one-GPU batch job.
5. `adm-slurm-partition-time-limit`: edit `slurm.conf` and make the controller re-read it without touching running jobs.
6. `wkl-k8s-pod-pending-nodeselector`: a pod is Pending; its events and the node labels say why.

`greenray next` walks you through them in this order and moves on when they are passed.

## Guided, then unguided

Every scenario can be run two ways.

**Guided** (`greenray lab <id>`) shows the title, the objectives and hints on request. Use it the first time you meet a topic.

**Incident** (`greenray incident`) shows only the incident report: symptoms, impact and constraints, as a ticket would. The scenario's id, title and success criteria stay hidden, so nothing points at the cause. `check` tells you how many criteria are still unmet, not which. Pass one of these without hints and without revealing the objectives and it counts as an independent pass in `greenray progress`.

`greenray incident -t slurm` picks a random Slurm incident you have not yet passed independently; `-d` restricts the domain instead.

## `next` and `plan`

`greenray next` looks at your progress and recommends one thing, with the reason:

1. the next on-ramp lab, if any is unpassed;
2. otherwise the easiest unpassed lab, preferring your weakest domain from the last five exams;
3. otherwise a lab you passed with guidance, to redo as an unguided incident;
4. otherwise the mock exam.

`greenray plan` prints a four-week schedule, five study days a week, that covers every scenario once, with domain drills in the right weeks and mock exams in the last. Passed labs are ticked, so it doubles as a tracker. The pace is a suggestion; `next` always knows what is left.

## The mock exam

The real exam, per NVIDIA's certification page at the time of writing, is 30 multiple-choice questions plus 3 hands-on labs in one 120-minute, remotely proctored session, with the labs on live Slurm, Kubernetes and Base Command Manager systems. Re-check the official page before booking.

`greenray mock` reproduces that shape: 30 questions drawn in blueprint proportions, three labs (one Slurm, one Kubernetes, one BCM), one clock, no hints and no grading until you finish. Labs keep their state when you switch between sections. `greenray mock --labs-only` runs the three labs alone in 40 minutes for pacing practice.

NVIDIA does not publish how the two sections are weighted or the pass mark. The mock weights questions and labs 50/50 and uses a 70% practice target; treat the number as a guide, not a prediction.

Every mock ends with a **report card**: a bar per domain combining your questions and labs, and a practise-next list built from the labs that fell short, the labs linked from the questions you missed, and your weakest domain. The missed and flagged questions can be reviewed with their explanations straight away.

## A four-week outline

| Week | Focus |
| --- | --- |
| 1 | The on-ramp and the easy labs, guided. A first domain drill at the end. |
| 2 | Installation and Deployment, then Administration, guided. Domain drills on the way. |
| 3 | Workload Management, then Troubleshooting and Optimization, guided, ending with the cross-system incidents. |
| 4 | Everything again as incidents, `greenray mock`, the weakest domain, `greenray mock --labs-only`, a final mock. |

`greenray plan` prints the day-by-day version with your progress on it.
