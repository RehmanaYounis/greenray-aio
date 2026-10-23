"""Every option a help page lists must do something: change the output, change the cluster, or be rejected.

The help pages end with "GreenRay AIO simulates the options listed above", which makes them a
contract. This test holds the simulator to it: each option is run in a baseline situation where
it can show an effect, and the output or the cluster state must differ from the run without it.
An option that is a no-op in that situation for the real tool too (``-Q`` on a submission that
prints nothing anyway) is listed in ``NO_VISIBLE_EFFECT`` with the reason, so every exception is
a reviewed decision rather than a silent gap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pytest

from conftest import concrete
from greenray.commands import names as command_names
from greenray.content import helppages
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

CLOCK = 1_760_000_000.0  # fixed so timestamps written to the state are identical between runs

# Option lines: 1-8 spaces (or a tab), then -x[, --xx] [VALUE], then two spaces or end of line.
OPTION = re.compile(
    r"^[ \t]{1,8}(-[^\s,|:]+(?:\s*[,|]\s*-[^\s,|:]+)*)"
    r"(?:[ =:]+(<[^>]+>|[A-Z][A-Za-z_\-\[\].|,<>:0-9@#()']*|[a-z_]+(?:\|[a-z_]+)+|[a-z_()]+))?"
    r"(?=[ \t]{2,}|:?$)"
)
BOOL_DEFAULT = re.compile(r"^(--[\w-]+)=(true|false)$")

# Pages describe some options by shape rather than by name.
SPELLINGS = {"-NUM": "-2", "-<sig>": "-9", "-sigspec": "-9", "-i[SUFFIX]": "-i", "-": "-"}

# Values chosen from the help page's placeholder so the option is exercised rather than rejected.
VALUES = {
    "NODES": "dgx-04",
    "hosts...": "dgx-01",
    "hostlist": "dgx-04",
    "<node>": "dgx-01",
    "node_state": "down",
    "PARTITION": "nosuch",
    "partition": "nosuch",
    "partition(s)": "nosuch",
    "user_name": "alice",
    "user_name(s)": "alice",
    "<user>": "root",
    "U": "alice",
    "name": "nosuch",
    "jobname": "test",
    "job_name": "test",
    "job(s)": "1002",
    "ncpus": "2",
    "ntasks": "2",
    "N": "2",
    "n": "2",
    "S": "1",
    "NUM": "2",
    "minutes": "10",
    "MB": "4000",
    "value": "10",
    "<count>": "2",
    "format": "%N",
    "FORMAT": "%N",
    "state(s)": "pending",
    "states": "pending",
    "qos": "nosuch",
    "list": "nosuch",
    "opts": "enforce-binding",
    "type": "cyclic",
    "path": "/etc",
    "PATH": "/etc",
    "directory": "/etc",
    "out": "/root/out.txt",
    "err": "/root/err.txt",
    "names": "NONE",
    "type:jobid[:time]": "afterok:1002",
    "[R:]num[@time]": "B:USR1@60",
    "<device>": "mlx5_1",
    "<port_state>": "DN",
    "<arg0>": "2",
    "<package>": "slurm-client",
    "<pattern>": "slurm*",
    "<component>": "gpu",
    "<1-4>": "2",
    "<text>": "'GPU 3 fell off the bus; Xid 79 in dmesg'",
    "<command>": "id",
    "FILE": "/home/alice/resnet50/job.env",
    "DIR": "/raid/datasets/imagenet",
    "0|1": "1",
    "KEY[=VAL]": "X=1",
    "FSTAB": "/root:/mnt",
    "NAME": "c1",
    "MODE": "enabled",
    "D": "research",
    "P": "nlp",
    "ROLE": "'Research manager'",
    "N|none": "2",
    "<arg0> <arg1>": "btl tcp",
    "script": "s/a/b/",
    "UNIT": "nvidia-persistenced",
    "KEY": "x",
    "[USER@][REGISTRY#]IMAGE[:TAG]|PATH": "nvcr.io#nvidia/pytorch:24.05-py3",
    "SRC:DST[:FLAGS][,SRC:DST...]": "/root:/root",
    "NAME[,NAME...]": "HOME",
    "string": "x",
    "stringArray": "mig.strategy=single",
    "strings": "/root/v.yaml",
    "duration": "1m",
}


@dataclass
class Baseline:
    """How to run a command so its options can show an effect."""

    args: tuple[str, ...] = ()  # positional arguments; the option goes after them, or before for wrappers
    host: str = "bcm-head"
    setup: tuple[str, ...] = ()
    scenario: str = ""  # start from this scenario's broken cluster instead of the healthy one
    prefix: str = ""  # text before the command, for a pipe
    option_first: bool = False
    values: dict[str, str] = field(default_factory=dict)  # option -> value when the placeholder is vague
    alternatives: dict[str, tuple[str, ...]] = field(default_factory=dict)  # option -> other positional args


SCRIPT = "echo -e '#!/bin/bash\\n#SBATCH --gres=gpu:1\\nhostname' > /root/job.sh"
JOBS = (
    SCRIPT,
    "sacctmgr -i add user alice account=root",
    "su - alice -c 'sbatch -H /root/job.sh'",  # 1001 pending, held, alice
    "sbatch /root/job.sh",  # 1002 running, root
    "srun hostname",  # 1003 completed, root, with steps
)
HELM = ("helm repo add nvidia https://helm.ngc.nvidia.com/nvidia",)
RELEASE = HELM + (
    "helm install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace --set mig.strategy=mixed",
)
POD = ("kubectl run test --image=nvcr.io/nvidia/pytorch:24.05-py3 -- sleep 1000",)
MANIFEST = ("echo -e 'apiVersion: v1\\nkind: ConfigMap\\nmetadata:\\n  name: cm1\\ndata:\\n  a: b' > /root/cm.yaml",)
CONTAINER = ("docker run -d --gpus all nvcr.io/nvidia/pytorch:24.05-py3 sleep 1000",)
JOB_VALUES = {"-D": "/etc", "--chdir": "/etc", "-A": "nosuch", "--account": "nosuch", "--hint": "nomultithread"}
IMAGE = "nvcr.io/nvidia/pytorch:24.05-py3"

BASELINES: dict[tuple[str, str | None], Baseline] = {
    ("apt", None): Baseline(args=("list",), alternatives={"--installed": ("list",)}),
    ("apt-get", None): Baseline(
        args=("install", "slurm-client"),
        host="dgx-01",
        alternatives={"--purge": ("remove", "nvidia-container-toolkit")},
    ),
    ("awk", None): Baseline(args=("'{print $1}'", "/etc/slurm/slurm.conf")),
    ("cat", None): Baseline(args=("/etc/slurm/slurm.conf",)),
    ("chmod", None): Baseline(args=("644", "/etc/slurm")),
    ("chown", None): Baseline(args=("slurm", "/etc/slurm")),
    ("cmha", None): Baseline(args=("status",)),
    ("cp", None): Baseline(args=("/etc/slurm/slurm.conf", "/root/s.conf")),
    ("create-munge-key", None): Baseline(host="dgx-01", setup=("create-munge-key",)),
    ("crictl", None): Baseline(args=("ps",), host="dgx-01", values={"--name": "flannel"}),
    ("dataloader-bench", None): Baseline(
        host="dgx-02",
        scenario="tsh-storage-io-bottleneck",
        values={"-c": "/home/alice/nosuch.env", "--config": "/home/alice/nosuch.env"},
    ),
    ("dpkg", None): Baseline(host="dgx-01", values={"-l": "", "--list": ""}),
    ("enroot", None): Baseline(
        args=("start", "pt"),
        host="dgx-01",
        setup=(
            "enroot import -o /root/pt.sqsh docker://nvcr.io#nvidia/pytorch:24.05-py3",
            "enroot create /root/pt.sqsh",
        ),
        option_first=True,
        values={"-o": "/root/x.sqsh", "-n": "c2", "--name": "c2"},
        alternatives={
            "-o": ("import", "docker://nvcr.io#nvidia/pytorch:24.05-py3"),
            "-n": ("create", "/root/pt.sqsh"),
            "--name": ("create", "/root/pt.sqsh"),
        },
    ),
    ("exportfs", None): Baseline(),
    ("free", None): Baseline(),
    ("grep", None): Baseline(args=("slurm", "/etc/slurm/slurm.conf")),
    ("head", None): Baseline(args=("/etc/slurm/slurm.conf",), values={"-n": "2"}),
    ("tail", None): Baseline(args=("/etc/slurm/slurm.conf",), values={"-n": "2"}),
    ("hwticket", None): Baseline(
        args=(
            "create",
            "--node",
            "dgx-01",
            "--component",
            "gpu",
            "--summary",
            "'GPU 3 fell off the bus; Xid 79 in dmesg'",
        ),
        host="dgx-01",
        values={"--component": "fan", "--node": "dgx-02", "--summary": "'Fan 3 reports 0 RPM'", "--severity": "1"},
    ),
    ("journalctl", None): Baseline(host="dgx-01"),
    ("kill", None): Baseline(args=("4242",), host="dgx-01", setup=CONTAINER),
    ("pkill", None): Baseline(args=("sleep",), host="dgx-01", setup=CONTAINER),
    ("ls", None): Baseline(args=("/etc",)),
    ("mkdir", None): Baseline(args=("/root/a/b",)),
    ("mlxconfig", None): Baseline(args=("-d", "mlx5_0", "q"), host="dgx-01", setup=("mst start",)),
    ("mlxlink", None): Baseline(
        args=("-d", "mlx5_0"), host="dgx-01", setup=("mst start",), values={"-d": "mlx5_9", "--device": "mlx5_9"}
    ),
    ("modprobe", None): Baseline(args=("nvidia-peermem",), host="dgx-01"),
    ("mpirun", None): Baseline(
        args=("--allow-run-as-root", "hostname"),
        host="dgx-01",
        values={
            "-H": "dgx-01,dgx-02",
            "-host": "dgx-01,dgx-02",
            "--host": "dgx-01,dgx-02",
            "-hostfile": "/root/hosts",
            "--hostfile": "/root/hosts",
            "-machinefile": "/root/hosts",
            "--machinefile": "/root/hosts",
            "--map-by": "node",
            "--bind-to": "socket",
            "--rank-by": "node",
            "--prefix": "/opt/nosuch",
            "-wdir": "/nosuch",
            "--wdir": "/nosuch",
            "-x": "NCCL_DEBUG=INFO",
        },
        alternatives={
            "--allow-run-as-root": ("hostname",),
            "--oversubscribe": ("--allow-run-as-root", "-np", "40", "hostname"),
            "--map-by": ("--allow-run-as-root", "-H", "dgx-01:2,dgx-02:2", "hostname"),
            "--rank-by": ("--allow-run-as-root", "-H", "dgx-01:2,dgx-02:2", "hostname"),
            "--bind-to": ("--allow-run-as-root", "--report-bindings", "hostname"),
            "-x": ("--allow-run-as-root", "-np", "2", "all_reduce_perf", "-b", "8", "-e", "8M", "-g", "1"),
        },
    ),
    ("munge", None): Baseline(host="dgx-01", prefix="echo payload | "),
    ("mungekey", None): Baseline(host="dgx-01", setup=("mungekey",)),
    ("numactl", None): Baseline(args=("hostname",), host="dgx-01"),
    ("ping", None): Baseline(args=("dgx-02",)),
    ("pwd", None): Baseline(),
    ("rm", None): Baseline(args=("/root/nothing",)),
    ("runai-ui", None): Baseline(
        args=("project", "create", "team-b", "--department", "research", "--gpu-quota", "4"),
        scenario="adm-runai-project-quota",
        setup=("runai-ui project create nlp --department research --gpu-quota 6",),
        values={
            "--department": "nosuch",
            "--over-quota": "disabled",
            "--gpu-quota": "0",
            "--max-gpus": "6",
            "--user": "bob@corp.ai",
            "--project": "nosuch",
        },
        alternatives={
            "--user": ("access-rule", "create", "--user", "alice@corp.ai", "--project", "nlp"),
            "--project": ("access-rule", "create", "--user", "alice@corp.ai", "--project", "nlp"),
            "--role": ("access-rule", "create", "--user", "alice@corp.ai", "--project", "nlp"),
        },
    ),
    ("runuser", None): Baseline(args=("--", "id")),
    ("sacct", None): Baseline(
        setup=JOBS,
        values={
            "-j": "1002",
            "--jobs": "1002",
            "-o": "jobid,user",
            "--format": "jobid,user",
            "-s": "R",
            "--state": "R",
            "-u": "alice",
            "--user": "alice",
        },
    ),
    ("sacctmgr", None): Baseline(args=("show", "user")),
    ("salloc", None): Baseline(values=JOB_VALUES),
    ("sbatch", None): Baseline(args=("/root/job.sh",), setup=(SCRIPT,), values=JOB_VALUES),
    ("srun", None): Baseline(args=("hostname",), values=JOB_VALUES),
    ("scancel", None): Baseline(
        setup=JOBS,
        values={"-n": "job.sh", "--name": "job.sh", "--jobname": "job.sh", "-t": "pending", "--state": "pending"},
    ),
    ("sed", None): Baseline(args=("s/superpod/x/", "/etc/slurm/slurm.conf"), values={"-e": "s/a/b/"}),
    ("showmount", None): Baseline(args=("bcm-head",), host="dgx-01"),
    ("sinfo", None): Baseline(values={"-t": "down", "--states": "down"}),
    ("slurmd", None): Baseline(host="dgx-01"),
    ("sort", None): Baseline(args=("/etc/slurm/slurm.conf",)),
    ("sprio", None): Baseline(setup=JOBS, values={"-j": "1002", "--jobs": "1002"}),
    ("squeue", None): Baseline(setup=JOBS, values={"-j": "1002", "--jobs": "1002"}),
    ("sshare", None): Baseline(
        setup=JOBS[:2], values={"-A": "nosuch", "--accounts": "nosuch", "-u": "alice", "--users": "alice"}
    ),
    ("su", None): Baseline(args=("alice",), setup=JOBS[:2], values={"-c": "id", "--command": "id"}),
    ("systemctl", None): Baseline(args=("enable", "nvidia-fabricmanager"), host="dgx-01"),
    ("taskset", None): Baseline(
        args=("0x1", "hostname"),
        host="dgx-01",
        option_first=True,
        values={"-c": "", "--cpu-list": "", "-p": "", "--pid": ""},
        alternatives={
            "-c": ("0-3", "hostname"),
            "--cpu-list": ("0-3", "hostname"),
            "-p": ("4242",),
            "--pid": ("4242",),
        },
    ),
    ("tee", None): Baseline(args=("/root/t.txt",), setup=("echo one > /root/t.txt",), prefix="echo two | "),
    ("umount", None): Baseline(args=("/home",), host="dgx-01"),
    ("uname", None): Baseline(host="dgx-01"),
    ("wc", None): Baseline(args=("/etc/slurm/slurm.conf",)),
    ("helm", "list"): Baseline(setup=RELEASE, values={"-n": "gpu-operator", "--namespace": "gpu-operator"}),
    ("helm", "status"): Baseline(args=("gpu-operator", "-n", "gpu-operator"), setup=RELEASE),
    ("helm", "get"): Baseline(
        args=("values", "gpu-operator", "-n", "gpu-operator"), setup=RELEASE, values={"-o": "json", "--output": "json"}
    ),
    ("helm", "show"): Baseline(args=("values", "nvidia/gpu-operator"), setup=HELM, values={"--version": "v23.9.1"}),
    ("helm", "history"): Baseline(args=("gpu-operator", "-n", "gpu-operator"), setup=RELEASE),
    ("helm", "repo"): Baseline(args=("list",), setup=HELM),
    ("helm", "install"): Baseline(
        args=("gpu2", "nvidia/gpu-operator", "-n", "gpu2", "--create-namespace"),
        setup=RELEASE,
        values={"--version": "v23.9.1", "--timeout": "1m"},
        alternatives={"--create-namespace": ("gpu2", "nvidia/gpu-operator", "-n", "gpu2")},
    ),
    ("helm", "upgrade"): Baseline(
        args=("gpu-operator", "nvidia/gpu-operator", "-n", "gpu-operator"),
        setup=RELEASE,
        values={"--version": "v23.9.1", "--timeout": "1m"},
        alternatives={
            "-i": ("gpu3", "nvidia/gpu-operator", "-n", "gpu3"),
            "--install": ("gpu3", "nvidia/gpu-operator", "-n", "gpu3"),
        },
    ),
    ("helm", "rollback"): Baseline(
        args=("gpu-operator", "1", "-n", "gpu-operator"),
        setup=RELEASE + ("helm upgrade gpu-operator nvidia/gpu-operator -n gpu-operator --set mig.strategy=single",),
    ),
    ("helm", "uninstall"): Baseline(args=("gpu-operator", "-n", "gpu-operator"), setup=RELEASE),
    ("kubectl", "get"): Baseline(
        args=("pods", "-n", "kube-system"),
        values={
            "-l": "k8s-app=kube-dns",
            "--selector": "k8s-app=kube-dns",
            "-L": "k8s-app",
            "--label-columns": "k8s-app",
            "--field-selector": "spec.nodeName=dgx-01",
            "-o": "wide",
            "--output": "wide",
        },
    ),
    ("kubectl", "describe"): Baseline(
        args=("pods", "-n", "kube-system"),
        values={"-l": "k8s-app=kube-dns", "--selector": "k8s-app=kube-dns"},
    ),
    ("kubectl", "delete"): Baseline(
        args=("pod", "test"),
        setup=POD + MANIFEST,
        values={"-f": "/root/cm.yaml", "--filename": "/root/cm.yaml", "-l": "run=test", "--selector": "run=test"},
        alternatives={"-f": (), "--filename": (), "--ignore-not-found": ("pod", "nosuch")},
    ),
    ("kubectl", "logs"): Baseline(
        args=("test",),
        setup=POD,
        values={"-c": "nosuch", "--container": "nosuch", "--tail": "1", "-l": "run=test", "--selector": "run=test"},
    ),
    ("kubectl", "exec"): Baseline(
        args=("test", "--", "hostname"), setup=POD, option_first=True, values={"-c": "nosuch", "--container": "nosuch"}
    ),
    ("kubectl", "drain"): Baseline(
        args=("dgx-01", "--ignore-daemonsets"), setup=POD, alternatives={"--ignore-daemonsets": ("dgx-01",)}
    ),
    ("kubectl", "apply"): Baseline(
        args=("-f", "/root/cm.yaml"),
        setup=MANIFEST,
        values={"-f": "/root/nosuch.yaml", "--filename": "/root/nosuch.yaml"},
        alternatives={"-f": (), "--filename": ()},
    ),
    ("kubectl", "create"): Baseline(
        args=("secret", "generic", "s1", "--from-literal=a=b"),
        setup=MANIFEST + ("echo data > /root/key.txt",),
        values={
            "-f": "/root/cm.yaml",
            "--filename": "/root/cm.yaml",
            "--from-file": "/root/key.txt",
            "--from-literal": "c=d",
            "--image": IMAGE,
            "--replicas": "2",
            "--type": "kubernetes.io/tls",
            "--docker-server": "nvcr.io",
            "--docker-username": "'$oauthtoken'",
            "--docker-password": "x",
            "--docker-email": "a@b.c",
        },
        alternatives={
            "-f": (),
            "--filename": (),
            "--image": ("deployment", "d1"),
            "--replicas": ("deployment", "d1", f"--image={IMAGE}"),
            "--docker-server": ("secret", "docker-registry", "regcred", "--docker-username=u", "--docker-password=p"),
            "--docker-username": ("secret", "docker-registry", "regcred", "--docker-password=p"),
            "--docker-password": ("secret", "docker-registry", "regcred", "--docker-username=u"),
            "--docker-email": ("secret", "docker-registry", "regcred", "--docker-username=u", "--docker-password=p"),
        },
    ),
    ("kubectl", "label"): Baseline(
        args=("node", "dgx-01", "team=a"),
        setup=("kubectl label node dgx-01 team=b",) + POD,
        values={"-l": "kubernetes.io/hostname=dgx-02", "--selector": "kubernetes.io/hostname=dgx-02"},
        alternatives={
            "--all": ("node", "team=a"),
            "-l": ("node", "team=a"),
            "--selector": ("node", "team=a"),
            "-n": ("pod", "test", "team=a"),
            "--namespace": ("pod", "test", "team=a"),
        },
    ),
    ("kubectl", "taint"): Baseline(
        args=("node", "dgx-01", "team=a:NoSchedule"),
        setup=("kubectl taint node dgx-01 team=b:NoSchedule",),
        values={"-l": "kubernetes.io/hostname=dgx-02", "--selector": "kubernetes.io/hostname=dgx-02"},
        alternatives={
            "--all": ("node", "team=a:NoSchedule"),
            "-l": ("node", "team=a:NoSchedule"),
            "--selector": ("node", "team=a:NoSchedule"),
        },
    ),
    ("kubectl", "patch"): Baseline(
        args=("deploy", "coredns", "-n", "kube-system", "-p", '\'{"spec":{"replicas":3}}\''),
        values={"-p": '\'{"spec":{"replicas":4}}\'', "--patch": '\'{"spec":{"replicas":4}}\'', "--type": "json"},
    ),
    ("kubectl", "rollout"): Baseline(args=("status", "deploy/coredns", "-n", "kube-system")),
    ("kubectl", "scale"): Baseline(
        args=("deploy", "coredns", "-n", "kube-system", "--replicas=3"), values={"--replicas": "4"}
    ),
    ("kubectl", "run"): Baseline(
        args=("t2", f"--image={IMAGE}"),
        values={
            "--image": "nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04",
            "--labels": "team=a",
            "--overrides": '\'{"spec":{"nodeSelector":{"team":"a"}}}\'',
            "--restart": "Never",
        },
        alternatives={"--command": ("t2", f"--image={IMAGE}", "--", "hostname", "-s")},
    ),
    ("kubectl", "wait"): Baseline(
        args=("pod", "test", "--for=condition=Ready"),
        setup=POD,
        values={"--for": "delete", "--timeout": "1s", "-l": "run=test", "--selector": "run=test"},
    ),
    ("kubectl", "cordon"): Baseline(args=("dgx-01",), alternatives={"-l": ("node",), "--selector": ("node",)}),
    ("kubectl", "uncordon"): Baseline(args=("dgx-01",), setup=("kubectl cordon dgx-01",)),
}

# Options that correctly leave no trace in the baseline situation. Each entry is a decision, not a gap.
NO_VISIBLE_EFFECT: dict[tuple[str, str], str] = {
    ("apt", "-y"): "apt list asks no question",
    ("apt", "--yes"): "apt list asks no question",
    ("apt", "--assume-yes"): "apt list asks no question",
    ("apt-get", "-y"): "the simulated install never prompts, as with a non-interactive terminal",
    ("chmod", "-R"): "file modes are not modelled; the command is accepted for scripts that use it",
    ("chmod", "--recursive"): "file modes are not modelled; the command is accepted for scripts that use it",
    ("chown", "-R"): "file ownership is not modelled; the command is accepted for scripts that use it",
    ("chown", "--recursive"): "file ownership is not modelled; the command is accepted for scripts that use it",
    ("crictl", "--all"): "a healthy node has no exited containers to add",
    ("crictl", "-a"): "a healthy node has no exited containers to add",
    ("free", "-k"): "kibibytes are the default unit",
    ("free", "--kibi"): "kibibytes are the default unit",
    ("helm", "--timeout"): "operations complete at once in the simulator",
    ("helm", "--wait"): "operations complete at once in the simulator",
    ("helm", "--reuse-values"): "helm 3 reuses the previous values when none are given",
    ("journalctl", "--no-pager"): "output never goes through a pager here",
    ("kubectl", "--ignore-not-found"): "only matters for a missing object; covered by the alternative invocation",
    ("kubectl", "-f"): "the follow stream ends at once for a simulated log",
    ("kubectl", "--follow"): "the follow stream ends at once for a simulated log",
    ("kubectl", "-i"): "attaching a terminal is not simulated; the pod is created or executed the same way",
    ("kubectl", "--stdin"): "attaching a terminal is not simulated; the pod is created or executed the same way",
    ("kubectl", "-t"): "attaching a terminal is not simulated; the pod is created or executed the same way",
    ("kubectl", "--tty"): "attaching a terminal is not simulated; the pod is created or executed the same way",
    ("kubectl", "--timeout"): "the condition is already met, so the wait returns at once",
    ("kubectl", "--force"): "the drain baseline already fails on the unmanaged pod before --force is consulted",
    ("ls", "-1"): "output to a pipe is one name per line already",
    ("mlxconfig", "-y"): "the query command asks no question",
    ("mlxconfig", "--yes"): "the query command asks no question",
    ("mpirun", "-v"): "Open MPI prints nothing extra for a plain launch",
    ("mpirun", "--verbose"): "Open MPI prints nothing extra for a plain launch",
    ("pkill", "-<sig>"): "every signal ends the simulated process the same way",
    ("salloc", "--no-shell"): "no shell is started here either way; the allocation is granted and exported",
    ("sbatch", "--requeue"): "requeue is the default for batch jobs; --no-requeue is the change",
    ("taskset", "-c"): "the affinity of a launched command is not modelled; -p reports a process's mask",
    ("taskset", "--cpu-list"): "the affinity of a launched command is not modelled; -p reports a process's mask",
    ("mungekey", "-c"): "creating the key is the default command",
    ("mungekey", "--create"): "creating the key is the default command",
    ("numactl", "--localalloc"): "memory placement is not modelled",
    ("numactl", "-l"): "memory placement is not modelled",
    ("numactl", "--all"): "cpuset restrictions are not modelled",
    ("numactl", "-a"): "cpuset restrictions are not modelled",
    ("pwd", "-L"): "there are no symbolic links in the simulated tree",
    ("pwd", "-P"): "there are no symbolic links in the simulated tree",
    ("sbatch", "-Q"): "a successful submission prints no informational message to suppress",
    ("sbatch", "--quiet"): "a successful submission prints no informational message to suppress",
    ("srun", "-Q"): "a step that starts at once prints no informational message to suppress",
    ("srun", "--quiet"): "a step that starts at once prints no informational message to suppress",
    ("srun", "-I"): "resources are available, so immediate allocation is what happens anyway",
    ("srun", "--immediate"): "resources are available, so immediate allocation is what happens anyway",
    ("salloc", "-I"): "resources are available, so immediate allocation is what happens anyway",
    ("salloc", "--immediate"): "resources are available, so immediate allocation is what happens anyway",
    ("srun", "-k"): "no node fails during a simulated step",
    ("srun", "--no-kill"): "no node fails during a simulated step",
    ("salloc", "-k"): "no node fails during a simulated allocation",
    ("salloc", "--no-kill"): "no node fails during a simulated allocation",
    ("sbatch", "-k"): "no node fails during a simulated job",
    ("sbatch", "--no-kill"): "no node fails during a simulated job",
    ("srun", "-K"): "the step's only task exits 0",
    ("srun", "--kill-on-bad-exit"): "the step's only task exits 0",
    ("srun", "-u"): "output is not buffered in the simulator",
    ("srun", "--unbuffered"): "output is not buffered in the simulator",
    ("srun", "--overlap"): "nothing else runs in the allocation to overlap with",
    ("srun", "--pty"): "a pseudo-terminal only changes how bash is started; hostname runs the same way",
    ("umount", "-f"): "the NFS server is reachable, so a normal unmount succeeds",
    ("umount", "--force"): "the NFS server is reachable, so a normal unmount succeeds",
    ("umount", "-l"): "nothing holds the filesystem open, so lazy and normal unmount agree",
    ("umount", "--lazy"): "nothing holds the filesystem open, so lazy and normal unmount agree",
}
# Options that are no-ops only for particular sub-commands.
NO_VISIBLE_EFFECT_FOR: dict[tuple[str, str, str], str] = {
    ("helm", "repo", "-n"): "repositories are not namespaced",
    ("helm", "repo", "--namespace"): "repositories are not namespaced",
    ("helm", "repo", "--kube-context"): "helm repo does not contact the cluster",
    ("helm", "show", "-n"): "chart metadata is not namespaced",
    ("helm", "show", "--namespace"): "chart metadata is not namespaced",
    ("helm", "show", "--kube-context"): "helm show does not contact the cluster",
    ("kubectl", "logs", "-c"): "covered: the single container is addressed by name, a wrong name is rejected",
}


def parse_options(text: str) -> list[tuple[list[str], str | None]]:
    """[(flags, placeholder)] for each option line; placeholder None when the option takes no value."""
    out = []
    for line in text.splitlines():
        m = OPTION.match(line)
        if not m:
            continue
        spec, placeholder = m.group(1), m.group(2)
        flags = []
        for raw in re.split(r"\s*[,|]\s*", spec):
            b = BOOL_DEFAULT.match(raw)
            if b:
                flags.append(b.group(1))
                placeholder = None
                continue
            name, eq, default = raw.partition("=")
            flags.append(name)
            if eq and placeholder is None:
                placeholder = default.rstrip(":") or "string"
        out.append((flags, placeholder))
    return out


def cases():
    registered = set(command_names())
    for cmd in helppages.commands_with_help():
        folder = helppages._index()[cmd]
        if cmd not in registered or folder.name != cmd:  # aliases share the implementation
            continue
        help_flags = set(helppages._meta(folder).get("flags", helppages.DEFAULT_FLAGS))
        for page in sorted(folder.glob("*.txt")):
            if page.stem == "man":
                continue
            sub = None if page.stem == "help" else page.stem
            for flags, placeholder in parse_options(page.read_text(encoding="utf-8")):
                for flag in flags:
                    if flag in help_flags or flag in ("--version", "-V", "--help"):
                        continue
                    yield cmd, sub, flag, placeholder


def value_for(base: Baseline, flag: str, placeholder: str | None) -> str | None:
    if flag in base.values:
        return base.values[flag] or None
    if placeholder is None:
        return None
    if placeholder in VALUES:
        return VALUES[placeholder]
    return placeholder.strip("<>[]'").split("|")[0] or "x"


def digest(state) -> str:
    clean = json.loads(json.dumps(state, default=str))
    clean.get("flags", {}).pop("clock", None)
    return json.dumps(clean, sort_keys=True)


def run(base: Baseline, line: str) -> tuple[str, str]:
    state = concrete(base.scenario).build_state() if base.scenario else make_cluster()
    state.flags["clock"] = CLOCK
    shell = Shell(state)
    if base.host != "bcm-head":
        shell.run(f"ssh {base.host}")
    for step in base.setup:
        shell.run(step)
    out = shell.run(base.prefix + line)
    return out, digest(state)


ALL = list(cases())


@pytest.mark.parametrize(
    ("cmd", "sub", "flag", "placeholder"), ALL, ids=[f"{c} {s or ''} {f}".replace("  ", " ") for c, s, f, _ in ALL]
)
def test_option_leaves_a_trace(cmd, sub, flag, placeholder):
    base = BASELINES.get((cmd, sub), Baseline())
    head = [cmd] + ([sub] if sub else [])
    args = list(base.alternatives.get(flag, base.args))
    value = value_for(base, flag, placeholder)
    spelled = SPELLINGS.get(flag, flag)
    if value is None:
        option = [spelled]
    elif cmd == "kubectl" and flag.startswith("--"):
        option = [f"{flag}={value}"]
    else:
        option = [spelled, value]
    # A wrapper's options go before the wrapped command; elsewhere the option comes last so it wins.
    first = base.option_first or helppages._meta(helppages._index()[cmd]).get("wrapper")
    argv = head + option + args if first else head + args + option
    plain_out, plain_state = run(base, " ".join(head + args))
    out, state = run(base, " ".join(argv))
    assert "internal simulator error" not in out, out
    if (cmd, flag) in NO_VISIBLE_EFFECT or (cmd, sub or "", flag) in NO_VISIBLE_EFFECT_FOR:
        return
    assert out != plain_out or state != plain_state, (
        f"{' '.join(argv)!r} behaves exactly like the invocation without {flag}"
    )


def test_every_help_page_belongs_to_a_registered_command():
    missing = sorted(set(helppages.commands_with_help()) - set(command_names()))
    assert missing == [], missing


def test_no_op_list_names_only_options_that_exist():
    listed = {(c, f) for c, _, f, _ in ALL}
    assert set(NO_VISIBLE_EFFECT) <= listed, set(NO_VISIBLE_EFFECT) - listed
    assert {(c, f) for c, _, f in NO_VISIBLE_EFFECT_FOR} <= listed
