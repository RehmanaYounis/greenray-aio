"""Invocations the real tools reject must fail here too, with the real tool's wording.

A simulator that accepts a wrong command teaches a wrong habit. This table is the contract;
extend it whenever a report shows the real tool being stricter than the simulator.
"""

import pytest

from greenray.commands import helm
from greenray.commands import kubectl as kc
from greenray.core.errors import CommandError
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

CASES = [
    # Slurm
    ("scontrol update nodename=dgx-09 state=drain reason=x", "Invalid node name"),
    ("scontrol update nodename=dgx-01 state=bogus reason=x", "Invalid"),
    ("scontrol update nodename=dgx-01 state=drain", "must specify a reason"),
    ("scontrol update jobid=9999 timelimit=1:00:00", "Invalid job id"),
    ("scontrol update partitionname=nope state=up", "Invalid partition name"),
    ("scontrol update partitionname=defq maxtime=abc", "Invalid time limit"),
    ("scontrol show node dgx-09", "not found"),
    ("scontrol reboot dgx-09", "Invalid node name"),
    ("sbatch --gres=gpu:9 --wrap=hostname", "Requested node configuration is not available"),
    ("sbatch --gres=gpuz:1 --wrap=hostname", "Invalid generic resource"),
    ("sbatch -p nope --wrap=hostname", "invalid partition"),
    ("sbatch -N 5 --wrap=hostname", "Node count specification invalid"),
    ("sbatch /root/missing.sbatch", "Unable to open file"),
    ("scancel 9999", "Invalid job id"),
    ("scancel", "No job identification provided"),
    ("sacctmgr -i add user nobody account=nope", "account"),
    ("sacct -j abc", "Invalid job id"),
    ("ssh dgx-09", "Could not resolve hostname"),
    # GPU tools
    ("ssh dgx-01 nvidia-smi -pl 900", "not a valid power limit"),
    ("ssh dgx-01 nvidia-smi -i 9 -q", "No devices were found"),
    ("ssh dgx-01 nvidia-smi -pm 2", "Invalid persistence mode"),
    ("ssh dgx-01 nvidia-smi -lgc 9999", "not allowed"),
    ("ssh dgx-01 nvidia-smi mig -cgi 19 -i 0", "MIG mode is not enabled"),
    ("ssh dgx-01 nvidia-smi --query-gpu=bogus --format=csv", "not a valid field"),
    ("ssh dgx-01 dcgmi diag -r 9", "Invalid"),
    ("ssh dgx-01 dcgmi group -d 99", "Not found"),
    # node
    ("ssh dgx-01 systemctl restart nope", "could not be found"),
    ("ssh dgx-01 modprobe nosuchmodule", "Module nosuchmodule not found"),
    ("ssh dgx-01 rm /etc/nosuch", "No such file or directory"),
    ("ssh dgx-01 cat /etc/nosuch", "No such file or directory"),
    ("ssh dgx-01 cd /nosuch", "No such file or directory"),
    ("ssh dgx-01 ibstat mlx5_9", "mlx5_9"),
    ("ssh dgx-01 docker run --gpus all nosuch/image:1 nvidia-smi", "repository does not exist"),
    ("ssh dgx-01 docker pull nosuch/image:1", "repository does not exist"),
    ("ssh dgx-01 enroot import docker://nosuch/image:1", "404 Not Found"),
    ("ssh dgx-01 docker exec nosuch nvidia-smi", "No such container"),
    ("ssh dgx-01 nvidia-ctk runtime configure --runtime=bogus", "unrecognized runtime"),
    # Kubernetes
    ("kubectl get pod nosuch", "not found"),
    ("kubectl describe node dgx-09", "not found"),
    ("kubectl delete pod nosuch", "not found"),
    ("kubectl apply -f /root/missing.yaml", "does not exist"),
    ("kubectl cordon dgx-09", "not found"),
    ("kubectl taint nodes dgx-01 key=value:BadEffect", "invalid taint effect"),
    ("kubectl get frobs", "doesn't have a resource type"),
    ("helm upgrade nosuch nvidia/gpu-operator", "has no deployed releases"),
    ("helm uninstall nosuch", "not found"),
    ("helm install x nosuchrepo/chart", "not found"),
    # BCM
    ('cmsh -c "device; use dgx-09; show"', "dgx-09"),
    ('cmsh -c "device; use dgx-01; set nosuchfield 1; commit"', "nosuchfield"),
    ('cmsh -c "device; use dgx-01; set category nosuch; commit"', "nosuch"),
    ('cmsh -c "category; use nosuch; show"', "nosuch"),
    ('cmsh -c "nosuchmode; list"', "nosuchmode"),
    ('cmsh -c "device; power off -n dgx-09"', "dgx-09"),
]


@pytest.fixture(scope="module")
def state():
    st = make_cluster()
    helm.preinstall(st, "nvidia/gpu-operator", values={"driver": {"enabled": False}}, settle=86400)
    kc.reconcile(st)
    return st


@pytest.mark.parametrize(("command", "wording"), CASES, ids=[c for c, _ in CASES])
def test_wrong_invocation_is_rejected(state, command, wording):
    sh = Shell(state)
    try:
        out = sh.run(command)
    except CommandError as e:
        out = str(e)
    else:
        assert out.strip(), f"{command!r} was accepted silently"
    assert wording.lower() in out.lower(), f"{command!r} ->\n{out}"
