"""Commands added in 1.6.0: pdsh, lsblk, ibdev2netdev, last, the extra kubectl verbs and GPU tool subsystems."""

from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def sh():
    return Shell(make_cluster())


def test_pdsh_runs_on_every_named_node():
    s = sh()
    assert s.run("pdsh -w dgx-0[1-2] hostname").splitlines() == ["dgx-01: dgx-01", "dgx-02: dgx-02"]
    assert s.run("pdsh -g dgx-h100 -x dgx-0[2-4] hostname") == "dgx-01: dgx-01"
    assert len(s.run("pdsh -a 'nvidia-smi -L | wc -l'").splitlines()) == 4
    s.run("ipmitool -H dgx-04-bmc power off") if False else s.ctx.state.nodes["dgx-04"].update({"up": False})
    assert "dgx-04: ssh: connect to host dgx-04" in s.run("pdsh -w dgx-04 hostname")
    assert "no remote hosts" in s.run("pdsh hostname")
    assert "Usage: pdsh" in s.run("pdsh -h")
    assert s.run("pdsh -w dgx-01 df -h").startswith("dgx-01: ")  # -h belongs to df


def test_lsblk_ibdev2netdev_last_and_chassis():
    s = sh()
    assert "sda" in s.run("lsblk")  # the head node has SATA drives
    s.run("ssh dgx-01")
    out = s.run("lsblk")
    assert "nvme0n1" in out and "/raid" in out and "raid0" in out
    assert s.run("lsblk -d -o NAME,SIZE").splitlines()[1].split() == ["nvme0n1", "1.7T"]
    assert "unknown column" in s.run("lsblk -o NOPE")
    lines = s.run("ibdev2netdev").splitlines()
    assert len(lines) == 8 and lines[0] == "mlx5_0 port 1 ==> ibp27s0 (Up)"
    assert "ConnectX-7" in s.run("ibdev2netdev -v")
    assert "System Power         : on" in s.run("ipmitool chassis status")
    last = s.run("last")
    assert "system boot" in last and "wtmp begins" in last
    assert len(s.run("last -1").splitlines()) == 3


def test_kubectl_replicasets_crds_explain_config():
    s = sh()
    rs = s.run("kubectl get rs -n kube-system")
    assert rs.splitlines()[0].split()[:4] == ["NAME", "DESIRED", "CURRENT", "READY"] and "coredns-" in rs
    assert "clusterpolicies.nvidia.com" in s.run("kubectl get crd")
    assert "nicclusterpolicies" not in s.run("kubectl get crd")
    assert "nvidia.com/gpu" in s.run("kubectl explain pods.spec.containers.resources")
    assert "KIND:       Deployment" in s.run("kubectl explain deploy")
    assert 'field "spec.nope" does not exist' in s.run("kubectl explain pods.spec.nope")
    assert s.run("kubectl config current-context") == "kubernetes-admin@kubernetes"
    assert "*" in s.run("kubectl config get-contexts")
    assert "server: https://10.141.255.254:6443" in s.run("kubectl config view")


def test_kubectl_set_image_rollout_history_undo_and_expose():
    s = sh()
    s.run("kubectl create deployment web --image=nginx:1.25")
    assert s.run("kubectl set image deploy/web nginx=nginx:1.26") == "deployment.apps/web image updated"
    hist = s.run("kubectl rollout history deploy/web")
    assert "2         kubectl set image deploy/web nginx=nginx:1.26" in hist
    assert "nginx:1.26" in s.run("kubectl get deploy web -o yaml")
    rs = s.run("kubectl get rs")
    assert len(rs.splitlines()) == 3 and "0         0         0" in rs  # the old replica set is kept at zero
    assert s.run("kubectl rollout undo deploy/web") == "deployment.apps/web rolled back"
    assert "nginx:1.25" in s.run("kubectl get deploy web -o yaml")
    assert "Image:\tnginx:1.25" in s.run("kubectl rollout history deploy/web --revision=3")
    assert 'unable to find container named "nosuch"' in s.run("kubectl set image deploy/web nosuch=x")
    assert s.run("kubectl expose deploy/web --port=80 --type=NodePort") == "service/web exposed"
    svc = s.run("kubectl get svc web")
    assert "NodePort" in svc and "80:30000/TCP" in svc
    assert "already exists" in s.run("kubectl expose deploy/web --port=80")
    assert "couldn't find port" in s.run("kubectl expose deploy/web --name=web2")
    assert s.run("kubectl delete svc web") == 'service "web" deleted'
    assert "web" not in s.run("kubectl get svc")


def test_gpu_tool_subsystems():
    s = sh()
    s.run("ssh dgx-01")
    assert "no GPU instances exist" in s.run("nvidia-smi mig -i 0 -lcip")
    s.run("nvidia-smi -i 0 -mig 1")
    s.run("nvidia-smi mig -i 0 -cgi 9")
    assert "MIG 1c.3g.40gb" in s.run("nvidia-smi mig -i 0 -lcip") and "3/3" in s.run("nvidia-smi mig -i 0 -lcip")
    assert "UUUUUU UUUUUU UUUUUU" in s.run("dcgmi nvlink -s")
    s.ctx.node.gpus[1].nvlink_down = [0, 1]
    assert "DDUUUU" in s.run("dcgmi nvlink -s")
    assert "CRC FLIT Error: 40000" in s.run("dcgmi nvlink -e -g 1")
    assert "NV16" in s.run("dcgmi topo")
    assert "Connected via an NVSwitch with 18 NVLINKs" in s.run("dcgmi topo -g 2")
    assert "DCGM_INTERNAL_1SEC" in s.run("dcgmi fieldgroup -l")
    assert 'field group "mine" with a field group ID of 4' in s.run("dcgmi fieldgroup -c mine -f 150,155")
    assert "Field IDs: 150, 155" in s.run("dcgmi fieldgroup -i -g 4")
    assert "not a valid field" in s.run("dcgmi fieldgroup -c bad -f 424242")
    assert "built-in" in s.run("dcgmi fieldgroup -d -g 0")
    assert s.run("dcgmi fieldgroup -d -g 4") == "Successfully removed field group 4"
    assert "gr_engine_active" in s.run("dcgmi profile -l")


def test_nvsm_show_targets_and_docker_housekeeping():
    s = sh()
    s.run("ssh dgx-01")
    gpus = s.run("nvsm show gpus")
    assert "GPU7:" in gpus and "Health                   = OK" in gpus
    s.ctx.node.gpus[3].ecc_volatile_dbe = 2
    assert "Health                   = Critical" in s.run("nvsm show gpus")
    assert "RAID 0" in s.run("nvsm show storage")
    assert "PSU6" in s.run("nvsm show power")
    assert "FAN1" in s.run("nvsm show fans")
    assert s.run("nvsm show version") == "nvsm 24.09.01"
    assert "Images" in s.run("docker system df")
    assert s.run("docker volume create data") == "data"
    assert "local    data" in s.run("docker volume ls")
    assert "no such volume" in s.run("docker volume rm nosuch")
    assert s.run("docker volume rm data") == "data"
    assert "bridge" in s.run("docker network ls")
    assert "Total reclaimed space" in s.run("docker system prune -f")
    assert "needs -f" in s.run("docker system prune")
