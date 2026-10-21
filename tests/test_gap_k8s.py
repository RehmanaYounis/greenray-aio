"""Tests for GPU sharing (time-slicing), Run:ai, and Docker / NVIDIA Container Toolkit emulation."""

import json

import pytest

from greenray.commands import container as ct
from greenray.commands import helm
from greenray.commands import kubectl as kc
from greenray.commands import runai as ra
from greenray.core.shell import Shell
from greenray.core.state import make_cluster

TS_CM = """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: ts-config
  namespace: gpu-operator
data:
  ts4: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4
  renamed: |-
    version: v1
    sharing:
      timeSlicing:
        renameByDefault: true
        resources:
        - name: nvidia.com/gpu
          replicas: 2
  broken: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 0
"""


def operator_state(settle=600):
    st = make_cluster()
    helm.preinstall(st, "nvidia/gpu-operator", values={"driver": {"enabled": False}}, settle=settle)
    return st


def dp_pod(st, node):
    return next(p for p in st.k8s.pods if p.component == "devicePlugin" and p.node == node)


# ---------------------------------------------------------------- time-slicing


def test_time_slicing_per_node_label():
    st = operator_state()
    sh = Shell(st)
    kc.apply_yaml(st, TS_CM)
    sh.run(
        "helm upgrade gpu-operator nvidia/gpu-operator -n gpu-operator --reuse-values --set devicePlugin.config.name=ts-config"
    )
    # no default and no label: nothing changes
    assert st.k8s.nodes["dgx-04"].allocatable["nvidia.com/gpu"] == "8"
    sh.run("kubectl label node dgx-04 nvidia.com/device-plugin.config=ts4")
    lab = st.k8s.nodes["dgx-04"].labels
    assert st.k8s.nodes["dgx-04"].allocatable["nvidia.com/gpu"] == "32"
    assert lab["nvidia.com/gpu.replicas"] == "4"
    assert lab["nvidia.com/gpu.sharing-strategy"] == "time-slicing"
    assert lab["nvidia.com/gpu.product"].endswith("-SHARED")
    assert lab["nvidia.com/gpu.count"] == "8"
    assert st.k8s.nodes["dgx-01"].allocatable["nvidia.com/gpu"] == "8"
    assert "create" not in st.k8s.clusterpolicy.spec.devicePlugin.config


def test_time_slicing_default_applies_everywhere_and_pods_share_gpus():
    st = operator_state()
    kc.apply_yaml(st, TS_CM)
    helm.upgrade(
        st,
        helm.find_release(st, "gpu-operator", "gpu-operator"),
        "nvidia/gpu-operator",
        {"devicePlugin": {"config": {"name": "ts-config", "default": "ts4"}}},
        reuse=True,
    )
    assert all(st.k8s.nodes[n].allocatable["nvidia.com/gpu"] == "32" for n in ("dgx-01", "dgx-04"))
    kc.apply_yaml(
        st,
        """
apiVersion: apps/v1
kind: Deployment
metadata: {name: many, namespace: default}
spec:
  replicas: 10
  selector: {matchLabels: {app: many}}
  template:
    metadata: {labels: {app: many}}
    spec:
      nodeSelector: {kubernetes.io/hostname: dgx-02}
      containers:
      - name: c
        image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
        resources: {limits: {nvidia.com/gpu: 1}}
""",
    )
    running = [p for p in st.k8s.pods if p.name.startswith("many-") and p.status == "Running"]
    assert len(running) == 10
    # replicas are spread over the physical GPUs first, then stacked
    gpus = [kc.visible_devices(st, p)[0][0] for p in running]
    assert len(set(gpus)) == 8
    sh = Shell(st)
    out = sh.run(f"kubectl exec {running[0].name} -- nvidia-smi -L")
    assert out.count("GPU ") == 1


def test_time_slicing_rename_missing_key_and_bad_config():
    st = operator_state()
    sh = Shell(st)
    kc.apply_yaml(st, TS_CM)
    sh.run(
        "helm upgrade gpu-operator nvidia/gpu-operator -n gpu-operator --reuse-values --set devicePlugin.config.name=ts-config"
    )
    sh.run("kubectl label node dgx-03 nvidia.com/device-plugin.config=renamed")
    kn = st.k8s.nodes["dgx-03"]
    assert kn.allocatable["nvidia.com/gpu.shared"] == "16"
    assert kn.allocatable["nvidia.com/gpu"] == "0"
    assert not kn.labels["nvidia.com/gpu.product"].endswith("-SHARED")
    sh.run("kubectl label node dgx-02 nvidia.com/device-plugin.config=nope")
    assert dp_pod(st, "dgx-02").status == "Init:CrashLoopBackOff"
    sh.run("kubectl label node dgx-01 nvidia.com/device-plugin.config=broken")
    p = dp_pod(st, "dgx-01")
    assert p.status == "Init:CrashLoopBackOff"
    assert "replicas" in "\n".join(p.logs)


def test_time_slicing_missing_configmap_blocks_device_plugin():
    st = operator_state()
    sh = Shell(st)
    sh.run(
        "helm upgrade gpu-operator nvidia/gpu-operator -n gpu-operator --reuse-values --set devicePlugin.config.name=nothere"
    )
    p = dp_pod(st, "dgx-01")
    assert p.status == "ContainerCreating"
    assert 'configmap "nothere" not found' in sh.run("kubectl describe pod -n gpu-operator " + p.name)
    kc.apply_yaml(st, TS_CM.replace("ts-config", "nothere"))
    assert dp_pod(st, "dgx-01").status == "Running"


def test_editing_configmap_needs_device_plugin_restart():
    st = operator_state()
    sh = Shell(st)
    kc.apply_yaml(st, TS_CM)
    sh.run(
        "helm upgrade gpu-operator nvidia/gpu-operator -n gpu-operator --reuse-values --set devicePlugin.config.name=ts-config"
    )
    sh.run("kubectl label node dgx-04 nvidia.com/device-plugin.config=ts4")
    kc.apply_yaml(st, TS_CM.replace("replicas: 4", "replicas: 8"))
    assert st.k8s.nodes["dgx-04"].allocatable["nvidia.com/gpu"] == "32"  # not re-read yet
    sh.run("kubectl rollout restart -n gpu-operator daemonset/nvidia-device-plugin-daemonset")
    sh.run("kubectl rollout restart -n gpu-operator daemonset/gpu-feature-discovery")
    assert st.k8s.nodes["dgx-04"].allocatable["nvidia.com/gpu"] == "64"
    assert st.k8s.nodes["dgx-04"].labels["nvidia.com/gpu.replicas"] == "8"


def test_helm_creates_device_plugin_configmap_and_patch_configmap():
    st = operator_state()
    data = {
        "ts2": "version: v1\nsharing:\n  timeSlicing:\n    resources:\n    - name: nvidia.com/gpu\n      replicas: 2\n"
    }
    helm.upgrade(
        st,
        helm.find_release(st, "gpu-operator", "gpu-operator"),
        "nvidia/gpu-operator",
        {"devicePlugin": {"config": {"create": True, "name": "chart-ts", "default": "ts2", "data": data}}},
        reuse=True,
    )
    assert any(c.name == "chart-ts" for c in st.k8s.configmaps)
    assert st.k8s.nodes["dgx-02"].allocatable["nvidia.com/gpu"] == "16"
    sh = Shell(st)
    out = sh.run("""kubectl patch cm chart-ts -n gpu-operator --type merge -p '{"data":{"extra":"x"}}'""")
    assert "patched" in out
    assert "already exists" in sh.run("kubectl create configmap chart-ts -n gpu-operator --from-literal=a=b")


# ---------------------------------------------------------------- Run:ai


@pytest.fixture
def runai_state():
    st = operator_state()
    ra.install(st, nodes=["dgx-03", "dgx-04"])
    ra.create_department(st, "research", 16)
    return st


def test_runai_project_creation_and_kubectl_views(runai_state):
    sh = Shell(runai_state)
    out = sh.run("runai-ui project create nlp --department research --gpu-quota 6")
    assert "runai-nlp" in out and "runai-nlp" in runai_state.k8s.namespaces
    assert "exceed" in sh.run("runai-ui project create vision --department research --gpu-quota 11")
    assert "invalid project name" in sh.run("runai-ui project create Bad_Name --gpu-quota 1")
    assert "nlp" in sh.run("kubectl get projects")
    assert "research" in sh.run("kubectl get departments")
    assert "deservedGpus: 6" in sh.run("kubectl get project nlp -o yaml")
    lines = iter(["apiVersion: run.ai/v2", "kind: Project", "metadata:", "  name: x", "EOF"])
    sh.run("cat > /root/p.yaml <<EOF", heredoc_reader=lambda _p: next(lines))
    assert "managed by the Run:ai control plane" in sh.run("kubectl apply -f /root/p.yaml")
    assert "Research manager" in sh.run(
        "runai-ui access-rule create --user c@x --project nlp --role 'research manager'"
    )
    assert "unknown role" in sh.run("runai-ui access-rule create --user c@x --project nlp --role boss")
    assert ra.members(runai_state, "nlp") == ["c@x"]


def test_runai_preemption_reclaims_deserved_quota(runai_state):
    st = runai_state
    ra.create_project(st, "sandbox", "default", 0)
    ra.create_project(st, "vision", "research", 10, over_quota=False)
    for i in range(4):
        ra.submit(st, f"hpo-{i}", "sandbox", "nvcr.io/nvidia/pytorch:24.05-py3", 4, created=kc.now(st) + i)
    kc.reconcile(st)
    assert all(w.status == "Running" for w in st.runai.workloads)
    sh = Shell(st)
    sh.run("runai submit smoke -i nvcr.io/nvidia/pytorch:24.05-py3 -g 8 -p vision")
    smoke = ra.workload(st, "smoke")
    assert smoke.status == "Running" and len(smoke.alloc) == 8
    pre = [w for w in st.runai.workloads if w.project == "sandbox" and w.status == "Pending"]
    assert len(pre) == 2 and all(w.preemptions == 1 for w in pre)
    # vision cannot go over its quota (over-quota disabled)
    sh.run("runai submit more -i nvcr.io/nvidia/pytorch:24.05-py3 -g 4 -p vision")
    assert ra.workload(st, "more").status == "Pending"
    assert "over-quota is disabled" in sh.run("runai describe job more -p vision")
    # pods mirror the workloads in the project namespace
    pod = next(p for p in st.k8s.pods if p.name == "smoke-0-0")
    assert pod.namespace == "runai-vision" and pod.status == "Running" and pod.requests == {"nvidia.com/gpu": 8}


def test_runai_non_preemptible_interactive_and_fractions(runai_state):
    st = runai_state
    ra.create_project(st, "nlp", "research", 1)
    sh = Shell(st)
    sh.run("runai config project nlp")
    sh.run("runai submit a -i nvcr.io/nvidia/pytorch:24.05-py3 -g 1 --interactive")
    sh.run("runai submit b -i nvcr.io/nvidia/pytorch:24.05-py3 -g 1 --interactive")
    assert ra.workload(st, "b").status == "Pending"
    assert "Non-preemptible" in sh.run("runai describe job b")
    sh.run("runai delete job a b")
    sh.run("runai workspace submit a -i nvcr.io/nvidia/pytorch:24.05-py3 --gpu-portion-request 0.5")
    sh.run("runai submit b -i nvcr.io/nvidia/pytorch:24.05-py3 -g 0.5 --interactive")
    a, b = ra.workload(st, "a"), ra.workload(st, "b")
    assert a.status == b.status == "Running"
    assert (a.node, a.alloc[0][0]) == (b.node, b.alloc[0][0])
    res = [p for p in st.k8s.pods if p.component == "runai-reservation"]
    assert len(res) == 1 and res[0].requests == {"nvidia.com/gpu": 1}
    out = sh.run("runai exec a -- nvidia-smi --query-gpu=memory.total --format=csv,noheader")
    assert "40779" in out
    assert "gpu" in sh.run("runai list nodes")
    assert "Workspace" in sh.run("runai workload list")
    # preemptible training may go over quota on idle GPUs
    sh.run("runai training submit t -i nvcr.io/nvidia/pytorch:24.05-py3 --gpu-devices-request 2")
    assert ra.workload(st, "t").status == "Running"
    # k8s scheduler sees Run:ai's GPUs as used
    assert kc.node_used(st, a.node).get("nvidia.com/gpu", 0) >= 1


def test_runai_cli_errors(runai_state):
    sh = Shell(runai_state)
    assert "no project specified" in sh.run("runai submit x -i img -g 1")
    assert "does not exist" in sh.run("runai submit x -i img -g 1 -p ghost")
    assert "Version" in sh.run("runai version")
    st = make_cluster()
    assert "not installed" in Shell(st).run("runai list jobs")


# ---------------------------------------------------------------- Docker / Container Toolkit


def fresh_node_state():
    st = make_cluster()
    n = st.nodes["dgx-03"]
    n.os = "Ubuntu 22.04.4 LTS"
    n.ctk_installed = False
    n.files.pop("/etc/nvidia-container-runtime/config.toml")
    n.files.pop("/etc/docker/daemon.json")
    n.files[ct.CTK_REPO_LIST] = "deb https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /\n"
    n.apt_fresh = False
    ct.container_init(st)
    sh = Shell(st)
    sh.run("ssh dgx-03")
    return st, sh


CUDA = "nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04"


def test_toolkit_install_flow():
    st, sh = fresh_node_state()
    assert "could not select device driver" in sh.run(f"docker run --rm --gpus all {CUDA} nvidia-smi -L")
    assert "Unable to locate package" in sh.run("apt-get install -y nvidia-container-toolkit")
    sh.run("apt-get update")
    assert "Abort." in sh.run("apt-get install nvidia-container-toolkit")
    assert "Setting up nvidia-container-toolkit" in sh.run("apt-get install -y nvidia-container-toolkit")
    assert "already the newest version" in sh.run("apt install -y nvidia-container-toolkit")
    assert "ii  nvidia-container-toolkit " in sh.run("dpkg -l nvidia-container-toolkit")
    # hook installed but dockerd hasn't been restarted
    assert "could not select device driver" in sh.run(f"docker run --rm --gpus all {CUDA} nvidia-smi -L")
    sh.run("nvidia-ctk runtime configure --runtime=docker")
    assert "unknown or invalid runtime name" in sh.run(f"docker run --rm --runtime=nvidia {CUDA} nvidia-smi -L")
    sh.run("systemctl restart docker")
    assert sh.run(f"docker run --rm --gpus all {CUDA} nvidia-smi -L").count("GPU ") == 8
    assert sh.run(f"docker run --rm --gpus '\"device=1,2\"' {CUDA} nvidia-smi -L").count("GPU ") == 2
    assert (
        sh.run(f"docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=3 {CUDA} nvidia-smi -L").count("GPU ") == 1
    )
    assert st.flags["gpu_containers"][0]["gpus"] == 8
    sh.run("apt-get purge -y nvidia-container-toolkit")
    assert not ct.toolkit_installed(st.nodes["dgx-03"])


@pytest.mark.parametrize(
    "text,expect",
    [
        ('{"a": 1,}', "invalid character '}' looking for beginning of object key string"),
        ('{"a": 1 "b": 2}', "invalid character '\"' after object key:value pair"),
        ("{", "unexpected end of JSON input"),
    ],
)
def test_go_style_json_errors(text, expect):
    try:
        json.loads(text)
    except ValueError as e:
        assert ct.go_json_error(text, e) == expect


def test_daemon_json_errors_and_default_runtime():
    st = make_cluster()
    sh = Shell(st)
    sh.run("ssh dgx-01")
    st.nodes["dgx-01"].files["/etc/docker/daemon.json"] = '{"default-runtime": "nvidia"}\n'
    assert "failed" in sh.run("systemctl restart docker")
    assert "specified default runtime 'nvidia' does not exist" in sh.run("journalctl -u docker")
    assert "Cannot connect to the Docker daemon" in sh.run("docker ps")


def test_compose_gpu_isolation_and_restart_policy():
    st = make_cluster()
    n = st.nodes["dgx-02"]
    n.files["/opt/s/compose.yaml"] = """\
services:
  a:
    image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
    restart: unless-stopped
    environment: {CUDA_VISIBLE_DEVICES: "2,3"}
  b:
    image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            device_ids: ["4", "5"]
            capabilities: [gpu]
  c:
    image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
    runtime: runc
"""
    sh = Shell(st)
    sh.run("ssh dgx-02")
    assert "no configuration file provided" in sh.run("docker compose up -d")
    out = sh.run("docker compose -f /opt/s/compose.yaml up -d")
    assert "s-a-1  Started" in out
    gpus = {c.name: c.gpus for c in n.containers}
    assert gpus == {"s-a-1": list(range(8)), "s-b-1": [4, 5], "s-c-1": []}
    assert "Running" in sh.run("docker compose -f /opt/s/compose.yaml up -d")
    assert "s-b-1" in sh.run("docker compose -f /opt/s/compose.yaml ps")
    assert sh.run("docker exec s-b-1 nvidia-smi -L").count("GPU ") == 2
    assert sh.run("docker inspect -f '{{.HostConfig.Runtime}}' s-a-1").strip() == "nvidia"
    # daemon restart without the nvidia runtime: restart-policy container cannot start
    n.files["/etc/docker/daemon.json"] = "{}\n"
    sh.run("systemctl restart docker")
    a = next(c for c in n.containers if c.name == "s-a-1")
    assert a.status.startswith("Exited")
    assert "unknown or invalid runtime name: nvidia" in sh.run("journalctl -u docker")
    assert "Removed" in sh.run("docker compose -f /opt/s/compose.yaml down")
    assert not any(c.get("compose_project") == "s" for c in n.containers)
