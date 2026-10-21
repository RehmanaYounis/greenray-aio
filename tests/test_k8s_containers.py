"""Unit tests for the Kubernetes / Helm / container tooling emulation."""

from greenray.commands import container as ct
from greenray.commands import helm
from greenray.commands import kubectl as kc
from greenray.core.shell import Shell
from greenray.core.state import make_cluster


def sh(state=None):
    return Shell(state or make_cluster())


def feed(lines):
    it = iter(lines + ["EOF"])
    return lambda _p: next(it)


def with_operator(**values):
    st = make_cluster()
    helm.preinstall(st, "nvidia/gpu-operator", values=values or {"driver": {"enabled": False}}, settle=600)
    return st


# ---------------------------------------------------------------- parsers


def test_label_selector_parsing():
    labels = {"app": "x", "tier": "gpu"}
    assert kc.selector_matches("app=x,tier", labels)
    assert not kc.selector_matches("app!=x", labels)
    assert kc.selector_matches("tier in (gpu,cpu)", labels)
    assert kc.selector_matches("!missing", labels)
    assert not kc.selector_matches("app=y", labels)


def test_tolerations_match_rules():
    taint = {"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"}
    assert kc.tolerates([{"key": "nvidia.com/gpu", "operator": "Exists"}], taint)
    assert kc.tolerates(
        [{"key": "nvidia.com/gpu", "operator": "Equal", "value": "present", "effect": "NoSchedule"}], taint
    )
    assert not kc.tolerates(
        [{"key": "nvidia.com/gpu", "operator": "Equal", "value": "true", "effect": "NoSchedule"}], taint
    )
    assert not kc.tolerates([{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoExecute"}], taint)
    assert kc.tolerates([{"operator": "Exists"}], taint)


def test_jsonpath_and_custom_columns():
    obj = {
        "items": [
            {
                "metadata": {"name": "a", "labels": {"nvidia.com/gpu.product": "H100"}},
                "status": {"allocatable": {"nvidia.com/gpu": "8"}},
            },
            {"metadata": {"name": "b", "labels": {}}, "status": {"allocatable": {}}},
        ]
    }
    assert kc.jsonpath(obj, "{.items[*].metadata.name}") == "a b"
    assert kc.jsonpath(obj, r"{.items[0].status.allocatable.nvidia\.com/gpu}") == "8"
    out = kc.jsonpath(obj, r'{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}')
    assert out == "a\t8\nb\t\n"
    table = kc.custom_columns(obj["items"], r"NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu")
    assert table.splitlines()[1].split() == ["a", "8"] and "<none>" in table


def test_helm_set_parsing():
    vals = helm.parse_sets(
        [
            "driver.enabled=false,mig.strategy=mixed",
            "daemonsets.tolerations[0].key=foo",
            r"nodeSelector.nvidia\.com/gpu=true",
            "list={a,b}",
            "n=3",
        ]
    )
    assert vals["driver"]["enabled"] is False and vals["mig"]["strategy"] == "mixed"
    assert vals["daemonsets"]["tolerations"][0]["key"] == "foo"
    assert vals["nodeSelector"]["nvidia.com/gpu"] is True
    assert vals["list"] == ["a", "b"] and vals["n"] == 3
    assert helm.parse_sets(["a=true"], as_string=True)["a"] == "true"


def test_image_reference_parsing():
    assert ct.parse_image("nvcr.io/nvidia/pytorch:24.05-py3") == ("nvcr.io", "nvidia/pytorch", "24.05-py3")
    assert ct.parse_image("ubuntu") == ("docker.io", "library/ubuntu", "latest")
    assert ct.norm_image("ubuntu:22.04") == "ubuntu:22.04"
    assert ct.parse_image("localhost:5000/team/img:1")[0] == "localhost:5000"


def test_containerd_config_roundtrip():
    stock = ct.parse_containerd(ct.containerd_config())
    assert stock == {"handlers": ["runc"], "default": "runc"}
    nv = ct.parse_containerd(ct.containerd_config(nvidia=True, default="nvidia"))
    assert "nvidia" in nv["handlers"] and nv["default"] == "nvidia"


def test_crash_backoff_counter():
    assert kc.crash_count(5)[0] == 0
    assert kc.crash_count(10)[0] == 1
    assert kc.crash_count(3600)[0] > 10


# ---------------------------------------------------------------- helm / operator


def test_helm_requires_repo_and_namespace():
    s = sh()
    assert "repo nvidia not found" in s.run(
        "helm install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace"
    )
    s.run("helm repo add nvidia https://helm.ngc.nvidia.com/nvidia")
    assert 'namespaces "gpu-operator" not found' in s.run(
        "helm install gpu-operator nvidia/gpu-operator -n gpu-operator"
    )
    assert "not a valid chart repository" in s.run("helm repo add bad https://example.com/charts")


def test_driver_enabled_on_dgx_crashloops_and_blocks_gpus():
    st = with_operator(driver={"enabled": True})
    drv = [p for p in st.k8s.pods if p.component == "driver"]
    assert drv and all(p.status == "CrashLoopBackOff" for p in drv)
    assert all(p.status.startswith("Init") for p in st.k8s.pods if p.component == "devicePlugin")
    assert "Could not unload NVIDIA driver kernel modules" in "\n".join(drv[0].logs)
    assert st.k8s.nodes["dgx-01"].allocatable.get("nvidia.com/gpu") in (None, "0")


def test_driver_disabled_advertises_gpus_and_configures_containerd():
    st = with_operator()
    assert all(st.k8s.nodes[n].allocatable["nvidia.com/gpu"] == "8" for n in ("dgx-01", "dgx-04"))
    assert st.nodes["dgx-01"].runtime_loaded.containerd.default == "nvidia"
    assert st.k8s.clusterpolicy.status.state == "ready"
    assert st.k8s.nodes["dgx-01"].labels["nvidia.com/gpu.product"] == "NVIDIA-H100-80GB-HBM3"


def test_toolkit_disabled_without_containerd_runtime_blocks_sandbox():
    st = with_operator(driver={"enabled": False}, toolkit={"enabled": False})
    dp = [p for p in st.k8s.pods if p.component == "devicePlugin"]
    assert dp and all(p.status == "ContainerCreating" for p in dp)


def test_helm_upgrade_without_reuse_values_resets_to_defaults():
    st = with_operator(driver={"enabled": False})
    rel = helm.find_release(st, "gpu-operator", "gpu-operator")
    helm.upgrade(st, rel, "nvidia/gpu-operator", {"mig": {"strategy": "mixed"}})
    assert rel.computed.driver.enabled is True
    helm.upgrade(st, rel, "nvidia/gpu-operator", {"driver": {"enabled": False}}, reuse=True)
    assert rel.computed.driver.enabled is False and rel.computed.mig.strategy == "mixed"


def test_mig_manager_and_strategies():
    st = with_operator()
    s = sh(st)
    assert "--overwrite is false" in s.run("kubectl label node dgx-04 nvidia.com/mig.config=all-1g.10gb")
    s.run("kubectl label node dgx-04 nvidia.com/mig.config=all-1g.10gb --overwrite")
    kn = st.k8s.nodes["dgx-04"]
    assert kn.labels["nvidia.com/mig.config.state"] == "success"
    assert kn.allocatable["nvidia.com/gpu"] == "56"  # single strategy
    assert kn.labels["nvidia.com/gpu.product"].endswith("-MIG-1g.10gb")
    s.run('kubectl patch clusterpolicy cluster-policy --type merge -p \'{"spec":{"mig":{"strategy":"mixed"}}}\'')
    assert kn.allocatable["nvidia.com/mig-1g.10gb"] == "56" and kn.allocatable["nvidia.com/gpu"] == "0"
    assert not any(g.mig_mode for g in st.nodes["dgx-01"].gpus)


# ---------------------------------------------------------------- scheduler / kubelet

POD = """apiVersion: v1
kind: Pod
metadata:
  name: p
spec:
  containers:
  - name: c
    image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
    command: ["sleep", "infinity"]
    resources:
      limits:
        nvidia.com/gpu: {n}
"""


def test_scheduler_message_and_immutable_pod_spec():
    st = with_operator()
    s = sh(st)
    s.run("cat > /root/p.yaml <<EOF", heredoc_reader=feed(POD.format(n=9).splitlines()))
    s.run("kubectl apply -f /root/p.yaml")
    pod = next(p for p in st.k8s.pods if p.name == "p")
    assert pod.status == "Pending"
    assert (
        "4 Insufficient nvidia.com/gpu" in pod.sched_msg
        and "{node-role.kubernetes.io/control-plane: }" in pod.sched_msg
    )
    s.run("sed -i 's/gpu: 9/gpu: 4/' /root/p.yaml")
    assert "Forbidden: pod updates may not change fields" in s.run("kubectl apply -f /root/p.yaml")
    s.run("kubectl delete pod p")
    s.run("kubectl apply -f /root/p.yaml")
    pod = next(p for p in st.k8s.pods if p.name == "p")
    assert pod.status == "Running" and len(pod.devices) == 4
    assert s.run("kubectl exec p -- nvidia-smi -L").count("GPU ") == 4


def test_cordon_taint_and_drain():
    st = with_operator()
    s = sh(st)
    s.run("kubectl cordon dgx-01")
    assert "SchedulingDisabled" in s.run("kubectl get nodes")
    s.run("kubectl taint nodes dgx-02 nvidia.com/gpu=present:NoSchedule")
    assert st.k8s.nodes["dgx-02"].taints[0].value == "present"
    assert "already has nvidia.com/gpu" in s.run("kubectl taint nodes dgx-02 nvidia.com/gpu=present:NoSchedule")
    s.run("kubectl taint nodes dgx-02 nvidia.com/gpu:NoSchedule-")
    assert not st.k8s.nodes["dgx-02"].taints
    assert "--ignore-daemonsets" in s.run("kubectl drain dgx-03")
    assert "drained" in s.run("kubectl drain dgx-03 --ignore-daemonsets")


def test_gpu_operator_pods_tolerate_gpu_taint():
    st = make_cluster()
    kc.k8s_init(st)
    for n in ("dgx-01", "dgx-02", "dgx-03", "dgx-04"):
        st.k8s.nodes[n].taints.append({"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"})
    helm.preinstall(st, "nvidia/gpu-operator", values={"driver": {"enabled": False}}, settle=60)
    assert len([p for p in st.k8s.pods if p.component == "devicePlugin" and p.status == "Running"]) == 4


def test_get_output_formats():
    st = with_operator()
    s = sh(st)
    assert "nvidia.com/gpu: '8'" in s.run("kubectl get node dgx-01 -o yaml")
    assert s.run("kubectl get node dgx-01 -o jsonpath='{.status.allocatable.nvidia\\.com/gpu}'") == "8"
    assert s.run("kubectl get nodes -o name").splitlines()[0] == "node/bcm-head"
    assert "No resources found in default namespace." in s.run("kubectl get pods")
    assert "localhost:8080 was refused" in sh(st).run("ssh dgx-01 kubectl get nodes")


# ---------------------------------------------------------------- registry / docker / ngc

KEY = "nvapi-TestKeyTestKeyTestKeyTestKey0123"


def private_state():
    st = make_cluster()
    st.ngc.private_repos = {"acme/llm/app": ["1.0"]}
    st.ngc.valid_api_keys = [KEY]
    return st


def test_docker_login_oauthtoken_quoting():
    s = sh(private_state())
    s.run("ssh dgx-01")
    assert "unauthorized" in s.run("docker pull nvcr.io/acme/llm/app:1.0")
    assert "requires at most 1 argument" in s.run(f"docker login nvcr.io -u $oauthtoken -p {KEY}")
    assert "non TTY" in s.run(f'docker login nvcr.io -u "$oauthtoken" -p {KEY}')
    assert "Login Succeeded" in s.run(f"docker login nvcr.io -u '$oauthtoken' -p {KEY}")
    assert "Downloaded newer image" in s.run("docker pull nvcr.io/acme/llm/app:1.0")
    assert "manifest unknown" in s.run("docker pull nvcr.io/nvidia/pytorch:99.99-py3")


def test_password_stdin_login():
    s = sh(private_state())
    s.run(f"echo {KEY} > /root/key")
    assert "Login Succeeded" in s.run("cat /root/key | docker login nvcr.io -u '$oauthtoken' --password-stdin")


def test_docker_run_gpu_selection():
    s = sh()
    s.run("ssh dgx-02")
    out = s.run("docker run --rm --gpus '\"device=2,3\"' nvcr.io/nvidia/pytorch:24.05-py3 nvidia-smi -L")
    assert out.count("GPU ") == 2 and "0203" in out
    assert "executable file not found" in s.run("docker run --rm ubuntu:22.04 nvidia-smi")


def test_ngc_config_and_pod_pull_secret():
    st = private_state()
    helm.preinstall(st, "nvidia/gpu-operator", values={"driver": {"enabled": False}}, settle=60)
    s = sh(st)
    assert "Missing API key" in s.run("ngc registry image list")
    s.run("ngc config set <<EOF", heredoc_reader=feed([KEY, "ascii", "acme", "llm", "no-ace"]))
    assert st.ngc.configured and st.ngc.org == "acme" and st.ngc.team == "llm"
    s.run("kubectl run app --image=nvcr.io/acme/llm/app:1.0")
    pod = next(p for p in st.k8s.pods if p.name == "app")
    assert pod.status in ("ErrImagePull", "ImagePullBackOff")
    s.run(
        f"kubectl create secret docker-registry ngc --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password={KEY}"
    )
    s.run('kubectl patch serviceaccount default -p \'{"imagePullSecrets":[{"name":"ngc"}]}\'')
    s.run("kubectl get pods")
    assert pod.status == "ImagePullBackOff"  # SA secrets only apply to new pods
    s.run("kubectl delete pod app")
    s.run("kubectl run app --image=nvcr.io/acme/llm/app:1.0")
    assert next(p for p in st.k8s.pods if p.name == "app").status == "Running"


def test_nvidia_ctk_needs_restart_and_enroot_uri():
    st = make_cluster()
    s = sh(st)
    s.run("ssh dgx-01")
    s.run("nvidia-ctk runtime configure --runtime=containerd --set-as-default")
    assert '"defaultRuntimeName": "runc"' in s.run("crictl info")
    s.run("systemctl restart containerd")
    assert '"defaultRuntimeName": "nvidia"' in s.run("crictl info")
    assert "401" in s.run("enroot import docker://nvcr.io/nvidia/pytorch:24.05-py3")
    assert "Creating 4.0 filesystem" in s.run("enroot import docker://nvcr.io#nvidia/pytorch:24.05-py3")
