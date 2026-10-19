"""Tests for GPUDirect RDMA (nvidia_peermem), GDS / storage diagnostics, DCGM policy + injection,
and BlueField DPU tooling (mlxconfig, rshim, bfb-install, static pods on the DPU)."""

import pytest

from aiolab import commands

commands._load_all()

from aiolab.commands.dpu import attach_dpu  # noqa: E402
from aiolab.commands.gds import pipeline  # noqa: E402
from aiolab.core.shell import Shell  # noqa: E402
from aiolab.core.state import make_cluster  # noqa: E402

MPI = "mpirun --allow-run-as-root -np 16 -H dgx-01:8,dgx-02:8"
BFB = "/root/bfb/bf-bundle-2.7.0-33_24.04_ubuntu-22.04_prod.bfb"


def sh(state=None):
    return Shell(state or make_cluster())


# ---------------------------------------------------------------- GPUDirect RDMA


def _drop_peermem(s, hosts=("dgx-01", "dgx-02")):
    for h in hosts:
        n = s.ctx.state.nodes[h]
        n.modules = [m for m in n.modules if m != "nvidia_peermem"]
        n.volatile_modules = ["nvidia_peermem"]


def test_missing_peermem_disables_gdr_and_halves_busbw():
    s = sh()
    _drop_peermem(s)
    out = s.run(f"{MPI} -x NCCL_DEBUG=INFO all_reduce_perf -b 8G -e 8G -g 1")
    assert "GPU Direct RDMA Disabled for HCA 0 'mlx5_0' (no module)" in out
    assert "via NET/IB/0/GDRDMA" not in out and "via NET/IB/0" in out
    r = s.ctx.state.flags.nccl_last
    assert r.transport == "IB" and 150 < r.busbw < 220 and r.gdr == {"dgx-01": False, "dgx-02": False}


def test_modprobe_with_dash_restores_gdr():
    s = sh()
    _drop_peermem(s)
    for h in ("dgx-01", "dgx-02"):
        s.run(f"ssh {h} modprobe nvidia-peermem")
    out = s.run(f"{MPI} -x NCCL_DEBUG=INFO all_reduce_perf -b 8G -e 8G -g 1")
    assert "via NET/IB/0/GDRDMA" in out and "GPU Direct RDMA Disabled" not in out
    assert s.ctx.state.flags.nccl_last.busbw > 350


def test_volatile_module_needs_modules_load_d_to_survive_reboot():
    s = sh()
    _drop_peermem(s, ("dgx-01",))
    s.run("ssh dgx-01")
    s.run("modprobe nvidia-peermem")
    s.run("reboot")
    assert "nvidia_peermem" not in s.run("ssh dgx-01 lsmod")
    s.run("ssh dgx-01")
    s.run("modprobe nvidia_peermem")
    s.run("echo nvidia-peermem > /etc/modules-load.d/nvidia-peermem.conf")
    s.run("reboot")
    assert any(m.replace("-", "_") == "nvidia_peermem" for m in s.ctx.state.nodes["dgx-01"].modules)


# ---------------------------------------------------------------- DCGM policy / injection


def _dcgm_ready(s, host="dgx-04"):
    s.run(f"ssh {host}")
    s.run("dcgmi group -c prod --default")
    s.run("dcgmi health -g 2 -s a")


def test_policy_set_get_and_validation():
    s = sh()
    _dcgm_ready(s)
    assert "successfully set" in s.run("dcgmi policy -g 2 --set 0,0 -e -x -T 85 -P 650")
    out = s.run("dcgmi policy -g 2 --get")
    assert "Double-bit ECC errors" in out and "Max temperature - 85 C" in out and "Max power - 650 W" in out
    assert "invalid --set" in s.run("dcgmi policy -g 2 --set 5,0 -e")
    assert "numeric threshold" in s.run("dcgmi policy -g 2 --set 0,0 -T")


def test_injected_dbe_reaches_listener_and_health_but_not_nvidia_smi():
    s = sh()
    _dcgm_ready(s)
    s.run("dcgmi policy -g 2 --set 0,0 -e -x")
    s.run("dcgmi policy -g 2 --reg")
    out = s.run("dcgmi test --inject --gpuid 3 -f 311 -v 2")
    assert "Successfully injected field info." in out and "double-bit ECC error has violated" in out
    v = s.ctx.state.dcgm.policy_violations["dgx-04"]
    assert v[0]["gpu"] == 3 and v[0]["condition"] == "DBE" and v[0]["action"] == "None"
    assert "Failure" in s.run("dcgmi health -g 2 -c")
    assert s.ctx.state.nodes["dgx-04"].gpus[3].ecc_volatile_dbe == 0  # only DCGM's cache changed
    # a second identical injection is not reported twice
    s.run("dcgmi test --inject --gpuid 3 -f 311 -v 2")
    assert len(s.ctx.state.dcgm.policy_violations["dgx-04"]) == 1


def test_unregistered_policy_records_nothing_and_restart_forgets_injection():
    s = sh()
    _dcgm_ready(s)
    s.run("dcgmi policy -g 2 --set 0,0 -T 80")
    s.run("dcgmi test --inject --gpuid 0 -f 150 -v 92")
    assert not s.ctx.state.dcgm.policy_violations.get("dgx-04")
    assert "Thermal" in s.run("dcgmi health -g 2 -c")
    s.run("systemctl restart nvidia-dcgm")
    assert not s.ctx.state.dcgm.injected.get("dgx-04")
    assert "not found" in s.run("dcgmi health -g 2 -c")


def test_inject_rejects_unknown_field():
    s = sh()
    s.run("ssh dgx-01")
    assert "cannot be injected" in s.run("dcgmi test --inject --gpuid 0 -f 999 -v 1")


# ---------------------------------------------------------------- GDS / storage


def _storage_state(rsize=8192):
    st = make_cluster()
    n = st.nodes["dgx-02"]
    n.mounts = [
        {"source": "master:/home", "target": "/home", "fstype": "nfs", "options": f"rw,rsize={rsize},proto=tcp"},
        {"source": "mgs@o2ib:/exafs", "target": "/lustre", "fstype": "lustre", "options": "rw,flock"},
    ]
    for root in ("/home/ds", "/lustre/ds"):
        n.files[root + "/MANIFEST"] = "x\n"
    n.files["/job.env"] = "DATA_DIR=/home/ds\nNUM_WORKERS=2\nUSE_GDS=1\n"
    n.training = {"config": "/job.env", "gpus": 8, "gpu_rate": 3000, "sample_kb": 150, "launched": {}}
    return st


def test_pipeline_model_limits():
    st = _storage_state()
    n = st.nodes["dgx-02"]
    p = pipeline(n, {"DATA_DIR": "/home/ds", "NUM_WORKERS": "2", "USE_GDS": "1"})
    assert p["bound"] == "storage" and not p["gds"] and 1900 < p["rate"] < 2100
    p = pipeline(n, {"DATA_DIR": "/lustre/ds", "NUM_WORKERS": "8", "USE_GDS": "1"})
    assert p["bound"] == "cpu" and not p["gds"]  # compat mode costs CPU per sample
    n.modules.append("nvidia-fs")
    p = pipeline(n, {"DATA_DIR": "/lustre/ds", "NUM_WORKERS": "8", "USE_GDS": "1"})
    assert p["bound"] == "gpu" and p["gds"]
    p = pipeline(n, {"DATA_DIR": "/home/ds", "NUM_WORKERS": "8", "USE_GDS": "1"})
    assert not p["gds"] and "NFS over TCP" in p["gds_reason"]


def test_gdscheck_reflects_nvidia_fs_and_mounts():
    s = sh(_storage_state())
    s.run("ssh dgx-02")
    out = s.run("/usr/local/cuda/gds/tools/gdscheck -p")
    assert "DDN EXAScaler      : Unsupported" in out and "not loaded" in out
    s.run("modprobe nvidia_fs")
    out = s.run("gdscheck -p")
    assert "DDN EXAScaler      : Supported" in out and "NFS                : Unsupported" in out
    assert "Platform verification succeeded" in out


def test_dataloader_bench_records_overrides_and_compat_off_errors():
    st = _storage_state()
    s = sh(st)
    s.run("ssh dgx-02")
    out = s.run("dataloader-bench --data /lustre/ds --workers 16")
    assert "command-line overrides" in out
    assert st.flags.dataloader_bench["dgx-02"]["from_config"] is False
    st.nodes["dgx-02"].files["/etc/cufile.json"] = '{"properties": {"allow_compat_mode": false}}'
    assert "compat mode disabled" in s.run("dataloader-bench")
    assert "No such file" in s.run("dataloader-bench --data /nowhere")


def test_dmon_without_training_is_unchanged_and_with_training_oscillates():
    s = sh()
    s.run("ssh dgx-01")
    assert len(s.run("nvidia-smi dmon").strip().splitlines()) == 2 + 8
    st = _storage_state()
    st.nodes["dgx-02"].training.launched = {"DATA_DIR": "/home/ds", "NUM_WORKERS": "2", "USE_GDS": "1"}
    s = sh(st)
    s.run("ssh dgx-02")
    rows = [l.split() for l in s.run("nvidia-smi dmon -c 10 -i 0").splitlines() if not l.startswith("#")]
    sm = [int(r[4]) for r in rows]
    assert len(sm) == 10 and max(sm) > 90 and min(sm) < 10


def test_nfsiostat_and_iostat():
    st = _storage_state()
    st.nodes["dgx-02"].training.launched = {"DATA_DIR": "/home/ds", "NUM_WORKERS": "2", "USE_GDS": "1"}
    s = sh(st)
    s.run("ssh dgx-02")
    out = s.run("nfsiostat /home")
    assert "master:/home mounted on /home" in out and "8.000" in out
    assert "md1" in s.run("iostat -x")
    idle = sh().run("ssh dgx-01 nfsiostat /home")  # default BCM mounts come from storage.py
    assert "mounted on /home" in idle and "0.000" in idle


# ---------------------------------------------------------------- BlueField DPU


def _dpu_state(**kw):
    st = make_cluster()
    attach_dpu(st, **kw)
    st.nodes["dgx-01"].files[BFB] = "bfb\n"
    return st


def test_mlxconfig_query_set_needs_confirmation_and_reboot():
    st = _dpu_state(nic_mode=True)
    s = sh(st)
    s.run("ssh dgx-01")
    assert "run 'mst start'" in s.run("mlxconfig -d /dev/mst/mt41692_pciconf0 q")
    out = s.run("mlxconfig -d c1:00.0 q")
    assert "INTERNAL_CPU_OFFLOAD_ENGINE                 DISABLED(1)" in out
    assert "Aborted by user" in s.run("mlxconfig -d c1:00.0 s INTERNAL_CPU_OFFLOAD_ENGINE=0")
    s.run("echo y | mlxconfig -d c1:00.0 r")
    d = st.nodes["dgx-01"].dpus[0]
    assert d.cfg_next["INTERNAL_CPU_OFFLOAD_ENGINE"] == 0 and d.cfg["INTERNAL_CPU_OFFLOAD_ENGINE"] == 1
    assert "Invalid value" in s.run("mlxconfig -d c1:00.0 -y s INTERNAL_CPU_OFFLOAD_ENGINE=MAYBE")
    s.run("reboot")
    assert d.cfg["INTERNAL_CPU_OFFLOAD_ENGINE"] == 0
    assert "ConnectX7" in s.run("ssh dgx-01 mlxconfig -d mlx5_0 q")


def test_bfb_install_needs_rshim_and_dpu_mode():
    st = _dpu_state(nic_mode=True)
    s = sh(st)
    s.run("ssh dgx-01")
    assert "rshim service" in s.run(f"bfb-install --bfb {BFB} --rshim rshim0")
    s.run("systemctl start rshim")
    assert "operation mode" in s.run(f"bfb-install --bfb {BFB} --rshim rshim0")
    assert not st.nodes["dpu-01"].up


def test_bfb_install_reimages_dpu_and_static_pods_run():
    st = _dpu_state(nic_mode=False, rshim="active")
    s = sh(st)
    s.run("ssh dgx-01")
    assert "Installation finished" in s.run(f"bfb-install --bfb {BFB} --rshim /dev/rshim0")
    assert st.nodes["dpu-01"].up and st.nodes["dgx-01"].dpus[0].bfb == "2.7.0"
    st.nodes["dgx-01"].files["/root/dts.yaml"] = (
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: dts\nspec:\n  containers:\n"
        "  - name: dts\n    image: nvcr.io/nvidia/doca/doca_telemetry:1.17.2-doca2.7.0\n"
    )
    st.nodes["dgx-01"].files["/root/bad.yaml"] = (
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: bad\nspec:\n  containers:\n"
        "  - name: bad\n    image: nvcr.io/nvidia/doca/doca_telemetry:9.9.9\n"
    )
    s.run("scp /root/dts.yaml dpu-01:/etc/kubelet.d/")
    s.run("scp /root/bad.yaml dpu-01:/etc/kubelet.d/bad.yaml")
    s.run("exit")
    s.run("ssh dpu-01")
    assert "NotReady" in s.run("crictl pods")
    out = s.run("crictl ps")
    assert "dts-dpu-01" in out and "bad-dpu-01" not in out
    assert st.flags.dpu_running["dpu-01"] == ["dts"]
    assert "not found" in s.run("crictl pull nvcr.io/nvidia/doca/doca_telemetry:9.9.9")
    assert "doca_telemetry" in s.run("crictl images")
    assert "ErrImagePull" not in s.run("journalctl -u kubelet") and "not found" in s.run("journalctl -u kubelet")


def test_crictl_falls_through_on_non_dpu_nodes():
    s = sh(_dpu_state())
    s.run("ssh dgx-02")
    out = s.run("crictl ps")
    assert out.startswith("CONTAINER")


def test_dpu_unreachable_until_bfb_installed():
    s = sh(_dpu_state())
    assert "No route to host" in s.run("ssh dpu-01")
    s.run("ssh dgx-01")
    assert "No route to host" in s.run("scp /etc/hostname dpu-01:/tmp/")


def test_attach_dpu_validates_host():
    with pytest.raises(KeyError):
        attach_dpu(make_cluster(), host="nope")
