"""Tests for cmsh (BCM), InfiniBand tools, NCCL tests and ops helpers."""

from aiolab import commands

# Importing a command module directly fills the registry partially; dispatch() only
# loads every module when the registry is empty, so load them all explicitly first.
commands._load_all()

from aiolab.commands.cmsh import _split_cmds, expand_nodes  # noqa: E402
from aiolab.commands.ib import match_list, nccl_conf  # noqa: E402
from aiolab.core.shell import Shell  # noqa: E402
from aiolab.core.state import make_cluster  # noqa: E402

MPI = "mpirun --allow-run-as-root -np 16 -H dgx-01:8,dgx-02:8"


def sh():
    return Shell(make_cluster())


# ---------------------------------------------------------------- parsers


def test_expand_nodes():
    assert expand_nodes("dgx-01..dgx-03") == ["dgx-01", "dgx-02", "dgx-03"]
    assert expand_nodes("dgx-0[1-2],bcm-head") == ["dgx-01", "dgx-02", "bcm-head"]


def test_split_cmds_respects_quotes():
    assert _split_cmds("device; pexec -n dgx-01 'a; b'; list") == ["device", "pexec -n dgx-01 'a; b'", "list"]


def test_nccl_list_semantics():
    names = ["mlx5_0", "mlx5_1", "mlx5_10", "mlx5_2"]
    assert match_list("mlx5_1", names) == ["mlx5_1", "mlx5_10"]  # prefix
    assert match_list("=mlx5_1", names) == ["mlx5_1"]  # exact
    assert match_list("^mlx5_1", names) == ["mlx5_0", "mlx5_2"]  # exclude
    assert match_list("mlx5_0:1,mlx5_2:1", names) == ["mlx5_0", "mlx5_2"]
    assert match_list("mlx5_bond_0", names) == []
    assert match_list("", names) == names


def test_nccl_conf_parsing():
    s = make_cluster()
    s.nodes["dgx-01"].files["/etc/nccl.conf"] = "# c\nexport NCCL_IB_HCA=mlx5\nNCCL_DEBUG = INFO # x\n"
    assert nccl_conf(s.nodes["dgx-01"]) == {"NCCL_IB_HCA": "mlx5", "NCCL_DEBUG": "INFO"}


# ---------------------------------------------------------------- cmsh


def test_cmsh_only_on_head():
    s = sh()
    s.run("ssh dgx-01")
    assert "command not found" in s.run('cmsh -c "device; list"')


def test_cmsh_set_is_staged_until_commit():
    s = sh()
    bcm = s.ctx.state.bcm
    out = s.run('cmsh -c "device; use dgx-03; set category default"')
    assert "uncommitted" in out and "*" in out
    assert bcm.devices["dgx-03"].category == "dgx-h100"
    assert "default" in s.run('cmsh -c "device; use dgx-03; get category"')  # visible in session
    s.run('cmsh -c "device; commit"')  # later invocation
    assert bcm.devices["dgx-03"].category == "default" and not bcm.uncommitted


def test_cmsh_refresh_discards():
    s = sh()
    s.run('cmsh -c "device; use dgx-03; set category default"')
    s.run('cmsh -c "device; use dgx-03; refresh"')
    assert not s.ctx.state.bcm.uncommitted


def test_cmsh_errors_set_exit_code():
    s = sh()
    out = s.run('cmsh -c "device; use dgx-01; set category nope; commit" && echo OK')
    assert "does not exist" in out and "OK" not in out


def test_cmsh_power_off_on():
    s = sh()
    st = s.ctx.state
    s.run('cmsh -c "device; power off -n dgx-02"')
    assert not st.nodes["dgx-02"].up and st.nodes["dgx-02"].power == "off"
    assert "DOWN" in s.run('cmsh -c "device; status -n dgx-02"')
    assert "No route" in s.run("ssh dgx-02")
    s.run('cmsh -c "device; power on -n dgx-02"')
    assert st.nodes["dgx-02"].up and st.bcm.devices["dgx-02"].status == "UP"


def test_cmsh_bmc_override_breaks_power():
    s = sh()
    s.ctx.state.bcm.devices["dgx-01"].bmcsettings = {"password": "wrong"}
    assert "FAILED" in s.run('cmsh -c "device; power status -n dgx-01"')
    s.run('cmsh -c "device; use dgx-01; bmcsettings; clear password; commit"')
    assert "[    ON    ] dgx-01" in s.run('cmsh -c "device; power status -n dgx-01"')


def test_softwareimage_clone_chroot_and_imageupdate():
    s = sh()
    st = s.ctx.state
    s.run('cmsh -c "softwareimage; clone dgx-os-6.2-h100-image img2"')
    # not committed yet: directory does not exist
    assert "No such file" in s.run("chroot /cm/images/img2 apt-get install -y nvidia-container-toolkit")
    s.run('cmsh -c "softwareimage; commit"')
    s.run("chroot /cm/images/img2 apt-get install -y nvidia-container-toolkit=1.16.2-1")
    assert st.bcm.softwareimages["img2"].packages["nvidia-container-toolkit"] == "1.16.2-1"
    assert st.bcm.softwareimages["dgx-os-6.2-h100-image"].packages["nvidia-container-toolkit"] == "1.14.6-1"
    s.run('cmsh -c "category; use dgx-h100; set softwareimage img2; commit"')
    dry = s.run('cmsh -c "device; imageupdate -n dgx-01"')
    assert "dry run = yes" in dry
    assert st.nodes["dgx-01"].get("packages", {}).get("nvidia-container-toolkit") != "1.16.2-1"
    s.run('cmsh -c "device; imageupdate -w -n dgx-01"')
    assert st.nodes["dgx-01"].packages["nvidia-container-toolkit"] == "1.16.2-1"
    assert st.bcm.devices["dgx-01"].provisioned_image_version == 2


def test_kernelversion_must_exist_in_image():
    s = sh()
    out = s.run('cmsh -c "softwareimage; use dgx-os-6.2-h100-image; set kernelversion 5.15.0-1053-nvidia"')
    assert "not found" in out
    s.run("chroot /cm/images/dgx-os-6.2-h100-image apt-get install -y linux-image-5.15.0-1053-nvidia")
    out = s.run('cmsh -c "softwareimage; use dgx-os-6.2-h100-image; set kernelversion 5.15.0-1053-nvidia; commit"')
    assert "ramdisk" in out
    assert s.ctx.state.bcm.softwareimages["dgx-os-6.2-h100-image"].kernel == "5.15.0-1053-nvidia"


def test_chroot_edits_image_files_not_host():
    s = sh()
    s.run("echo hello > /cm/images/dgx-os-6.2-h100-image/etc/motd")
    s.run("chroot /cm/images/dgx-os-6.2-h100-image sed -i s/hello/bye/ /etc/motd")
    files = s.ctx.state.nodes["bcm-head"].files
    assert files["/cm/images/dgx-os-6.2-h100-image/etc/motd"] == "bye\n"
    assert "/etc/motd" not in files


def test_cmsh_drain_sets_slurm_drain():
    s = sh()
    s.run("cmsh -c \"device; drain -n dgx-02 --reason 'fan'\"")
    sn = s.ctx.state.slurm.nodes["dgx-02"]
    assert sn.state == "DRAIN" and sn.reason == "fan"


# ---------------------------------------------------------------- InfiniBand


def _down_port(s, host="dgx-03", idx=3):
    p = s.ctx.state.nodes[host].ib[idx]
    p.state, p.phys_state, p.cable_ok = "Down", "Polling", False
    return p


def test_ibstat_down_port_shows_rate_10():
    s = sh()
    _down_port(s)
    s.run("ssh dgx-03")
    out = s.run("ibstat mlx5_3")
    assert "State: Down" in out and "Physical state: Polling" in out and "Rate: 10" in out
    assert "4X NDR" in s.run("ibstatus mlx5_0")
    assert "ibp92s0" in s.run("ip -br a") and "DOWN" in s.run("ip -br a | grep ibp92s0")


def test_mlxlink_toggle_vs_reseat():
    s = sh()
    p = _down_port(s)
    s.run("ssh dgx-03")
    assert "-40.00" in s.run("mlxlink -d mlx5_3 -m")
    s.run("mlxlink -d mlx5_3 --port_state TG")
    assert p.state == "Down"  # no light: toggling does not help
    s.run("reseat-cable dgx-03 mlx5_3")
    assert p.state == "Active" and p.rate_gbps == 400
    s.run("mlxlink -d mlx5_3 --port_state DN")
    assert p.phys_state == "Disabled"
    s.run("mlxlink -d /dev/mst/mt4129_pciconf3 --port_state UP")
    assert p.state == "Active"


def test_ibdiagnet_reports_speed_and_down_links():
    s = sh()
    _down_port(s)
    s.ctx.state.nodes["dgx-02"].ib[5].rate_gbps = 200
    s.ctx.state.nodes["dgx-02"].ib[5].width = "2X"
    out = s.run("ibdiagnet")
    assert "dgx-03/mlx5_3" in out and "200 Gb/s" in out


def test_perfquery_counters_and_reset():
    s = sh()
    s.ctx.state.nodes["dgx-01"].ib[2].symbol_errors = 77
    s.run("ssh dgx-01")
    assert "SymbolErrorCounter:..............77" in s.run("perfquery -C mlx5_2")
    s.run("perfquery -C mlx5_2 -R")
    assert s.ctx.state.nodes["dgx-01"].ib[2].symbol_errors == 0


# ---------------------------------------------------------------- NCCL


def test_single_node_allreduce_nvlink():
    s = sh()
    s.run("ssh dgx-01")
    s.run("all_reduce_perf -b 8 -e 8G -f 2 -g 8")
    r = s.ctx.state.flags.nccl_last
    assert r.ok and r.transport == "NVLink" and 470 <= r.busbw <= 485


def test_mpirun_requires_allow_root():
    assert "run as root" in sh().run("mpirun -np 2 -H dgx-01,dgx-02 all_reduce_perf")


def test_multinode_ib_and_bad_hca_fallback():
    s = sh()
    s.run(f"{MPI} all_reduce_perf -b 8G -e 8G -g 1")
    assert s.ctx.state.flags.nccl_last.transport == "IB" and s.ctx.state.flags.nccl_last.busbw > 350
    for n in ("dgx-01", "dgx-02"):
        s.ctx.state.nodes[n].files["/etc/nccl.conf"] = "NCCL_IB_HCA=mlx5_bond_0\n"
    out = s.run(f"{MPI} -x NCCL_DEBUG=INFO all_reduce_perf -b 8G -e 8G -g 1")
    assert "Using network Socket" in out and "NCCL WARN" in out
    assert s.ctx.state.flags.nccl_last.busbw < 5
    s.run("export NCCL_IB_HCA=mlx5")  # head-node env does not reach remote ranks
    s.run(f"{MPI} all_reduce_perf -b 8G -e 8G -g 1")
    assert s.ctx.state.flags.nccl_last.transport == "Socket"
    s.run(f"{MPI} -x NCCL_IB_HCA all_reduce_perf -b 8G -e 8G -g 1")
    assert s.ctx.state.flags.nccl_last.transport == "IB"


def test_down_port_reduces_bandwidth():
    s = sh()
    _down_port(s, "dgx-02", 3)
    s.run(f"{MPI} all_reduce_perf -b 8G -e 8G -g 1")
    assert 300 < s.ctx.state.flags.nccl_last.busbw < 340


def test_bad_socket_ifname_fails():
    s = sh()
    s.run("ssh dgx-01")
    s.run("export NCCL_SOCKET_IFNAME=eth0")
    assert "unhandled system error" in s.run("all_reduce_perf -b 8 -e 1M -g 8")


# ---------------------------------------------------------------- Fabric Manager / ops


def test_fabricmanager_down_gives_802_and_version_mismatch_fails_start():
    s = sh()
    n = s.ctx.state.nodes["dgx-01"]
    n.services["nvidia-fabricmanager"] = "inactive"
    s.run("ssh dgx-01")
    assert "802" in s.run("deviceQuery")
    assert "system not yet initialized" in s.run("all_reduce_perf -g 8")
    n.fm_version = "550.54.15"
    out = s.run("systemctl start nvidia-fabricmanager")
    assert n.services["nvidia-fabricmanager"] == "failed" and "failed" in out
    assert "don't match" in s.run("journalctl -u nvidia-fabricmanager")
    n.fm_version = n.driver_version
    s.run("systemctl start nvidia-fabricmanager")
    assert "Result = PASS" in s.run("deviceQuery")


def test_disabled_fabricmanager_stays_down_after_reboot():
    s = sh()
    n = s.ctx.state.nodes["dgx-01"]
    s.run("ssh dgx-01")
    s.run("systemctl disable nvidia-fabricmanager")
    s.run("reboot")
    assert n.services["nvidia-fabricmanager"] == "inactive"


def test_hwticket_and_bug_report():
    s = sh()
    assert "at least 10" in s.run("hwticket create --node dgx-02 --component fan --summary x")
    s.run("ssh dgx-02")
    s.run("nvidia-bug-report.sh")
    s.run("exit")
    out = s.run('hwticket create --node dgx-02 --component fan --summary "FAN6 failed, GPU5 hot"')
    assert "HW-1041" in out and "LAB HELPER" in out
    t = s.ctx.state.flags.hwtickets[0]
    assert t.bug_report_attached and t.component == "fan"
