"""Tests for cmsh (BCM), InfiniBand tools, NCCL tests and ops helpers."""

from greenray import commands

# Importing a command module directly fills the registry partially; dispatch() only
# loads every module when the registry is empty, so load them all explicitly first.
commands._load_all()

from greenray.commands.cmsh.model import _split_cmds, expand_nodes  # noqa: E402
from greenray.commands.ib import match_list, nccl_conf  # noqa: E402
from greenray.core.shell import Shell  # noqa: E402
from greenray.core.state import make_cluster  # noqa: E402

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


# ---------------------------------------------------------------- BCM-managed Slurm (wlm, overlays)

CONF = "/etc/slurm/slurm.conf"


def test_wlm_lists_the_instance_and_its_queues():
    s = sh()
    assert "slurm" in s.run('cmsh -c "wlm; list"') and "Slurm" in s.run('cmsh -c "wlm; use slurm; show"')
    out = s.run('cmsh -c "wlm; use slurm; jobqueue; list"')
    assert "defq" in out and "dgx-[01-04]" in out and "yes" in out


def test_jobqueue_commit_regenerates_file_and_controller():
    s = sh()
    out = s.run('cmsh -c "wlm; use slurm; jobqueue; add short; set nodes dgx-03,dgx-04; set maxtime 4:00:00; commit"')
    assert "slurmctld reconfigured" in out
    conf = s.ctx.state.nodes["bcm-head"].files[CONF]
    assert "PartitionName=short Nodes=dgx-[03-04] Default=NO MaxTime=4:00:00 State=UP" in conf
    assert s.ctx.state.slurm.partitions["short"].max_time == "4:00:00"
    assert "short" in s.run("sinfo")
    s.run('cmsh -c "wlm; use slurm; jobqueue; remove short; commit"')
    assert "short" not in s.ctx.state.nodes["bcm-head"].files[CONF] and "short" not in s.ctx.state.slurm.partitions


def test_jobqueue_properties_are_validated():
    s = sh()
    for cmd, msg in (
        ("use defq; set maxtime bogus", "invalid time"),
        ("use defq; set nodes dgx-09", "unknown compute node"),
        ("use defq; set default maybe", "yes or no"),
        ("use defq; set state SLEEPY", "state must be"),
        ("use defq; set name other", "read-only"),
    ):
        out = s.run(f'cmsh -c "wlm; use slurm; jobqueue; {cmd}" && echo OK')
        assert msg in out and "OK" not in out, (cmd, out)
    assert not s.ctx.state.bcm.uncommitted


def test_two_default_queues_warn_and_slurm_keeps_the_last():
    s = sh()
    out = s.run('cmsh -c "wlm; use slurm; jobqueue; add gpu; set nodes dgx-01; set default yes; commit"')
    assert "more than one default" in out
    assert s.ctx.state.slurm.partitions["gpu"].get("default") is True


def test_hand_added_partition_is_lost_when_cmd_restarts():
    s = sh()
    s.run(f"echo 'PartitionName=tmp Nodes=dgx-01 Default=NO MaxTime=1:00:00 State=UP' >> {CONF}")
    s.run("scontrol reconfigure")
    assert "tmp" in s.ctx.state.slurm.partitions
    s.run("systemctl restart cmd")
    assert "tmp" not in s.ctx.state.nodes["bcm-head"].files[CONF] and "tmp" not in s.ctx.state.slurm.partitions
    assert any("cmd started" in e for e in s.ctx.state.bcm.events)


def test_overlay_append_category_brings_node_into_slurm_with_gpus():
    s = sh()
    bcm = s.ctx.state.bcm
    s.run('cmsh -c "category; clone dgx-h100 dgx-h100-b; commit"')
    s.run('cmsh -c "device; use dgx-04; set category dgx-h100-b; commit"')
    s.run("systemctl restart cmd")
    assert "dgx-04" not in s.ctx.state.slurm.nodes
    assert "NodeName=dgx-[01-03]" in s.ctx.state.nodes["bcm-head"].files[CONF]
    s.run('cmsh -c "configurationoverlay; use slurm-client; append categories dgx-h100-b; commit"')
    sn = s.ctx.state.slurm.nodes["dgx-04"]
    assert sn.state == "IDLE" and sn.gres in ("", "(null)")  # the plain client overlay knows nothing about GPUs
    s.run('cmsh -c "configurationoverlay; use slurm-client-gpu; append categories dgx-h100-b; commit"')
    assert s.ctx.state.slurm.nodes["dgx-04"].gres == "gpu:h100:8"
    assert "dgx-h100-b" in s.run('cmsh -c "configurationoverlay; list"')
    assert "dgx-h100-b" in bcm.overlays["slurm-client-gpu"].categories


def test_overlay_validation_and_commit_scope():
    s = sh()
    out = s.run('cmsh -c "configurationoverlay; use slurm-client; append categories nope" && echo OK')
    assert "does not exist" in out and "OK" not in out
    out = s.run('cmsh -c "configurationoverlay; use slurm-client; set priority high" && echo OK')
    assert "OK" not in out
    # a commit applies to the selected object only; -a commits everything that is staged
    s.run(
        'cmsh -c "configurationoverlay; use slurm-client; set priority 510; use slurm-client-gpu; set priority 460; commit"'
    )
    bcm = s.ctx.state.bcm
    assert bcm.overlays["slurm-client-gpu"].priority == 460 and bcm.overlays["slurm-client"].priority == 500
    s.run('cmsh -c "configurationoverlay; commit -a"')
    assert bcm.overlays["slurm-client"].priority == 510 and not bcm.uncommitted


def test_foreach_and_range_apply_to_every_selected_node():
    s = sh()
    out = s.run('cmsh -c "device; foreach -c dgx-h100 (get category)"')
    assert out.count("dgx-h100") == 4 and "dgx-04: dgx-h100" in out
    s.run('cmsh -c "device; range -n dgx-[01-02]; set installmode FULL; commit"')
    devs = s.ctx.state.bcm.devices
    assert devs["dgx-01"].installmode == "FULL" and devs["dgx-02"].installmode == "FULL"
    assert devs["dgx-03"].installmode == "AUTO"
    s.run('cmsh -c "device; foreach -n dgx-[03-04] (set installmode FULL; commit)"')
    assert devs["dgx-03"].installmode == "FULL" and devs["dgx-04"].installmode == "FULL"
    assert not s.ctx.state.bcm.uncommitted


def test_kernelmodules_load_only_after_the_node_boots_from_the_image():
    s = sh()
    out = s.run('cmsh -c "softwareimage; use dgx-os-6.2-h100-image; kernelmodules; add gdrdrv; commit"')
    assert "Initial ramdisk" in out
    assert "gdrdrv" in s.ctx.state.bcm.softwareimages["dgx-os-6.2-h100-image"].kernelmodules
    assert "gdrdrv" not in s.ctx.state.nodes["dgx-02"].modules
    out = s.run('cmsh -c "softwareimage; use dgx-os-6.2-h100-image; kernelmodules; add nosuchmod" && echo OK')
    assert "not found" in out and "OK" not in out
    s.run('cmsh -c "device; reboot -n dgx-02"')
    assert "gdrdrv" in s.ctx.state.nodes["dgx-02"].modules
    assert "gdrdrv" in s.run("ssh dgx-02 lsmod")


def test_setup_wizards_point_at_cmsh():
    s = sh()
    assert "wlm" in s.run("cm-wlm-setup")
    assert "does not exist" in s.run("cm-kubernetes-setup --add-user nobody")
    s.run('cmsh -c "user; add dana; commit"')
    assert "dana-restricted" in s.run("cm-kubernetes-setup --add-user dana")
    assert "dana" in s.run("cm-kubernetes-setup --list-users")


# ---------------------------------------------------------------- BCM monitoring (triggers)


def _heat(s, host, temp):
    for g in s.ctx.state.nodes[host].gpus:
        g.temp_c = temp


HOT = "cmsh -c \"monitoring; trigger; add hot; set expression 'gpu_temp > 85'; set enteractions drain; set leaveactions undrain; commit\""


def test_monitoring_lists_measurables_actions_and_latest_data():
    s = sh()
    assert "gpu_temp" in s.run('cmsh -c "monitoring; measurable; list"')
    assert "drain" in s.run('cmsh -c "monitoring; action; list"')
    _heat(s, "dgx-02", 91)
    assert "91 C" in s.run('cmsh -c "device; latestmetricdata -n dgx-02"')
    assert "PASS" in s.run('cmsh -c "device; latesthealthdata -n dgx-02"')
    s.ctx.state.nodes["dgx-02"].services["slurmd"] = "failed"
    assert "FAIL" in s.run('cmsh -c "device; latesthealthdata -n dgx-02"')


def test_trigger_properties_are_validated():
    s = sh()
    s.run('cmsh -c "monitoring; trigger; add t"')  # staged, so the checks below run on an existing object
    for cmd, msg in (
        ("set expression 'gpu_temp >> 85'", "must look like"),
        ("set expression 'nosuch > 85'", "unknown measurable"),
        ("set expression 'gpu_temp > hot'", "not a number"),
        ("set expression 'ssh2node > 1'", "health check"),
        ("set enteractions nosuch", "unknown action"),
        ("set severity loud", "severity must be"),
    ):
        out = s.run(f'cmsh -c "monitoring; trigger; use t; {cmd}" && echo OK')
        assert msg in out and "OK" not in out, (cmd, out)
    out = s.run('cmsh -c "monitoring; action; remove drain" && echo OK')
    assert "built-in" in out and "OK" not in out


def test_trigger_drains_on_enter_and_undrains_on_leave():
    s = sh()
    _heat(s, "dgx-02", 91)
    out = s.run(HOT)
    assert "fired for dgx-02" in out and "drain: drained" in out
    sn = s.ctx.state.slurm.nodes["dgx-02"]
    assert sn.drain and sn.reason == "BCM trigger hot" and sn.state == "DRAIN"
    assert "dgx-02" in s.run('cmsh -c "monitoring; trigger; use hot; show"')
    s.run("sinfo")  # the condition still holds: no second firing
    assert sum("fired for dgx-02" in e for e in s.ctx.state.bcm.events) == 1
    _heat(s, "dgx-02", 40)
    s.run("sinfo")  # any command evaluates the triggers
    assert not sn.drain and sn.state == "IDLE"
    assert any("cleared for dgx-02" in e and "undrained" in e for e in s.ctx.state.bcm.events)
    assert any("monitoring: trigger hot" in line for line in s.ctx.state.nodes["bcm-head"].journal["cmd"])


def test_trigger_undrain_leaves_a_manual_drain_alone():
    s = sh()
    _heat(s, "dgx-03", 91)
    s.run(HOT)
    s.run("scontrol update nodename=dgx-03 state=drain reason=hands")
    _heat(s, "dgx-03", 40)
    s.run("sinfo")
    sn = s.ctx.state.slurm.nodes["dgx-03"]
    assert sn.drain and sn.reason == "hands"


def test_disabled_trigger_and_script_action():
    s = sh()
    s.run('cmsh -c "monitoring; action; add scriptaction page-oncall; set script /cm/local/page.sh; commit"')
    assert "/cm/local/page.sh" in s.run('cmsh -c "monitoring; action; list"')
    _heat(s, "dgx-04", 95)
    s.run(
        "cmsh -c \"monitoring; trigger; add hot; set expression 'gpu_temp > 85'; set enteractions drain,page-oncall; set disabled yes; commit\""
    )
    assert not s.ctx.state.slurm.nodes["dgx-04"].get("drain")
    out = s.run('cmsh -c "monitoring; trigger; use hot; set disabled no; commit"')
    assert "page-oncall: ran" in out and s.ctx.state.slurm.nodes["dgx-04"].drain


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
