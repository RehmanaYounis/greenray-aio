"""Scenario variants: placeholders fill, other braces survive, every variant validates, retry cycles."""

import pytest

from greenray import cli
from greenray.content import scenarios as sc


def test_fill_touches_only_known_placeholders():
    text = "node {node} gpu {gpu} keep {unknown} and {'id': 1} and {}"
    assert sc.fill(text, {"node": "dgx-02", "gpu": 5}) == "node dgx-02 gpu 5 keep {unknown} and {'id': 1} and {}"
    assert sc.fill({"a": ["{x}", 3]}, {"x": "y"}) == {"a": ["y", 3]}


def test_variant_fills_problem_and_solution():
    raw = sc.find("tsh-xid79-drained-node")
    assert len(raw.variants) == 3 and raw.variant_label == ""
    v = raw.variant(1)
    node = raw.variants[1]["node"]
    assert node in v.title and node in v.briefing and "{node}" not in v.briefing
    assert all(node in c["expr"] or "{node}" not in c["expr"] for c in v.checks)
    assert any(node in step for step in v.solution) and "{node}" not in v.explanation
    assert v.variant_label == "variant 2 of 3"
    with pytest.raises(IndexError):
        raw.variant(3)


def test_raw_scenario_with_variants_cannot_be_built():
    with pytest.raises(ValueError):
        sc.find("tsh-xid79-drained-node").build_state()
    assert sc.find("adm-mig-h100-1g").all_variants()[0] is sc.find("adm-mig-h100-1g")


def test_pick_variant_avoids_the_current_one():
    raw = sc.find("tsh-xid79-drained-node")
    assert all(sc.pick_variant(raw, seed=i, avoid=0).variant_index != 0 for i in range(20))
    plain = sc.find("adm-mig-h100-1g")
    assert sc.pick_variant(plain) is plain


def test_cli_variant_flag_is_checked():
    assert cli.main(["--no-color", "lab", "tsh-xid79-drained-node", "--variant", "4"]) == 1
    assert cli.main(["--no-color", "lab", "adm-mig-h100-1g", "--variant", "2"]) == 1
