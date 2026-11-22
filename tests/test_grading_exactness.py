"""Criteria grade what was asked (the role, the restart, the GPU count) and what the learner actually saw."""

from __future__ import annotations

import re

import pytest

from conftest import concrete, requires_solutions
from greenray.content import scenarios as sc
from greenray.core.shell import Shell

pytestmark = requires_solutions


def solved(code: str, variant: int = 0, edit=None, skip: str | None = None, first: tuple[str, ...] = ()):
    """The reference solution, with steps rewritten by ``edit`` or left out when they match ``skip``."""
    s = concrete(code, variant)
    shell = Shell(s.build_state())
    for line in first:
        shell.run(line.format(**s.params) if s.params else line)
    for step in s.solution:
        if skip and re.search(skip, step):
            continue
        lines = (edit(step) if edit else step).split("\n")
        feed = iter(lines[1:])
        shell.run(lines[0], heredoc_reader=lambda _p, feed=feed: next(feed, None))
    return s, shell


def marks(s: sc.Scenario, shell: Shell) -> dict[str, bool]:
    out = {}
    for desc, ok, err in s.grade(shell):
        assert not err, (desc, err)
        out[desc] = ok
    return out


def one(results: dict[str, bool], wording: str) -> bool:
    found = [ok for desc, ok in results.items() if wording in desc]
    assert len(found) == 1, (wording, list(results))
    return found[0]


# ---- adm-7: the requested roles and scopes, nothing more

ALICE = '--user alice@corp.ai --project nlp --role "L1 researcher"'
CAROL_NLP = '--user carol@corp.ai --project nlp --role "Research manager"'
RULE = "runai-ui access-rule create "


@pytest.mark.parametrize(
    ("change", "broken"),
    [
        (lambda st: st.replace(ALICE, ALICE.replace("L1 researcher", "Research manager")), "alice is a researcher"),
        (lambda st: st.replace(ALICE, ALICE.replace("nlp", "vision")), "alice is a researcher"),
        (lambda st: st.replace(ALICE, "--user alice@corp.ai,bob@corp.ai --project nlp"), "alice is a researcher"),
        (lambda st: st.replace(CAROL_NLP, CAROL_NLP.replace("Research manager", "L1 researcher")), "carol is Research"),
    ],
    ids=["alice-manager", "alice-on-vision", "bob-also-on-nlp", "carol-researcher"],
)
def test_adm7_a_wrong_role_or_scope_fails(change, broken):
    s, shell = solved("adm-7", edit=change)
    results = marks(s, shell)
    assert not one(results, broken)
    assert one(results, "vision-smoke is Running")


@pytest.mark.parametrize(
    ("extra", "broken"),
    [
        (RULE + "--user alice@corp.ai --project vision", "alice is a researcher"),
        (RULE + '--user carol@corp.ai --project sandbox --role "Research manager"', "carol is Research"),
        (RULE + '--user carol@corp.ai --project nlp --role "L1 researcher"', "carol is Research"),
    ],
    ids=["alice-extra-project", "carol-extra-project", "carol-second-role"],
)
def test_adm7_extra_access_fails(extra, broken):
    s, shell = solved("adm-7")
    assert all(marks(s, shell).values())
    shell.run(extra)
    assert not one(marks(s, shell), broken)


def test_adm7_either_researcher_role_and_the_default_role_pass():
    for change in (
        lambda st: st.replace(ALICE, ALICE.replace("L1 researcher", "L2 researcher")),
        lambda st: st.replace(ALICE, "--user alice@corp.ai --project nlp"),  # the role defaults to L1 researcher
    ):
        s, shell = solved("adm-7", edit=change)
        assert all(marks(s, shell).values()), marks(s, shell)


@pytest.mark.parametrize("code", ["adm-7"])
def test_reference_solution_still_earns_every_criterion(code):
    for s in sc.find(code).all_variants():
        shell = Shell(s.build_state())
        sc.replay_solution(s, shell)
        assert all(ok for _desc, ok, _err in s.grade(shell)), (s.variant_label, s.grade(shell))
