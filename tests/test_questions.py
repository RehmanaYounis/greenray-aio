from conftest import requires_solutions
from greenray.content import questions as qb
from greenray.content.domains import DOMAINS


def test_bank_covers_every_domain():
    assert {q["domain"] for q in qb.load_bank()} == set(DOMAINS)


def test_pick_follows_blueprint_weights():
    qs = qb.pick(qb.load_bank(), 30, seed=1)
    assert len(qs) == 30
    by_domain = {d: sum(q["domain"] == d for q in qs) for d in DOMAINS}
    assert by_domain["Installation and Deployment"] >= 9


def test_pick_single_domain():
    qs = qb.pick(qb.load_bank(), 10, domain="Administration", seed=2)
    assert {q["domain"] for q in qs} == {"Administration"}


def test_scoring_and_parsing():
    qs = [{"id": "a", "domain": DOMAINS[0], "answer": [1]}, {"id": "b", "domain": DOMAINS[1], "answer": [0, 2]}]
    res = qb.score(qs, {"a": {1}, "b": {0}})
    assert res["correct"] == 1 and res["missed"] == ["b"]
    assert qb.parse_answer("a c", 4) == {0, 2}
    assert qb.parse_answer("A,C", 4) == {0, 2}
    assert qb.parse_answer("Z", 4) is None
    assert qb.parse_answer("", 4) is None


@requires_solutions
def test_answer_key_complete():
    bank = qb.load_bank()
    assert qb.has_answers(bank)
    assert all(q.get("explanation") for q in bank)
