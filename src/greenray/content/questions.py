"""Exam question bank.

Public files in ``greenray/data/questions/*.yaml`` hold the questions::

    - id: adm-001
      domain: Administration
      difficulty: medium          # easy | medium | hard
      question: |
        ...scenario text...
      choices: [first option, second option, ...]
      lab: adm-mig-h100-1g         # optional: scenario that practises it

The answer key (``answer`` as zero-based indexes, ``explanation``) for each file
lives in the private solutions repository under ``questions/<same file name>``.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import yaml

from . import solutions
from .domains import DOMAIN_WEIGHTS, DOMAINS

BANK_DIR = Path(__file__).resolve().parent.parent / "data" / "questions"
LETTERS = "ABCDEFGH"
PUBLIC_KEYS = {"id", "domain", "difficulty", "question", "choices", "lab"}
PRIVATE_KEYS = {"answer", "explanation"}


def validate(q: dict) -> None:
    where = f"{q.get('_file')}:{q.get('id')}"
    for key in ("id", "domain", "question", "choices"):
        if key not in q:
            raise ValueError(f"{where}: missing '{key}'")
    unknown = set(q) - PUBLIC_KEYS - PRIVATE_KEYS - {"_file"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    if q["domain"] not in DOMAINS:
        raise ValueError(f"{where}: bad domain {q['domain']!r}")
    if not (2 <= len(q["choices"]) <= len(LETTERS)):
        raise ValueError(f"{where}: needs 2-{len(LETTERS)} choices")
    if "answer" in q:
        if not q["answer"] or any(not (0 <= a < len(q["choices"])) for a in q["answer"]):
            raise ValueError(f"{where}: answer index out of range")


@lru_cache(maxsize=1)
def _load_cached() -> tuple[dict, ...]:
    qs = []
    for path in sorted(BANK_DIR.glob("*.yaml")):
        key = solutions.answer_key(path.name)
        for q in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            leaked = PRIVATE_KEYS & set(q)
            if leaked:
                raise ValueError(f"{path.name}:{q.get('id')}: {sorted(leaked)} belong in the private answer key")
            q = {**q, **key.get(q["id"], {}), "_file": path.name}
            validate(q)
            qs.append(q)
    ids = [q["id"] for q in qs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate question ids: {sorted(dupes)}")
    return tuple(qs)


def load_bank() -> list[dict]:
    """All questions, with answers merged in when the private answer key is installed."""
    return [dict(q) for q in _load_cached()]


def has_answers(bank: list[dict]) -> bool:
    return bool(bank) and all("answer" in q for q in bank)


def pick(bank: list[dict], n: int, domain: str | None = None, seed: int | None = None) -> list[dict]:
    """Pick ``n`` questions: one domain, or all domains in blueprint proportions."""
    rng = random.Random(seed)
    if domain:
        pool = [q for q in bank if q["domain"] == domain]
        rng.shuffle(pool)
        return pool[:n]
    by_dom = {d: [q for q in bank if q["domain"] == d] for d in DOMAINS}
    for pool in by_dom.values():
        rng.shuffle(pool)
    quotas = {d: round(n * w) for d, w in DOMAIN_WEIGHTS.items()}
    chosen = [q for d, k in quotas.items() for q in by_dom[d][:k]]
    leftovers = [q for d in DOMAINS for q in by_dom[d][quotas[d] :]]
    rng.shuffle(leftovers)
    chosen += leftovers[: max(0, n - len(chosen))]
    rng.shuffle(chosen)
    return chosen[:n]


def parse_answer(text: str, nchoices: int) -> set[int] | None:
    """``"a c"`` / ``"A,C"`` / ``"ac"`` -> ``{0, 2}``; ``None`` if invalid."""
    out = set()
    for tok in text.replace(",", " ").replace(";", " ").upper().split():
        for ch in tok:
            idx = LETTERS.find(ch)
            if idx < 0 or idx >= nchoices:
                return None
            out.add(idx)
    return out or None


def score(questions: list[dict], answers: dict) -> dict:
    per = {d: [0, 0] for d in DOMAINS}
    missed = []
    for q in questions:
        per[q["domain"]][1] += 1
        if answers.get(q["id"]) == set(q["answer"]):
            per[q["domain"]][0] += 1
        else:
            missed.append(q["id"])
    correct = sum(c for c, _ in per.values())
    total = len(questions)
    return {
        "correct": correct,
        "total": total,
        "pct": 100.0 * correct / total if total else 0.0,
        "per_domain": {d: f"{c}/{t}" for d, (c, t) in per.items() if t},
        "missed": missed,
    }
