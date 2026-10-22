"""Scenario loading, fault injection and grading.

Public scenario files live in ``greenray/data/scenarios/<id>.yaml``. Their reference
solution, explanation and edge cases live in the private solutions repository
(see :mod:`greenray.content.solutions`). Format: ``docs/scenario-format.md``.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import yaml

from ..core.shell import INCONSISTENT_FLAG
from ..core.state import AttrDict, append_path, get_path, make_cluster, set_path, wrap
from . import solutions
from .domains import DOMAINS

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "data" / "scenarios"
DIFFICULTIES = ("easy", "medium", "hard")
PRIVATE_KEYS = ("solution", "explanation", "edge_cases")
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def fill(value, params: dict):
    """Substitute ``{name}`` for every name in ``params``, recursively; other braces are left alone."""
    if isinstance(value, str):
        return PLACEHOLDER.sub(lambda m: str(params[m.group(1)]) if m.group(1) in params else m.group(0), value)
    if isinstance(value, list):
        return [fill(v, params) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, params) for k, v in value.items()}
    return value


@dataclass
class Scenario:
    id: str
    title: str
    domain: str
    briefing: str
    checks: list
    difficulty: str = "medium"
    time_limit_min: int = 15
    objectives: list = field(default_factory=list)
    setup: list = field(default_factory=list)
    hints: list = field(default_factory=list)
    references: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    variants: list = field(default_factory=list)
    path: str = ""
    variant_index: int | None = None

    @classmethod
    def from_file(cls, path: Path) -> Scenario:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        leaked = [k for k in PRIVATE_KEYS if k in data]
        if leaked:
            raise ValueError(f"{path.name}: {leaked} belong in the private solutions repo, not the public file")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path.name}: unknown keys {sorted(unknown)}")
        if data.get("domain") not in DOMAINS:
            raise ValueError(f"{path.name}: domain must be one of {DOMAINS}")
        if data.get("difficulty", "medium") not in DIFFICULTIES:
            raise ValueError(f"{path.name}: difficulty must be one of {DIFFICULTIES}")
        if path.stem != data.get("id"):
            raise ValueError(f"{path.name}: file name must match id {data.get('id')!r}")
        variants = data.get("variants") or []
        if variants and (not all(isinstance(v, dict) for v in variants) or len({frozenset(v) for v in variants}) != 1):
            raise ValueError(f"{path.name}: variants must be dicts with the same keys")
        return cls(**data, path=str(path))

    # ------------------------------------------------------------ variants

    @property
    def params(self) -> dict:
        return self.variants[self.variant_index] if self.variant_index is not None else {}

    def variant(self, index: int) -> Scenario:
        """A concrete scenario with the placeholders of variant ``index`` filled in."""
        if not 0 <= index < len(self.variants):
            raise IndexError(f"{self.id} has {len(self.variants)} variants")
        params = self.variants[index]
        filled = {
            k: fill(copy.deepcopy(getattr(self, k)), params)
            for k in ("title", "briefing", "objectives", "setup", "checks", "hints")
        }
        return replace(self, variant_index=index, **filled)

    def all_variants(self) -> list[Scenario]:
        """Every concrete variant of a raw scenario; a concrete one is just itself."""
        if self.variant_index is not None or not self.variants:
            return [self]
        return [self.variant(i) for i in range(len(self.variants))]

    @property
    def variant_label(self) -> str:
        return f"variant {self.variant_index + 1} of {len(self.variants)}" if self.variant_index is not None else ""

    # ------------------------------------------------------------ private content

    @property
    def private(self) -> dict:
        return fill(solutions.scenario_solution(self.id), self.params)

    @property
    def has_solution(self) -> bool:
        return bool(self.private.get("solution"))

    @property
    def solution(self) -> list[str]:
        return list(self.private.get("solution", []))

    @property
    def explanation(self) -> str:
        return self.private.get("explanation", "")

    @property
    def edge_cases(self) -> list[str]:
        return list(self.private.get("edge_cases", []))

    # ------------------------------------------------------------ state

    def build_state(self) -> AttrDict:
        if self.variants and self.variant_index is None:
            raise ValueError(f"{self.id} has variants: pick one with variant(i) or pick_variant() first")
        state = make_cluster()
        for op in self.setup:
            apply_op(state, op)
        return state

    # ------------------------------------------------------------ grading

    def grade(self, shell) -> list[tuple[str, bool, str]]:
        """Return [(description, passed, error)] for every check."""
        env = check_env(shell)
        results = []
        broken = shell.ctx.state.get("flags", {}).get(INCONSISTENT_FLAG)
        if broken:  # a crashed command could not be rolled back: never grade a possibly half-applied state
            results.append(("cluster state is consistent (type `reset`)", False, f"internal error in {broken}"))
        for chk in self.checks:
            desc = chk.get("desc", chk["expr"])
            try:
                ok = bool(eval(chk["expr"], {"__builtins__": SAFE_BUILTINS, **env}))
                err = ""
            except Exception as e:  # a check that cannot evaluate counts as failed
                ok, err = False, f"{type(e).__name__}: {e}"
            results.append((desc, ok, err))
        return results


SAFE_BUILTINS = {
    "len": len,
    "any": any,
    "all": all,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
    "set": set,
    "list": list,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "isinstance": isinstance,
    "dict": dict,
    "True": True,
    "False": False,
    "None": None,
}


def check_env(shell) -> dict:
    state = shell.ctx.state
    executed = shell.ctx.executed

    def ran(pattern: str, host: str | None = None) -> bool:
        rx = re.compile(pattern)
        return any(rx.search(" ".join(argv)) for h, argv in executed if host is None or h == host)

    def file(host: str, path: str) -> str:
        return state.nodes[host].files.get(path, "")

    def pods(namespace=None, prefix=""):
        return [
            p for p in state.k8s.pods if (namespace is None or p.namespace == namespace) and p.name.startswith(prefix)
        ]

    def release(name):
        return next((r for r in state.k8s.helm_releases if r.name == name), None)

    def jobs(**kw):
        return [j for j in state.slurm.jobs if all(j.get(k) == v for k, v in kw.items())]

    return {
        "state": state,
        "nodes": state.nodes,
        "slurm": state.slurm,
        "k8s": state.k8s,
        "bcm": state.bcm,
        "ngc": state.ngc,
        "dcgm": state.dcgm,
        "flags": state.flags,
        "ran": ran,
        "file": file,
        "pods": pods,
        "release": release,
        "jobs": jobs,
        "re": re,
    }


def apply_op(state, op: dict):
    """Apply one setup operation to a fresh cluster.

    Supported ops:
      {set: path, value: v}           replace a value (dotted path, list indexes allowed)
      {append: path, value: v}        append to a list
      {merge: path, value: {...}}     update a dict
      {delete: path}                  remove a dict key
      {file: path, host: h, content}  write a file on a host
      {python: "code"}                exec code with `state`, `wrap` and helpers in scope
    """
    if "set" in op:
        set_path(state, op["set"], op["value"])
    elif "append" in op:
        append_path(state, op["append"], op["value"])
    elif "merge" in op:
        get_path(state, op["merge"]).update(wrap(op["value"]))
    elif "delete" in op:
        parent, _, key = op["delete"].rpartition(".")
        get_path(state, parent).pop(key, None)
    elif "file" in op:
        state.nodes[op.get("host", "bcm-head")].files[op["file"]] = op["content"]
    elif "python" in op:
        exec(op["python"], {"state": state, "wrap": wrap, "get_path": get_path, "set_path": set_path})
    else:
        raise ValueError(f"unknown setup op: {op}")


@lru_cache(maxsize=1)
def _load_all_cached() -> tuple[Scenario, ...]:
    return tuple(Scenario.from_file(p) for p in sorted(SCENARIO_DIR.glob("*.yaml")))


def load_all() -> list[Scenario]:
    out = list(_load_all_cached())
    ids = [s.id for s in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate scenario ids: {dupes}")
    return out


def find(scenario_id: str) -> Scenario:
    for s in load_all():
        if s.id == scenario_id:
            return s
    raise KeyError(scenario_id)


def pick_variant(s: Scenario, seed: int | None = None, avoid: int | None = None) -> Scenario:
    """A random variant (not ``avoid`` when there is a choice); the scenario itself when it has none."""
    if not s.variants:
        return s
    choices = [i for i in range(len(s.variants)) if i != avoid] or list(range(len(s.variants)))
    return s.variant(random.Random(seed).choice(choices))


def similar(scenario: Scenario, exclude: frozenset[str] | set[str] = frozenset()) -> Scenario | None:
    """The scenario sharing the most tags with ``scenario`` (ties: same domain first, then id)."""
    candidates = [
        (len(set(o.tags) & set(scenario.tags)), o.domain == scenario.domain, o)
        for o in load_all()
        if o.id != scenario.id and o.id not in exclude
    ]
    candidates = [c for c in candidates if c[0]]
    if not candidates:
        return None
    return min(candidates, key=lambda c: (-c[0], not c[1], c[2].id))[2]


def replay_solution(scenario: Scenario, shell) -> list[str]:
    """Run the reference solution. Multi-line steps feed lines 2+ as heredoc input."""
    if not scenario.has_solution:
        raise LookupError(f"no reference solution for {scenario.id}: {solutions.missing_message()}")
    outputs = []
    for step in scenario.solution:
        lines = step.split("\n")
        feed = iter(lines[1:])
        outputs.append(shell.run(lines[0], heredoc_reader=lambda _p, feed=feed: next(feed, None)))
    return outputs
