"""What to practise next, and a four-week schedule that covers every scenario."""

from __future__ import annotations

from dataclasses import dataclass

from ..content import scenarios as sc
from ..content.domains import DOMAINS
from ..content.scenarios import Scenario
from . import progress
from .ui import UI

ON_RAMP = "on-ramp"
# Teaching order for the on-ramp: read before you change, Slurm before Kubernetes.
ON_RAMP_ORDER = (
    "tsh-first-look-at-logs",
    "tsh-which-gpu-is-which",
    "adm-bcm-cmsh-inventory",
    "wkl-slurm-first-gpu-job",
    "adm-slurm-partition-time-limit",
    "wkl-k8s-pod-pending-nodeselector",
)
LEVEL = {"easy": 0, "medium": 1, "hard": 2}
DAYS_PER_WEEK = 5


def weakest_domain(data: dict, last: int = 5) -> tuple[str, float] | None:
    """Domain with the lowest question score over the last exams, or None without exams."""
    totals: dict[str, list[int]] = {}
    for e in data["exams"][-last:]:
        for d, v in e["per_domain"].items():
            c, t = map(int, v.split("/"))
            totals.setdefault(d, [0, 0])
            totals[d][0] += c
            totals[d][1] += t
    if not totals:
        return None
    d, (c, t) = min(totals.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1]))
    return d, 100 * c / max(1, t)


def _on_ramp_position(s: Scenario) -> int:
    return ON_RAMP_ORDER.index(s.id) if s.id in ON_RAMP_ORDER else len(ON_RAMP_ORDER)


@dataclass(frozen=True)
class Pick:
    scenario: Scenario
    command: str
    why: str


def recommend(data: dict | None = None, limit: int = 3) -> list[Pick]:
    """Unpassed labs first (on-ramp, then easiest, weak domain ahead), then guided passes to retry unguided."""
    data = progress.load() if data is None else data
    recs = data["scenarios"]
    weak = weakest_domain(data)
    weak_name = weak[0] if weak else None

    def rank(s: Scenario) -> tuple:
        return (
            ON_RAMP not in s.tags,
            _on_ramp_position(s),
            LEVEL.get(s.difficulty, 1),
            s.domain != weak_name,
            DOMAINS.index(s.domain),
            s.id,
        )

    def reason(s: Scenario, retry: bool) -> str:
        if retry:
            return "passed with guidance; prove it as an unguided incident"
        if ON_RAMP in s.tags:
            return "on-ramp: the basics every other lab assumes"
        if weak and s.domain == weak_name:
            return f"your weakest domain in recent exams ({weak[1]:.0f}%)"
        return f"{s.difficulty}, not passed yet"

    todo = sorted((s for s in sc.load_all() if not recs.get(s.id, {}).get("passed")), key=rank)
    picks = [Pick(s, f"greenray lab {s.code}", reason(s, False)) for s in todo]
    retry = sorted(
        (s for s in sc.load_all() if recs.get(s.id, {}).get("passed") and not recs[s.id].get("independent_passes")),
        key=rank,
    )
    picks += [Pick(s, f"greenray incident --id {s.code}", reason(s, True)) for s in retry]
    return picks[:limit]


def show_next(ui: UI) -> None:
    picks = recommend()
    if not picks:
        ui.out(ui.green("Every scenario is passed, unguided. Keep sharp with: greenray mock"))
        return
    top = picks[0]
    s = top.scenario
    body = [
        f"{ui.bold(s.title)}",
        f"{s.code} · {s.id}",
        f"{s.domain} · {s.difficulty} · ~{s.time_limit_min} min",
        f"Why: {top.why}",
        "",
        f"  {top.command}",
    ]
    ui.out("")
    ui.out(ui.panel("\n".join(body), title="Practise next"))
    if len(picks) > 1:
        ui.out(ui.dim("Then: " + "   ".join(p.command.removeprefix("greenray ") for p in picks[1:])))


# ---------------------------------------------------------------- four-week plan

WEEKS = (
    ("Basics", "on-ramp and easy labs, guided"),
    ("Install and administer", "Installation and Deployment, Administration"),
    ("Run and fix", "Workload Management, Troubleshooting and Optimization"),
    ("Exam week", "everything again as incidents, then mock exams"),
)


def _spread(items: list[str], days: int) -> list[list[str]]:
    """Deal items over days, earlier days getting the extra ones."""
    per, extra = divmod(len(items), days)
    out, i = [], 0
    for d in range(days):
        n = per + (1 if d < extra else 0)
        out.append(items[i : i + n])
        i += n
    return out


def plan() -> list[tuple[str, str, list[list[str]]]]:
    """[(week title, focus, [day activities])] covering every scenario once, plus drills and mocks."""
    scenarios = sc.load_all()

    def lab(s: Scenario) -> str:
        return f"lab {s.code}"

    def by_level(items: list[Scenario]) -> list[Scenario]:
        return sorted(items, key=lambda s: (LEVEL.get(s.difficulty, 1), s.id))

    basics = sorted((s for s in scenarios if ON_RAMP in s.tags), key=_on_ramp_position) + by_level(
        [s for s in scenarios if ON_RAMP not in s.tags and s.difficulty == "easy"]
    )
    rest = [s for s in scenarios if s not in basics]
    week2 = by_level([s for s in rest if s.domain in DOMAINS[:2]])
    week3 = by_level([s for s in rest if s.domain in DOMAINS[2:]])

    days1 = _spread([lab(s) for s in basics], DAYS_PER_WEEK)
    days1[-1].append('exam -n 20 -m 40 -d "Installation and Deployment"')
    days2 = _spread([lab(s) for s in week2], DAYS_PER_WEEK)
    days2[2].append('exam -n 20 -m 40 -d "Administration"')
    days2[-1].append('incident -d "Installation and Deployment"')
    days3 = _spread([lab(s) for s in week3], DAYS_PER_WEEK)
    days3[2].append('exam -n 20 -m 40 -d "Workload Management"')
    days3[-1].append('exam -n 20 -m 40 -d "Troubleshooting and Optimization"')
    days4 = [
        ["mock", "progress (note the weakest domain)"],
        ["incident -d <weakest domain>  (three times)", "exam -d <weakest domain>"],
        ["incident -t slurm", "incident -t kubernetes", "incident -t bcm"],
        ["mock --labs-only", "incident (any domain, twice)"],
        ["mock", "progress"],
    ]
    return [(WEEKS[i][0], WEEKS[i][1], d) for i, d in enumerate((days1, days2, days3, days4))]


def show_plan(ui: UI) -> None:
    recs = progress.load()["scenarios"]
    ui.out("")
    ui.out(ui.bold("Four-week plan: five study days a week, 30 to 60 minutes a day. Commands are `greenray ...`."))
    for w, (title, focus, days) in enumerate(plan(), 1):
        rows, styles = [], []
        for d, acts in enumerate(days, 1):
            for i, act in enumerate(acts):
                done = act.startswith("lab ") and recs.get(sc.find(act[4:]).id, {}).get("passed")
                rows.append([f"Day {d}" if i == 0 else "", "✔" if done else "", act])
                styles.append("green" if done else "")
        ui.out("")
        ui.out(ui.bold(f"Week {w}: {title}") + ui.dim(f"  ({focus})"))
        if ui.color:
            ui.out(ui.table(["", "", "greenray ..."], rows, styles))
        else:
            for day, mark, act in rows:
                ui.out(f"  {day:<6} {mark:1} {act}")
    ui.out(ui.dim("\nAdjust the pace to your week; `greenray next` always knows what is left."))
