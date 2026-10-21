"""Fair-share tree arithmetic over ``slurm.accounts`` / ``slurm.users`` (classic 2^(-usage/shares))."""

from __future__ import annotations


def _acct_parent(s, name):
    a = s.accounts.get(name)
    return (a.get("parent") or "root") if a is not None and name != "root" else None


def _acct_children(s, name):
    return [a.name for a in s.accounts.values() if a.name != "root" and (a.get("parent") or "root") == name]


def _assoc_shares(s, acct, user) -> int:
    u = s.users.get(user)
    if u is None:
        return 1
    return int((u.get("shares") or {}).get(acct, 1))


def _acct_users(s, acct):
    return [un for un in s.accounts.get(acct, {}).get("users", []) if un in s.users]


def _job_usage(state, j) -> int:
    if j.get("start_time") is None:
        return 0
    end = j.end_time if j.get("end_time") is not None else state.slurm.get("clock", 0)
    ncpu = int(j.num_nodes) * int(j.ntasks_per_node) * int(j.cpus_per_task)
    return max(0, end - j.start_time) * ncpu


def _acct_usage(state, acct) -> int:
    s = state.slurm
    names = {acct}
    stack = [acct]
    while stack:
        for c in _acct_children(s, stack.pop()):
            if c not in names:
                names.add(c)
                stack.append(c)
    return sum(_job_usage(state, j) for j in s.jobs if j.get("account") in names) + sum(
        int(s.accounts[n].get("raw_usage", 0) or 0) for n in names if n in s.accounts
    )


def _user_usage(state, acct, user) -> int:
    s = state.slurm
    extra = int((s.users.get(user, {}).get("raw_usage") or {}).get(acct, 0)) if s.users.get(user) else 0
    return extra + sum(_job_usage(state, j) for j in s.jobs if j.get("account") == acct and j.get("user") == user)


def _level_total(s, parent) -> int:
    tot = sum(int(s.accounts[c].get("fairshare", 1)) for c in _acct_children(s, parent))
    tot += sum(_assoc_shares(s, parent, u) for u in _acct_users(s, parent))
    return max(1, tot)


def _norm_acct(s, name) -> float:
    if name == "root":
        return 1.0
    parent = _acct_parent(s, name)
    return int(s.accounts[name].get("fairshare", 1)) / _level_total(s, parent)


def _global_share(s, acct, user=None) -> float:
    share = 1.0
    if user is not None:
        share = _assoc_shares(s, acct, user) / _level_total(s, acct)
    a, seen = acct, set()
    while a and a != "root" and a not in seen:
        seen.add(a)
        share *= _norm_acct(s, a)
        a = _acct_parent(s, a)
    return share


def _fs_factor(state, user, acct) -> float:
    """Fair-share factor of a user's association (classic 2^(-usage/shares) formula)."""
    s = state.slurm
    u = s.users.get(user, {})
    if not acct or acct == "root" or acct not in s.accounts:
        return float(u.get("fairshare_factor", 0.5))
    total = max(1, _acct_usage(state, "root"))
    eff = _user_usage(state, acct, user) / total
    share = _global_share(s, acct, user)
    return round(2 ** (-eff / share), 6) if share > 0 else 0.0
