"""sacctmgr: QOS, account and user/association management against slurmdbd."""

from __future__ import annotations

from .. import command
from .common import HEAD, SLURM_VERSION, _err, fmt_limit_upper, parse_time
from .fairshare import _acct_children, _acct_parent, _acct_users, _assoc_shares
from .scheduler import _ensure, schedule

QOS_KEYS = {
    "maxtresperuser": "max_tres_per_user",
    "maxtrespu": "max_tres_per_user",
    "maxtresperjob": "max_tres_per_job",
    "maxtres": "max_tres_per_job",
    "maxjobsperuser": "max_jobs_per_user",
    "maxjobspu": "max_jobs_per_user",
    "maxjobs": "max_jobs_per_user",
    "maxwall": "max_wall",
    "maxwalldurationperjob": "max_wall",
    "priority": "priority",
    "grptres": "grp_tres",
    "flags": "flags",
    "preempt": "preempt",
    "description": "description",
    "maxsubmitjobsperuser": "max_submit_pu",
    "maxsubmitpu": "max_submit_pu",
    "usagefactor": "usage_factor",
    "gracetime": "grace_time",
    "preemptmode": "preempt_mode",
}
QOS_COLS = {
    "name": ("Name", 10, "name"),
    "priority": ("Priority", 10, "priority"),
    "gracetime": ("GraceTime", 10, "grace_time"),
    "preempt": ("Preempt", 10, "preempt"),
    "preemptmode": ("PreemptMode", 11, "preempt_mode"),
    "flags": ("Flags", 20, "flags"),
    "usagefactor": ("UsageFactor", 11, "usage_factor"),
    "grptres": ("GrpTRES", 13, "grp_tres"),
    "maxtres": ("MaxTRES", 13, "max_tres_per_job"),
    "maxtresperjob": ("MaxTRES", 13, "max_tres_per_job"),
    "maxwall": ("MaxWall", 11, "max_wall"),
    "maxtrespu": ("MaxTRESPU", 13, "max_tres_per_user"),
    "maxtresperuser": ("MaxTRESPU", 13, "max_tres_per_user"),
    "maxjobspu": ("MaxJobsPU", 9, "max_jobs_per_user"),
    "maxjobsperuser": ("MaxJobsPU", 9, "max_jobs_per_user"),
    "maxsubmitpu": ("MaxSubmitPU", 11, "max_submit_pu"),
    "description": ("Descr", 11, "description"),
}
QOS_DEFAULT_COLS = [
    "name",
    "priority",
    "gracetime",
    "preempt",
    "preemptmode",
    "flags",
    "usagefactor",
    "grptres",
    "maxtres",
    "maxwall",
    "maxtrespu",
    "maxjobspu",
    "maxsubmitpu",
]


def _fmt_table(cols, rows, parsable=False, noheader=False, left=()):
    out = []
    if parsable:
        if not noheader:
            out.append("|".join(h for h, _ in cols) + "|")
        for r in rows:
            out.append("|".join(str(v) for v in r) + "|")
        return "\n".join(out) + "\n"
    if not noheader:
        out.append(" ".join(h[:w].rjust(w) for h, w in cols))
        out.append(" ".join("-" * w for _, w in cols))
    for r in rows:
        out.append(
            " ".join(
                (str(v)[:w].ljust(w) if i in left else str(v)[:w].rjust(w))
                for i, ((_, w), v) in enumerate(zip(cols, r))
            ).rstrip()
        )
    return "\n".join(out) + "\n"


def _merge_tres(old, new):
    cur = {}
    for tok in str(old or "").split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            cur[k] = v
    for tok in str(new).split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = "gres/gpu" if k == "gpu" else k
            if v in ("-1", ""):
                cur.pop(k, None)
            else:
                cur[k] = v
    order = ["cpu", "mem", "node", "billing"]
    keys = sorted(cur, key=lambda k: (order.index(k) if k in order else 10, k))
    return ",".join(f"{k}={cur[k]}" for k in keys)


def _set_qos_field(q, key, val):
    lk = key.lower()
    if lk not in QOS_KEYS:
        raise _err(f" Unknown option: {key}={val}\n Use keyword 'where' to modify condition")
    field = QOS_KEYS[lk]
    if field in ("max_tres_per_user", "max_tres_per_job", "grp_tres"):
        q[field] = _merge_tres(q.get(field), val)
    elif field in ("priority",):
        q[field] = int(val)
    elif field in ("max_jobs_per_user", "max_submit_pu"):
        q[field] = None if val in ("-1", "") else int(val)
    elif field == "max_wall":
        q[field] = None if val in ("-1", "") else val
    elif field == "flags":
        q[field] = "" if val in ("-1", "") else val
    else:
        q[field] = val


def _sacctmgr_split(args):
    """-> (entity, names, where{}, set{})"""
    entity = args[0].lower() if args else ""
    names, where, sets = [], {}, {}
    mode = "where"
    for a in args[1:]:
        la = a.lower()
        if la == "where":
            mode = "where"
            continue
        if la == "set":
            mode = "set"
            continue
        if "=" in a:
            k, v = a.split("=", 1)
            op = "="
            if k.endswith("+") or k.endswith("-"):
                op = k[-1] + "="
                k = k[:-1]
            (sets if mode == "set" else where)[k.lower()] = (v, op)
        else:
            names.extend(x for x in a.split(",") if x)
    if "name" in where:
        names += where.pop("name")[0].split(",")
    if "names" in where:
        names += where.pop("names")[0].split(",")
    return entity, names, where, sets


def _confirm(immediate):
    return (
        ""
        if immediate
        else "Would you like to commit changes? (You have 30 seconds to decide)\n(N/y): y  [simulator auto-confirmed; use -i to skip this prompt]\n"
    )


@command("sacctmgr", help="accounting manager: sacctmgr [-i] add|modify|show|delete qos|account|user|assoc ...")
def sacctmgr(ctx, args):
    state = ctx.state
    _ensure(state)
    immediate = False
    parsable = False
    noheader = False
    while args and args[0].startswith("-"):
        if args[0] in ("-i", "--immediate"):
            immediate = True
        elif args[0] in ("-p", "-P", "--parsable", "--parsable2"):
            parsable = True
        elif args[0] in ("-n", "--noheader"):
            noheader = True
        elif args[0] in ("-V", "--version"):
            return f"slurm {SLURM_VERSION}\n"
        args = args[1:]
    if not args:
        raise _err("sacctmgr: no command given")
    head = state.nodes.get(HEAD)
    if head is not None and head.services.get("slurmdbd", "active") != "active":
        raise _err(
            "sacctmgr: error: Sending PersistInit msg: Connection refused\n"
            "sacctmgr: error: Problem talking to the database: Connection refused"
        )
    verb = args[0].lower()
    s = state.slurm
    if verb in ("add", "create"):
        entity, names, where, sets = _sacctmgr_split(args[1:])
        allkv = {**where, **sets}
        if entity == "qos":
            if not names:
                raise _err(" Need name of qos to add.")
            out = [" Adding QOS(s)"]
            for n in names:
                if n in s.qos:
                    return f" Nothing new added.\nsacctmgr: error: QOS {n} already exists\n"
                q = {
                    "name": n,
                    "priority": 0,
                    "max_tres_per_user": "",
                    "preempt": "",
                    "max_tres_per_job": "",
                    "max_jobs_per_user": None,
                    "max_wall": None,
                    "grp_tres": "",
                    "flags": "",
                    "description": n,
                }
                for k, (v, _) in allkv.items():
                    _set_qos_field(q, k, v)
                s.qos[n] = q
                out.append(f"  {n}")
            out.append(" Settings")
            out.append(f"  Description    = {names[0]}")
            for k, (v, _) in allkv.items():
                out.append(f"  {k:<15}= {v}")
            return "\n".join(out) + "\n" + _confirm(immediate)
        if entity == "account":
            if not names:
                raise _err(" Need name of account to add.")
            parent = allkv.get("parent", ("root", "="))[0] or "root"
            if parent not in s.accounts:
                raise _err(f" Parent account {parent} doesn't exist.\nsacctmgr: error: Problem adding accounts")
            new = [n for n in names if n not in s.accounts]
            if not new:
                return " Nothing new added.\n"
            try:
                shares = int(allkv.get("fairshare", ("1", "="))[0])
            except ValueError:
                raise _err(" Bad FairShare value. Use a number, or 'parent'.") from None
            out = [" Adding Account(s)"]
            for n in new:
                s.accounts[n] = {
                    "name": n,
                    "fairshare": shares,
                    "users": [],
                    "parent": parent,
                    "description": allkv.get("description", (n, "="))[0],
                    "organization": allkv.get("organization", (n, "="))[0],
                }
                out.append(f"  {n}")
            out += [
                " Settings",
                f"  Description     = {s.accounts[new[0]].description}",
                f"  Organization    = {s.accounts[new[0]].organization}",
                " Associations",
            ]
            out += [f"  A = {n:<10} C = superpod  " for n in new]
            out += [" Settings", f"  Parent        = {parent}"]
            if "fairshare" in allkv:
                out.append(f"  Fairshare     = {shares}")
            return "\n".join(out) + "\n" + _confirm(immediate)
        if entity == "user":
            if not names:
                raise _err(" Need name of user to add.")
            accts = [a for a in allkv.get("account", allkv.get("accounts", ("", "=")))[0].split(",") if a]
            dflt = allkv.get("defaultaccount", ("", "="))[0]
            if not accts and dflt:
                accts = [dflt]
            if not accts:
                raise _err(" Need name of account to add user to.")
            for acct in accts + ([dflt] if dflt else []):
                if acct not in s.accounts:
                    raise _err(
                        f" There is no uncommitted association for account {acct}\n"
                        f"sacctmgr: error: Account {acct} does not exist"
                    )
            out = [" Adding User(s)"]
            assocs = []
            for n in names:
                u = s.users.get(n)
                if u is None:
                    s.users[n] = {
                        "name": n,
                        "default_account": dflt or accts[0],
                        "accounts": [],
                        "qos": ["normal"],
                        "default_qos": "",
                        "fairshare_factor": 0.5,
                        "shares": {},
                    }
                    u = s.users[n]
                elif dflt:
                    u.default_account = dflt
                u.setdefault("shares", {})
                for acct in accts:
                    if acct not in u.accounts:
                        u.accounts.append(acct)
                        assocs.append((n, acct))
                    if n not in s.accounts[acct].users:
                        s.accounts[acct].users.append(n)
                    if "fairshare" in allkv:
                        u.shares[acct] = int(allkv["fairshare"][0])
                if "qos" in allkv:
                    u.qos = allkv["qos"][0].split(",")
                if "defaultqos" in allkv:
                    u.default_qos = allkv["defaultqos"][0]
                out.append(f"  {n}")
            if not assocs:
                return " Nothing new added.\n"
            out += [" Settings", f"  Default Account = {s.users[names[0]].default_account}", " Associations"]
            out += [f"  U = {n:<9} A = {a:<10} C = superpod  " for n, a in assocs]
            return "\n".join(out) + "\n" + _confirm(immediate)
        raise _err(f" Unknown entity: {entity}")
    if verb in ("modify", "update"):
        entity, names, where, sets = _sacctmgr_split(args[1:])
        if not sets:
            raise _err(" You didn't give me anything to set")
        if entity == "qos":
            targets = [n for n in names if n in s.qos]
            if not targets:
                return " Nothing modified\n"
            for n in targets:
                for k, (v, _) in sets.items():
                    _set_qos_field(s.qos[n], k, v)
            out = [" Modified qos..."] + [f"  {n}" for n in targets]
            schedule(state)
            return "\n".join(out) + "\n" + _confirm(immediate)
        if entity == "user":
            targets = [n for n in names if n in s.users]
            if not targets:
                return " Nothing modified\n"
            for n in targets:
                u = s.users[n]
                for k, (v, op) in sets.items():
                    if k == "qos":
                        vals = v.split(",")
                        if op == "+=":
                            u.qos = list(dict.fromkeys(list(u.qos) + vals))
                        elif op == "-=":
                            u.qos = [x for x in u.qos if x not in vals]
                        else:
                            u.qos = vals
                    elif k == "defaultqos":
                        u.default_qos = v
                    elif k == "defaultaccount":
                        if v not in u.get("accounts", []):
                            raise _err(
                                f" Can't modify because these users aren't associated with new default account '{v}'...\n"
                                f"  U = {n}\n"
                                f"sacctmgr: error: Add the association first: sacctmgr add user {n} account={v}"
                            )
                        u.default_account = v
                    elif k == "fairshare":
                        u.setdefault("shares", {})
                        for acct in where.get("account", (",".join(u.get("accounts", [])), "="))[0].split(","):
                            if acct in u.get("accounts", []):
                                u.shares[acct] = int(v)
                    else:
                        raise _err(f" Unknown option: {k}={v}")
            out = [" Modified user associations..."] + [
                f"  C = superpod    A = {s.users[n].default_account:<10} U = {n}" for n in targets
            ]
            schedule(state)
            return "\n".join(out) + "\n" + _confirm(immediate)
        if entity == "account":
            targets = [n for n in names if n in s.accounts]
            for n in targets:
                for k, (v, _) in sets.items():
                    if k == "fairshare":
                        s.accounts[n].fairshare = int(v)
                    elif k == "parent":
                        if v not in s.accounts or v == n:
                            raise _err(f" Parent account {v} doesn't exist.")
                        s.accounts[n].parent = v
                    elif k in ("description", "organization"):
                        s.accounts[n][k] = v
                    else:
                        raise _err(f" Unknown option: {k}={v}")
            if targets:
                schedule(state)
            return (
                (" Modified account associations...\n" + "".join(f"  {n}\n" for n in targets) + _confirm(immediate))
                if targets
                else " Nothing modified\n"
            )
        raise _err(f" Unknown entity: {entity}")
    if verb in ("delete", "remove"):
        entity, names, where, _ = _sacctmgr_split(args[1:])
        table = {"qos": s.qos, "account": s.accounts, "user": s.users}.get(entity)
        if table is None:
            raise _err(f" Unknown entity: {entity}")
        gone = [n for n in names if n in table]
        if entity == "qos" and any(j.qos in gone and j.state in ("PENDING", "RUNNING") for j in s.jobs):
            raise _err(" Error with request: Job(s) active, cancel job(s) before remove\n  JobId = ...")
        if entity == "user" and "account" in where:
            accts = where["account"][0].split(",")
            done = []
            for n in gone:
                u = s.users[n]
                for a in accts:
                    if a in u.get("accounts", []):
                        u.accounts.remove(a)
                        done.append(f"  C = superpod    A = {a:<10} U = {n}")
                    if a in s.accounts and n in s.accounts[a].users:
                        s.accounts[a].users.remove(n)
                if not u.accounts:
                    del s.users[n]
                elif u.default_account not in u.accounts:
                    u.default_account = u.accounts[0]
            if not done:
                return " Nothing deleted\n"
            return " Deleting user associations...\n" + "\n".join(done) + "\n" + _confirm(immediate)
        if entity == "account":
            for n in gone:
                if _acct_children(s, n):
                    raise _err(
                        f" Error with request: account {n} has child accounts; remove or move them first\n"
                        "sacctmgr: error: Problem deleting accounts"
                    )
            for n in gone:
                for u in s.users.values():
                    if n in u.get("accounts", []):
                        u.accounts.remove(n)
                        if u.default_account == n and u.accounts:
                            u.default_account = u.accounts[0]
        if entity == "user":
            for n in gone:
                for a in s.accounts.values():
                    if n in a.users:
                        a.users.remove(n)
        for n in gone:
            del table[n]
        return (
            (f" Deleting {entity}(s)...\n" + "".join(f"  {n}\n" for n in gone) + _confirm(immediate))
            if gone
            else " Nothing deleted\n"
        )
    if verb in ("show", "list"):
        entity, names, where, _ = _sacctmgr_split(args[1:])
        fmt = where.pop("format", (None, ""))[0]
        if entity == "qos":
            keys = [k.split("%")[0].lower() for k in fmt.split(",")] if fmt else QOS_DEFAULT_COLS
            for k in keys:
                if k not in QOS_COLS:
                    raise _err(f'sacctmgr: error: Invalid field requested: "{k}"')
            cols = [(QOS_COLS[k][0], QOS_COLS[k][1]) for k in keys]
            rows = []
            for q in s.qos.values():
                if names and q.name not in names:
                    continue
                row = []
                for k in keys:
                    v = q.get(QOS_COLS[k][2], "")
                    if v is None:
                        v = ""
                    if k in ("gracetime",) and not v:
                        v = "00:00:00"
                    if k == "preemptmode" and not v:
                        v = "cluster"
                    if k == "usagefactor" and not v:
                        v = "1.000000"
                    if k == "maxwall" and v:
                        v = fmt_limit_upper(parse_time(v))
                    row.append(v)
                rows.append(row)
            return _fmt_table(cols, rows, parsable, noheader)
        if entity in ("assoc", "association", "associations"):
            tree = "tree" in [n.lower() for n in names]
            names = [n for n in names if n.lower() != "tree"]
            return _show_assoc(state, fmt, tree, where, names, parsable, noheader)
        if entity in ("user", "users"):
            withassoc = "withassoc" in [n.lower() for n in names]
            names = [n for n in names if n.lower() not in ("withassoc", "withcoord")]
            keys = [k.split("%")[0].lower() for k in fmt.split(",")] if fmt else ["user", "defaultaccount", "admin"]
            if withassoc and not fmt:
                keys += ["cluster", "account", "partition", "share", "qos", "defaultqos"]
            hdr = {
                "user": ("User", 10),
                "defaultaccount": ("Def Acct", 10),
                "admin": ("Admin", 9),
                "adminlevel": ("Admin", 9),
                "cluster": ("Cluster", 10),
                "account": ("Account", 10),
                "partition": ("Partition", 10),
                "share": ("Share", 9),
                "fairshare": ("Share", 9),
                "qos": ("QOS", 20),
                "defaultqos": ("Def QOS", 9),
            }
            for k in keys:
                if k not in hdr:
                    raise _err(f'sacctmgr: error: Invalid field requested: "{k}"')
            rows = []
            for u in s.users.values():
                if names and u.name not in names:
                    continue
                per_assoc = withassoc or "account" in keys
                accts = list(u.get("accounts") or [u.default_account]) if per_assoc else [None]
                for a in accts:
                    vals = {
                        "user": u.name,
                        "defaultaccount": u.default_account,
                        "admin": "Administ+" if u.name == "root" else "None",
                        "cluster": "superpod",
                        "account": a or "",
                        "partition": "",
                        "share": _assoc_shares(s, a, u.name) if a else "",
                        "qos": ",".join(u.get("qos", [])),
                        "defaultqos": u.get("default_qos", ""),
                    }
                    vals["adminlevel"] = vals["admin"]
                    vals["fairshare"] = vals["share"]
                    rows.append([vals[k] for k in keys])
            return _fmt_table([hdr[k] for k in keys], rows, parsable, noheader)
        if entity in ("account", "accounts"):
            names = [n for n in names if n.lower() not in ("withassoc", "withcoord")]
            keys = [k.split("%")[0].lower() for k in fmt.split(",")] if fmt else ["account", "descr", "org"]
            hdr = {
                "account": ("Account", 10),
                "descr": ("Descr", 20),
                "description": ("Descr", 20),
                "org": ("Org", 20),
                "organization": ("Org", 20),
                "parentname": ("Par Name", 10),
                "fairshare": ("Share", 9),
                "share": ("Share", 9),
            }
            for k in keys:
                if k not in hdr:
                    raise _err(f'sacctmgr: error: Invalid field requested: "{k}"')
            rows = []
            for a in s.accounts.values():
                if names and a.name not in names:
                    continue
                vals = {
                    "account": a.name,
                    "descr": a.description,
                    "org": a.organization,
                    "parentname": "" if a.name == "root" else a.get("parent") or "root",
                    "fairshare": a.fairshare,
                }
                vals["description"], vals["organization"], vals["share"] = vals["descr"], vals["org"], vals["fairshare"]
                rows.append([vals[k] for k in keys])
            return _fmt_table([hdr[k] for k in keys], rows, parsable, noheader)
        if entity == "cluster":
            return _fmt_table(
                [("Cluster", 10), ("ControlHost", 15), ("ControlPort", 12), ("RPC", 5)],
                [["superpod", "10.141.255.254", "6817", "10240"]],
                parsable,
                noheader,
            )
        raise _err(f" Unknown entity: {entity}")
    raise _err(f"sacctmgr: error: Invalid command: {args[0]}")


ASSOC_COLS = {
    "cluster": ("Cluster", 10),
    "account": ("Account", 20),
    "user": ("User", 10),
    "partition": ("Partition", 10),
    "share": ("Share", 9),
    "fairshare": ("Share", 9),
    "priority": ("Priority", 10),
    "grpjobs": ("GrpJobs", 7),
    "grptres": ("GrpTRES", 13),
    "maxjobs": ("MaxJobs", 7),
    "maxtres": ("MaxTRES", 13),
    "qos": ("QOS", 20),
    "defaultqos": ("Def QOS", 9),
    "parentname": ("Par Name", 10),
    "lineage": ("Lineage", 30),
}
ASSOC_DEFAULT = [
    "cluster",
    "account",
    "user",
    "partition",
    "share",
    "priority",
    "grpjobs",
    "grptres",
    "maxjobs",
    "maxtres",
    "qos",
    "defaultqos",
]


def _assoc_walk(s):
    """Associations in hierarchy order: [(depth, account, user|None)]."""
    out, seen = [], set()

    def walk(acct, depth):
        if acct in seen or acct not in s.accounts:
            return
        seen.add(acct)
        out.append((depth, acct, None))
        for un in _acct_users(s, acct):
            out.append((depth + 1, acct, un))
        for child in sorted(_acct_children(s, acct)):
            walk(child, depth + 1)

    walk("root", 0)
    for a in s.accounts:  # orphans (parent removed)
        if a not in seen:
            walk(a, 1)
    return out


def _lineage(s, acct, user=None):
    parts, a, guard = [], acct, 0
    while a and guard < 20:
        parts.insert(0, a)
        a = _acct_parent(s, a)
        guard += 1
    return "/" + "/".join(parts) + "/" + (f"0-{user}" if user else "")


def _show_assoc(state, fmt, tree, where, names, parsable, noheader):
    s = state.slurm
    keys = [k.split("%")[0].lower() for k in fmt.split(",")] if fmt else ASSOC_DEFAULT
    for k in keys:
        if k not in ASSOC_COLS:
            raise _err(f'sacctmgr: error: Invalid field requested: "{k}"')
    want_users = [u for u in where.get("user", ("", "="))[0].split(",") if u] + list(names)
    want_accts = [a for a in where.get("account", ("", "="))[0].split(",") if a]
    rows = []
    for depth, acct, user in _assoc_walk(s):
        if want_users and user not in want_users:
            continue
        if want_accts and acct not in want_accts:
            continue
        a = s.accounts[acct]
        u = s.users.get(user, {}) if user else {}
        if user:
            share = _assoc_shares(s, acct, user)
            qos = ",".join(u.get("qos", ["normal"]))
            dq = u.get("default_qos", "")
        else:
            share = a.get("fairshare", 1)
            qos, dq = "normal", ""
        vals = {
            "cluster": "superpod",
            "account": (" " * depth if tree else "") + acct,
            "user": user or "",
            "partition": "",
            "share": share,
            "fairshare": share,
            "priority": "",
            "grpjobs": "",
            "grptres": a.get("grp_tres", "") if not user else "",
            "maxjobs": "",
            "maxtres": "",
            "qos": qos,
            "defaultqos": dq,
            "parentname": "" if user or acct == "root" else (a.get("parent") or "root"),
            "lineage": _lineage(s, acct, user),
        }
        rows.append([vals[k] for k in keys])
    left = {i for i, k in enumerate(keys) if k == "account"} if tree else set()
    return _fmt_table([ASSOC_COLS[k] for k in keys], rows, parsable, noheader, left)
