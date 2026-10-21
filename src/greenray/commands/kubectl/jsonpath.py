"""Minimal JSONPath subset for ``-o jsonpath=`` and ``-o custom-columns=``."""

from __future__ import annotations

import json
import re

from .model import table


def _jp_steps(path):
    path = path.strip()
    if path.startswith("$"):
        path = path[1:]
    steps, i = [], 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            i += 1
            key = ""
            while i < len(path) and path[i] not in ".[":
                if path[i] == "\\" and i + 1 < len(path):
                    key += path[i + 1]
                    i += 2
                    continue
                key += path[i]
                i += 1
            if key == "*":
                steps.append(("all",))
            elif key:
                steps.append(("key", key))
        elif ch == "[":
            j = path.index("]", i) if "?(" not in path[i : i + 3] else path.index(")]", i) + 1
            inner = path[i + 1 : j].strip()
            i = j + 1
            if inner == "*":
                steps.append(("all",))
            elif inner.startswith("?("):
                m = re.match(r"\?\(@\.([\w.\\/-]+)\s*(==|!=)\s*['\"]?([^'\"]*)['\"]?\)", inner)
                if m:
                    steps.append(("filter", _jp_steps("." + m.group(1)), m.group(2), m.group(3)))
            elif inner[:1] in "'\"":
                steps.append(("key", inner.strip("'\"")))
            else:
                try:
                    steps.append(("idx", int(inner.split(":")[0] or 0)))
                except ValueError:
                    steps.append(("key", inner))
        elif ch == "@":
            i += 1
        else:
            key = ""
            while i < len(path) and path[i] not in ".[":
                key += path[i]
                i += 1
            steps.append(("key", key))
    return steps


def jp_eval(obj, path: str) -> list:
    cur = [obj]
    for st in _jp_steps(path):
        nxt = []
        for c in cur:
            if (
                st[0] == "key"
                and isinstance(c, dict)
                and st[1] in c
                or st[0] == "idx"
                and isinstance(c, list)
                and -len(c) <= st[1] < len(c)
            ):
                nxt.append(c[st[1]])
            elif st[0] == "all":
                nxt.extend(c if isinstance(c, list) else list(c.values()) if isinstance(c, dict) else [])
            elif st[0] == "filter" and isinstance(c, list):
                for item in c:
                    vals = jp_eval(item, "".join("." + s[1] for s in st[1] if s[0] == "key"))
                    v = str(vals[0]) if vals else None
                    if (st[2] == "==" and v == st[3]) or (st[2] == "!=" and v != st[3]):
                        nxt.append(item)
        cur = nxt
    return cur


def _jp_fmt(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"))
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def jsonpath(obj, tmpl: str) -> str:
    tmpl = tmpl.strip()
    if "{" not in tmpl:
        tmpl = "{" + tmpl + "}"
    tokens = re.split(r"(\{[^{}]*\})", tmpl)

    def render(toks, cur):
        out, i = [], 0
        while i < len(toks):
            t = toks[i]
            if t.startswith("{") and t.endswith("}"):
                expr = t[1:-1].strip()
                if expr.startswith("range "):
                    depth, j = 1, i + 1
                    while j < len(toks):
                        e = toks[j][1:-1].strip() if toks[j].startswith("{") else ""
                        if e.startswith("range "):
                            depth += 1
                        elif e == "end":
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    body = toks[i + 1 : j]
                    for item in jp_eval(cur if not expr[6:].strip().startswith("$") else obj, expr[6:].strip()):
                        out.append(render(body, item))
                    i = j + 1
                    continue
                if expr == "end":
                    i += 1
                    continue
                if expr[:1] in "'\"":
                    out.append(expr[1:-1].encode().decode("unicode_escape"))
                else:
                    out.append(" ".join(_jp_fmt(v) for v in jp_eval(obj if expr.startswith("$") else cur, expr)))
            else:
                out.append(t)
            i += 1
        return "".join(out)

    return render(tokens, obj)


def custom_columns(objs: list, spec: str) -> str:
    cols = []
    for part in re.split(r",(?![^\[]*\])", spec):
        if ":" in part:
            h, p = part.split(":", 1)
            cols.append((h, p))
    rows = [[h for h, _ in cols]]
    for o in objs:
        row = []
        for _, p in cols:
            vals = jp_eval(o, p)
            row.append(",".join(_jp_fmt(v) for v in vals) if vals else "<none>")
        rows.append(row)
    return table(rows)
