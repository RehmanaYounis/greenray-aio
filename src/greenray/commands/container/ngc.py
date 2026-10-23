"""The NGC CLI: configuration and the registry catalogue."""

from __future__ import annotations

import fnmatch

from ...core.errors import CommandError
from .. import command
from .images import key_valid
from .runtime import IMAGE_SIZES, NGC_CONFIG, container_init


def _ngc_org_choices(state):
    orgs = sorted({repo.split("/")[0] for repo in state.ngc.private_repos}) or ["nvidia"]
    return orgs


def _ngc_team_choices(state, org):
    teams = sorted(
        {repo.split("/")[1] for repo in state.ngc.private_repos if repo.startswith(org + "/") and repo.count("/") >= 2}
    )
    return teams + ["no-team"]


def _ngc_write_config(ctx):
    ngc = ctx.state.ngc
    text = (
        ";WARNING - This is a machine generated file.  Do not edit manually.\n"
        ';WARNING - To update local config settings, see "ngc config set -h" \n\n'
        "[CURRENT]\n"
        f"apikey = {ngc.api_key}\nformat_type = {ngc.format_type}\norg = {ngc.org or 'no-org'}\n"
        f"team = {ngc.team or 'no-team'}\nace = {ngc.ace}\n"
    )
    ctx.node.files[NGC_CONFIG] = text


def _ngc_config_set(ctx, args):
    state, ngc = ctx.state, ctx.state.ngc
    answers = [l.strip() for l in ctx.stdin.splitlines()]
    env = ctx.node.env
    key = env.get("NGC_CLI_API_KEY") or env.get("NGC_API_KEY") if not answers else None
    prompts = []

    def ask(prompt, default):
        prompts.append(prompt)
        if answers:
            v = answers.pop(0)
            return v or default
        return None

    if key is None:
        key = ask(
            f"Enter API key [{_mask(ngc.api_key) if ngc.api_key else 'no-apikey'}]. Choices: [<VALID_APIKEY>, 'no-apikey']: ",
            ngc.api_key or "no-apikey",
        )
        if key is None:
            raise CommandError(
                prompts[0] + "\nError: no interactive terminal. Feed the answers on stdin, e.g.\n"
                "  ngc config set <<EOF\n  <api key>\n  ascii\n  <org>\n  <team>\n  no-ace\n  EOF",
                1,
            )
    if key != "no-apikey" and not key_valid(state, key):
        raise CommandError("".join(prompts) + "\nInvalid apikey", 1)
    fmt = ask("Enter CLI output format type [ascii]. Choices: ['ascii', 'csv', 'json']: ", "ascii") or "ascii"
    orgs = _ngc_org_choices(state)
    org = (
        ask(f"Enter org [{ngc.org or 'no-org'}]. Choices: {orgs}: ", ngc.org or orgs[0])
        or env.get("NGC_CLI_ORG")
        or orgs[0]
    )
    if org not in orgs:
        raise CommandError(
            "".join(prompts) + f"\nInvalid org. Please re-run 'ngc config set' and choose one of: {orgs}", 1
        )
    teams = _ngc_team_choices(state, org)
    team = (
        ask(f"Enter team [{ngc.team or 'no-team'}]. Choices: {teams}: ", "no-team")
        or env.get("NGC_CLI_TEAM")
        or "no-team"
    )
    if team not in teams:
        raise CommandError(
            "".join(prompts) + f"\nInvalid team. Please re-run 'ngc config set' and choose one of: {teams}", 1
        )
    ace = ask("Enter ace [no-ace]. Choices: ['no-ace']: ", "no-ace") or "no-ace"
    ngc.api_key = "" if key == "no-apikey" else key
    ngc.configured = bool(ngc.api_key)
    ngc.format_type, ngc.org, ngc.team, ngc.ace = fmt, org, ("" if team == "no-team" else team), ace
    _ngc_write_config(ctx)
    return (
        "\n".join(p.rstrip() for p in prompts)
        + "\nValidating configuration...\nSuccessfully validated configuration.\n"
        f"Saving configuration...\nSuccessfully saved NGC configuration to {NGC_CONFIG}\n"
    )


def _mask(key):
    return "*" * 10 + key[-4:] if key else ""


def _ngc_require(ctx):
    ngc = ctx.state.ngc
    env_key = ctx.node.env.get("NGC_CLI_API_KEY", "")
    if env_key and key_valid(ctx.state, env_key):
        return
    if not ngc.configured or not key_valid(ctx.state, ngc.api_key):
        raise CommandError(
            "Missing API key and missing Email Authentication. This command requires an API key or "
            "authentication via browser login. Run 'ngc config set' to set your API key.",
            1,
        )


@command("ngc", help="NGC CLI: ngc config set|current, ngc registry image list|info")
def ngc_cli(ctx, args):
    container_init(ctx.state)
    ngc = ctx.state.ngc
    if not args or args[0] in ("-h", "--help"):
        return "usage: ngc [--version] [-h] {config,registry,org,team,user,diag} ...\n"
    if args[0] in ("--version", "version"):
        return "NGC CLI 3.48.0\n"
    if args[:2] == ["config", "set"]:
        return _ngc_config_set(ctx, args[2:])
    if args[:2] == ["config", "current"]:
        rows = [("key", "value", "source")]
        if ngc.api_key:
            rows.append(("apikey", _mask(ngc.api_key), NGC_CONFIG))
        rows += [("format_type", ngc.format_type, NGC_CONFIG)]
        if ngc.org:
            rows.append(("org", ngc.org, NGC_CONFIG))
        if ngc.team:
            rows.append(("team", ngc.team, NGC_CONFIG))
        return _ngc_table(rows)
    if args[:2] == ["config", "clear"]:
        ngc.api_key, ngc.configured, ngc.org, ngc.team = "", False, "", ""
        ctx.node.files.pop(NGC_CONFIG, None)
        return "Successfully cleared configuration.\n"
    if args[:1] == ["registry"] and len(args) >= 3 and args[1] == "image":
        _ngc_require(ctx)
        action = args[2]
        pattern = next((a for a in args[3:] if not a.startswith("-")), None)
        repos = dict(ngc.catalog)
        repos.update(ngc.private_repos)
        if action == "list":
            rows = [("Name", "Repository", "Latest Tag", "Image Size", "Updated Date", "Permission", "Signed Tag?")]
            for repo, tags in sorted(repos.items()):
                if pattern and not _glob_match(pattern.split(":")[0], repo):
                    continue
                perm = "unlocked" if repo in ngc.catalog else "Accessible"
                rows.append(
                    (
                        repo.rsplit("/", 1)[-1].replace("-", " ").title(),
                        repo,
                        tags[0],
                        next((v for k, v in IMAGE_SIZES.items() if repo.endswith(k)), "1.2 GB"),
                        "May 28, 2024",
                        perm,
                        "False",
                    )
                )
            return _ngc_table(rows)
        if action == "info":
            if not pattern:
                raise CommandError(
                    "ngc registry image info: error: the following arguments are required: <image>[:<tag>]", 2
                )
            repo, _, tag = pattern.partition(":")
            if repo not in repos:
                raise CommandError(f"Error: '{repo}' could not be found.", 1)
            if tag and tag not in repos[repo]:
                raise CommandError(f"Error: Tag '{tag}' could not be found for image '{repo}'.", 1)
            return _ngc_table(
                [
                    ("Image Information", ""),
                    ("Name", f"{repo}:{tag or repos[repo][0]}"),
                    ("Architecture", "amd64, arm64"),
                    ("Schema Version", "2"),
                    ("Tags", ", ".join(repos[repo])),
                ]
            )
        raise CommandError(
            f"ngc registry image: invalid choice: '{action}' (choose from 'info', 'list', 'pull', 'push', 'remove')", 2
        )
    if args[:2] == ["org", "list"] or args[:2] == ["user", "who"]:
        _ngc_require(ctx)
        return _ngc_table([("Name", "Org"), ("aiops-admin", ngc.org or "no-org")])
    raise CommandError(f"ngc: error: argument: invalid choice: '{args[0]}'", 2)


def _glob_match(pattern, text):

    return fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(text, pattern + "*")


def _ngc_table(rows):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep]
    for n, r in enumerate(rows):
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |")
        if n == 0:
            out.append(sep)
    out.append(sep)
    return "\n".join(out) + "\n"
