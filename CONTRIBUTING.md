# Contributing to GreenRay AIO

## Set up

```bash
uv sync                       # Python, dependencies and dev tools from uv.lock
uv run pre-commit install     # optional: lint every commit
```

## Before you open a pull request

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv run greenray validate
```

CI runs the same checks on Linux and Windows, and builds the documentation site:

```bash
uv sync --group docs
uv run mkdocs serve          # preview at http://127.0.0.1:8000
```

The site's pages live in `docs/`. The architecture, contributing, security and changelog pages are filled from the root files of the same name, so edit those, not the stubs.

## Adding or changing a scenario

1. Read [docs/scenario-format.md](docs/scenario-format.md).
2. Put the problem in `src/greenray/data/scenarios/<id>.yaml`: briefing, objectives, setup, checks and hints. Never put the answer there; the loader and a pre-commit hook reject `solution`, `explanation` and `edge_cases`.
3. Put the reference solution in `src/greenray/data/solutions/scenarios/<id>.yaml`.
4. Add the id to [docs/coverage.md](docs/coverage.md) under its blueprint topic.
5. `uv run greenray validate` must show `ok`: the scenario starts broken and its solution fixes it.

## Adding a command or help page

- Commands live in `src/greenray/commands/` and are registered with `@command("name", help="...")`. Registering an existing name is an error unless the module passes `override=True`.
- Raise `CommandError` with the real tool's wording for user mistakes. Any other exception is treated as a simulator bug: the shell rolls the cluster back and reports it.
- Only emit colour when `ctx.colorize` is true (the output goes straight to a colour terminal).
- Help pages live in `src/greenray/data/help/<command>/`; see `src/greenray/content/helppages.py`. Write original text in the real tool's layout, and list only options the simulator implements.
- `tests/test_strictness.py` is a table of invocations the real tools reject, with the wording they use. A new command should add its own rows: a simulator that accepts a wrong command teaches a wrong habit. When a user reports the real tool being stricter than the simulator, the fix starts with a row in that table.
- `tests/test_flags.py` runs every option listed on every help page and fails when one changes neither the output nor the cluster. A new help page therefore needs a `Baseline` entry there that puts the command in a situation where its options can show an effect, and any option that is a genuine no-op in that situation goes in `NO_VISIBLE_EFFECT` with the reason. Listing an option on a help page is a promise that the simulator honours it.
- `tests/test_golden.py` pins the exact output of the everyday tools on the healthy cluster. A deliberate layout change is made with `UPDATE_GOLDEN=1` after comparing the new text with the real tool.

## Versioning

GreenRay AIO uses [semantic versioning](https://semver.org/):

- A merged set of features bumps the **minor** version (1.0.0 → 1.1.0), a fix bumps the **patch** version (1.0.0 → 1.0.1), and a change that breaks existing scenario files bumps the **major** version.
- In the same pull request, bump `version` in `pyproject.toml` and add a section to `CHANGELOG.md`.
- Merging into `main` publishes. The release workflow reads the version from `pyproject.toml`, creates the `vX.Y.Z` tag if it does not exist, and then publishes to PyPI and GHCR and creates the GitHub Release with that version's changelog section as its notes. A merge that keeps the version, such as a documentation change, releases nothing. The workflow refuses a version that has no changelog section.

## Content rules

- Original wording only. Do not copy vendor documentation, man pages or exam questions.
- Never use the simulated cluster's hostnames or scenario ids in help pages or command guides.
- NVIDIA product names may describe what is being simulated, but GreenRay AIO must never look endorsed by NVIDIA.
