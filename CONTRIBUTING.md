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

CI runs the same checks on Linux and Windows.

## Adding or changing a scenario

1. Read [docs/SCENARIO_FORMAT.md](docs/SCENARIO_FORMAT.md).
2. Put the problem in `src/greenray/data/scenarios/<id>.yaml`: briefing, objectives, setup, checks and hints. Never put the answer there; the loader and a pre-commit hook reject `solution`, `explanation` and `edge_cases`.
3. Put the reference solution in `src/greenray/data/solutions/scenarios/<id>.yaml`.
4. Add the id to [docs/COVERAGE.md](docs/COVERAGE.md) under its blueprint topic.
5. `uv run greenray validate` must show `ok`: the scenario starts broken and its solution fixes it.

## Adding a command or help page

- Commands live in `src/greenray/commands/` and are registered with `@command("name", help="...")`. Registering an existing name is an error unless the module passes `override=True`.
- Raise `CommandError` with the real tool's wording for user mistakes. Any other exception is treated as a simulator bug: the shell rolls the cluster back and reports it.
- Only emit colour when `ctx.colorize` is true (the output goes straight to a colour terminal).
- Help pages live in `src/greenray/data/help/<command>/`; see `src/greenray/content/helppages.py`. Write original text in the real tool's layout, and list only options the simulator implements.

## Versioning

GreenRay AIO uses [semantic versioning](https://semver.org/):

- Below 1.0, each merged set of features bumps the **minor** version (0.3.0 → 0.4.0), and fixes bump the **patch** version (0.3.0 → 0.3.1).
- 1.0.0 is reserved for when the project goes public and the scenario format is stable.
- In the same pull request, bump `version` in `pyproject.toml` and add a section to `CHANGELOG.md`.
- To publish, tag `vX.Y.Z` on `main` (`git tag v0.4.0 && git push origin v0.4.0`). The release workflow then builds the package and creates the GitHub Release.

## Content rules

- Original wording only. Do not copy vendor documentation, man pages or exam questions.
- Never use the simulated cluster's hostnames or scenario ids in help pages or command guides.
- NVIDIA product names may describe what is being simulated, but GreenRay AIO must never look endorsed by NVIDIA.
