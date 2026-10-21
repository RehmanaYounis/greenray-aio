"""MkDocs hook: a page whose front matter names an `include:` file takes its body from that root file.

The root files link to `docs/...` paths, which are already the site root; the prefix is dropped.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def on_page_markdown(markdown: str, page, config, files) -> str:
    source = page.meta.get("include")
    if not source:
        return markdown
    return (ROOT / source).read_text(encoding="utf-8").replace("](docs/", "](")
