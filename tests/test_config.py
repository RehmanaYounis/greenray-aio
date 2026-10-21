"""Settings: GREENRAY_* variables with AIOLAB_* fallbacks, and progress-folder migration."""

import json

from greenray import config
from greenray.app import progress


def test_new_name_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("AIOLAB_SOLUTIONS", "/legacy")
    monkeypatch.setenv("GREENRAY_SOLUTIONS", "/new")
    assert config.env("SOLUTIONS") == "/new"


def test_legacy_name_still_works(monkeypatch):
    monkeypatch.delenv("GREENRAY_DEBUG", raising=False)
    monkeypatch.setenv("AIOLAB_DEBUG", "1")
    assert config.debug()


def test_legacy_progress_is_migrated_once(monkeypatch, tmp_path):
    monkeypatch.delenv("GREENRAY_HOME", raising=False)
    monkeypatch.delenv("AIOLAB_HOME", raising=False)
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)
    legacy = tmp_path / ".aiolab"
    legacy.mkdir()
    (legacy / "progress.json").write_text(json.dumps({"scenarios": {"x": {"passed": True}}, "exams": []}))
    assert progress.load()["scenarios"]["x"]["passed"]
    assert (tmp_path / ".greenray" / "progress.json").exists()
