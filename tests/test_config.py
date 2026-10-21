"""Settings come from GREENRAY_* variables; the progress folder is created on first use."""

from greenray import config


def test_env_reads_prefixed_variables(monkeypatch):
    monkeypatch.setenv("GREENRAY_SOLUTIONS", "/new")
    assert config.env("SOLUTIONS") == "/new"
    monkeypatch.setenv("GREENRAY_SOLUTIONS", "")
    assert config.env("SOLUTIONS", "fallback") == "fallback"


def test_home_is_created_where_asked(monkeypatch, tmp_path):
    monkeypatch.setenv("GREENRAY_HOME", str(tmp_path / "profile"))
    assert config.home() == tmp_path / "profile"
    assert (tmp_path / "profile").is_dir()


def test_debug_flag(monkeypatch):
    monkeypatch.delenv("GREENRAY_DEBUG", raising=False)
    assert not config.debug()
    monkeypatch.setenv("GREENRAY_DEBUG", "1")
    assert config.debug()
