"""Solutions come bundled; an external directory overrides them; none at all is reported clearly."""

from greenray.content import scenarios as sc
from greenray.content import solutions


def test_bundled_solutions_are_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("GREENRAY_SOLUTIONS", str(tmp_path / "nowhere"))
    monkeypatch.setattr(solutions, "_REPO_ROOT", tmp_path)
    solutions.clear_cache()
    assert solutions.solutions_dir() == solutions.BUNDLED
    assert solutions.source() == "bundled"
    assert all(s.has_solution for s in sc.load_all())


def test_external_directory_overrides_bundled(monkeypatch, tmp_path):
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "scenarios" / "tsh-power-cap-persistence.yaml").write_text("solution: [echo override]\n")
    monkeypatch.setenv("GREENRAY_SOLUTIONS", str(tmp_path))
    solutions.clear_cache()
    try:
        assert solutions.source() == str(tmp_path)
        assert sc.find("tsh-power-cap-persistence").solution == ["echo override"]
    finally:
        solutions.clear_cache()


def test_missing_everything_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("GREENRAY_SOLUTIONS", str(tmp_path / "nowhere"))
    monkeypatch.setattr(solutions, "BUNDLED", tmp_path / "gone")
    solutions.clear_cache()
    try:
        assert not solutions.available()
        assert "GREENRAY_SOLUTIONS" in solutions.missing_message()
    finally:
        solutions.clear_cache()


def test_external_directory_falls_back_to_bundled_per_file(monkeypatch, tmp_path):
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "scenarios" / "tsh-power-cap-persistence.yaml").write_text("solution: [echo override]\n")
    monkeypatch.setenv("GREENRAY_SOLUTIONS", str(tmp_path))
    solutions.clear_cache()
    try:
        assert sc.find("tsh-power-cap-persistence").solution == ["echo override"]
        assert sc.find("tsh-xid79-drained-node").has_solution
    finally:
        solutions.clear_cache()
