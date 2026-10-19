import pytest

from aiolab.content import solutions

requires_solutions = pytest.mark.skipif(not solutions.available(), reason="private solutions repository not installed")


@pytest.fixture(autouse=True)
def isolated_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("AIOLAB_HOME", str(tmp_path))
