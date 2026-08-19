import json
from pathlib import Path

import pandas as pd

from qtf.utils.paths import (
    relativize_absolute_paths,
    repo_relative_path,
    write_portable_csv,
)


def test_repo_relative_path_for_inside_and_outside_repository(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.chdir(repo)

    assert repo_relative_path(repo / "outputs" / "result.json") == "outputs/result.json"
    assert repo_relative_path(tmp_path / "external.pdb") == "../external.pdb"


def test_relativize_absolute_paths_is_recursive(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.chdir(repo)
    payload = {
        "path": str(repo / "run_outputs" / "result.json"),
        "nested": [Path(repo / "model.pdb"), {"external": str(tmp_path / "native.pdb")}],
        "label": "5AWL",
    }

    portable = relativize_absolute_paths(payload)

    assert portable == {
        "path": "run_outputs/result.json",
        "nested": ["model.pdb", {"external": "../native.pdb"}],
        "label": "5AWL",
    }
    assert not any(str(repo) in text for text in json.dumps(portable).splitlines())


def test_relativize_paths_embedded_in_commands_and_logs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.chdir(repo)

    text = (
        f"{tmp_path}/env/bin/python {repo}/qtf/run.py --outdir {repo}/outputs "
        f"warning at {tmp_path}/package/module.py:68"
    )
    portable = relativize_absolute_paths(text)

    assert str(tmp_path) not in portable
    assert "../env/bin/python" in portable
    assert "qtf/run.py" in portable
    assert "outputs" in portable
    assert "../package/module.py:68" in portable


def test_write_portable_csv_relativizes_path_cells(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.chdir(repo)
    output = repo / "results.csv"

    write_portable_csv(
        pd.DataFrame([{"pdb_path": str(repo / "models" / "one.pdb"), "score": 1.0}]),
        output,
    )

    assert "models/one.pdb" in output.read_text(encoding="utf-8")
    assert str(repo) not in output.read_text(encoding="utf-8")
