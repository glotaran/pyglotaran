from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.project import Project
from glotaran.project.project import PROJECT_FILE_TEMPLATE

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_start_creates_project_file(tmp_path: Path):
    """An empty folder gets a project.gta with the recommended fields left empty."""
    project = Project.start(tmp_path / "new")

    assert project.folder == (tmp_path / "new").resolve()
    assert project.results_folder == project.folder / "results"
    assert project.exports_folder == project.folder / "exports"
    assert project.project_file.read_text(encoding="utf8") == PROJECT_FILE_TEMPLATE
    fields = load_dict(project.project_file, is_file=True)
    assert set(fields) == {
        "title",
        "description",
        "data_description",
        "creators",
        "keywords",
        "license",
        "related_identifiers",
        "funding",
    }
    assert fields["title"] is None
    assert not project.results_folder.exists()


def test_start_keeps_existing_project_file(tmp_path: Path):
    """An existing project.gta, also one written by v0.7, is used byte for byte unchanged."""
    v07_project_file = b"version: 0.7.4\n\nname: lycopene\r\n"
    (tmp_path / "project.gta").write_bytes(v07_project_file)

    project = Project.start(tmp_path)
    again = Project.start(tmp_path)

    assert (tmp_path / "project.gta").read_bytes() == v07_project_file
    assert again.folder == project.folder
    assert again.results_folder == project.results_folder


def test_two_record_collections(tmp_path: Path):
    """Two instances with different results folders share project.gta and exports."""
    with_guide = Project.start(tmp_path, results="results_with_guide")
    no_guide = Project.start(tmp_path, results="results_no_guide")

    assert with_guide.results_folder == tmp_path.resolve() / "results_with_guide"
    assert no_guide.results_folder == tmp_path.resolve() / "results_no_guide"
    assert with_guide.project_file == no_guide.project_file
    assert with_guide.exports_folder == no_guide.exports_folder


def test_paths_do_not_follow_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The default folder is the working directory, resolved once in start."""
    monkeypatch.chdir(tmp_path)
    project = Project.start()
    monkeypatch.chdir(tmp_path.parent)

    assert project.folder == tmp_path.resolve()
    assert project.results_folder == tmp_path.resolve() / "results"
    assert repr(project) == (
        f"Project.start({tmp_path.resolve().as_posix()!r}, "
        f"results={(tmp_path.resolve() / 'results').as_posix()!r})"
    )
