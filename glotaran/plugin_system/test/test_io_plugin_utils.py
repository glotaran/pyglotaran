from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glotaran.plugin_system.io_plugin_utils import bool_str_repr
from glotaran.plugin_system.io_plugin_utils import bool_table_repr
from glotaran.plugin_system.io_plugin_utils import inferr_file_format
from glotaran.plugin_system.io_plugin_utils import not_implemented_to_value_error
from glotaran.plugin_system.io_plugin_utils import protect_from_overwrite

if TYPE_CHECKING:
    from pathlib import Path

    from py.path import local as LocalPath


@pytest.mark.parametrize(
    "extension,expected",
    (
        (
            "yaml",
            "yaml",
        ),
        (
            "sdt",
            "sdt",
        ),
        (
            "something.sdt",
            "sdt",
        ),
    ),
)
def test_inferr_file_format(tmp_path: Path, extension: str, expected: str):
    """Inferr type from existing files with extension."""
    file_path = tmp_path / f"dummy.{extension}"
    file_path.touch()

    assert inferr_file_format(file_path) == expected


def test_inferr_file_format_no_extension(tmp_path: Path):
    """Raise error if file has no extension."""
    file_path = tmp_path / "dummy"
    file_path.touch()

    with pytest.raises(
        ValueError, match="Cannot determine format of file .+?, please provide an explicit format"
    ):
        inferr_file_format(file_path)


@pytest.mark.parametrize("is_file", (True, False))
def test_inferr_file_format_allow_folder(tmp_path: Path, is_file: bool):
    """If there is no extension, return folder."""
    file_path = tmp_path / "dummy"
    if is_file:
        file_path.touch()

    assert inferr_file_format(file_path, allow_folder=True) == "folder"


def test_inferr_file_format_none_existing_file():
    """Raise error if file does not exists."""
    with pytest.raises(ValueError, match="There is no file "):
        inferr_file_format("none-existing-file.yml")


def test_not_implemented_to_value_error():
    """Redirect not NotImplementedError to ValueError."""

    @not_implemented_to_value_error
    def dummy():
        raise NotImplementedError("This isn't working")

    with pytest.raises(ValueError, match="This isn't working"):
        dummy()


def test_protect_from_overwrite_allow_overwrite(tmp_path: Path):
    """Nothing happens when allow_overwrite=True"""
    path = tmp_path / "dummy.txt"
    path.touch()

    protect_from_overwrite(path, allow_overwrite=True)


def test_protect_from_overwrite_file_exists(tmp_path: Path):
    """Error by default if file exists."""
    path = tmp_path / "dummy.txt"
    path.touch()

    with pytest.raises(FileExistsError, match="The file .+? already exists"):
        protect_from_overwrite(path)


def test_protect_from_overwrite_empty_dir(tmpdir: LocalPath):
    """Nothing happens when the folder is empty"""
    path = tmpdir / "dummy"
    path.mkdir()

    protect_from_overwrite(path)


def test_protect_from_overwrite_not_empty_dir(tmpdir: LocalPath):
    """Error by default if path is an not empty dir."""
    path = tmpdir / "dummy"
    path.mkdir()
    (path / "dummy.txt").write_text("test", encoding="utf8")

    with pytest.raises(FileExistsError, match="The folder .+? already exists and is not empty"):
        protect_from_overwrite(path)


def test_bool_str_repr():
    """Only bools are replaced"""
    assert bool_str_repr(True) == "*"
    assert bool_str_repr(False) == "/"
    assert bool_str_repr("foo") == "foo"
    assert bool_str_repr(0) == 0
    assert bool_str_repr(1) == 1


def test_bool_table_repr():
    """All bools get replaced by their repr."""
    table_data = [["foo", True, False], ["bar", False, True]]
    expected = [["foo", "*", "/"], ["bar", "/", "*"]]

    for row, expected_row in zip(bool_table_repr(table_data), expected):
        assert list(row) == expected_row
