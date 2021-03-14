from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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
