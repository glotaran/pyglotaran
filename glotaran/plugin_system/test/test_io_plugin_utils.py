from pathlib import Path

import pytest

from glotaran.plugin_system.io_plugin_utils import inferr_file_format
from glotaran.plugin_system.io_plugin_utils import not_implemented_to_value_error


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
