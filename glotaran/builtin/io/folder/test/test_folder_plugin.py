from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from glotaran.io import save_result
from glotaran.project.test.test_result import dummy_result  # noqa: F401

if TYPE_CHECKING:
    from typing import Literal

    from py.path import local as TmpDir

    from glotaran.project.result import Result


@pytest.mark.parametrize("format_name", ("folder", "legacy"))
def test_save_result_folder(
    tmpdir: TmpDir,
    dummy_result: Result,  # noqa: F811
    format_name: Literal["folder", "legacy"],
):
    """Check all files exist."""

    result_dir = Path(tmpdir / "testresult")
    save_result(result_path=str(result_dir), format_name=format_name, result=dummy_result)

    assert (result_dir / "result.md").exists()
    assert (result_dir / "optimized_parameters.csv").exists()
    assert (result_dir / "dataset1.nc").exists()
    assert (result_dir / "dataset2.nc").exists()
    assert (result_dir / "dataset3.nc").exists()


@pytest.mark.parametrize("format_name", ("folder", "legacy"))
def test_save_result_folder_error_path_is_file(
    tmpdir: TmpDir,
    dummy_result: Result,  # noqa: F811
    format_name: Literal["folder", "legacy"],
):
    """Raise error if result_path is a file without extension and overwrite is true."""

    result_dir = Path(tmpdir / "testresult")
    result_dir.touch()

    with pytest.raises(ValueError, match="The path '.+?' is not a directory."):
        save_result(
            result_path=str(result_dir),
            format_name=format_name,
            result=dummy_result,
            allow_overwrite=True,
        )
