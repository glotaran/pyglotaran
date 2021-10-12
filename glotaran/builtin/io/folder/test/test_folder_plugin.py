from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from glotaran.io import save_result
from glotaran.project.test.test_result import dummy_result  # noqa: F401

if TYPE_CHECKING:
    from typing import Literal

    from glotaran.project.result import Result


@pytest.mark.parametrize("format_name", ("folder", "legacy"))
def test_save_result_folder(
    tmp_path: Path,
    dummy_result: Result,  # noqa: F811
    format_name: Literal["folder", "legacy"],
):
    """Check all files exist."""

    result_dir = tmp_path / "testresult"
    save_paths = save_result(
        result_path=str(result_dir), format_name=format_name, result=dummy_result
    )

    wanted_files = [
        "result.md",
        "scheme.yml",
        "model.yml",
        "initial_parameters.csv",
        "optimized_parameters.csv",
        "parameter_history.csv",
        "dataset1.nc",
        "dataset2.nc",
        "dataset3.nc",
    ]
    for wanted in wanted_files:
        assert (result_dir / wanted).exists()
        assert (result_dir / wanted).as_posix() in save_paths


@pytest.mark.parametrize("format_name", ("folder", "legacy"))
def test_save_result_folder_error_path_is_file(
    tmp_path: Path,
    dummy_result: Result,  # noqa: F811
    format_name: Literal["folder", "legacy"],
):
    """Raise error if result_path is a file without extension and overwrite is true."""

    result_dir = tmp_path / "testresult"
    result_dir.touch()

    with pytest.raises(ValueError, match="The path '.+?' is not a directory."):
        save_result(
            result_path=str(result_dir),
            format_name=format_name,
            result=dummy_result,
            allow_overwrite=True,
        )
