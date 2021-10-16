from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from glotaran.analysis.optimize import optimize
from glotaran.examples.sequential_spectral_decay import SCHEME
from glotaran.io import save_result

if TYPE_CHECKING:

    from glotaran.project.result import Result


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""
    print(SCHEME.data["dataset_1"])
    yield optimize(SCHEME, raise_exception=True)


def test_save_result_folder(
    tmp_path: Path,
    dummy_result: Result,
):
    """Check all files exist."""

    result_dir = tmp_path / "testresult"
    save_paths = save_result(
        result_path=str(result_dir), format_name="folder", result=dummy_result
    )

    wanted_files = [
        "result.md",
        "scheme.yml",
        "model.yml",
        "initial_parameters.csv",
        "optimized_parameters.csv",
        "parameter_history.csv",
        "dataset_1.nc",
    ]
    for wanted in wanted_files:
        assert (result_dir / wanted).exists()
        assert (result_dir / wanted).as_posix() in save_paths


def test_save_result_folder_error_path_is_file(
    tmp_path: Path,
    dummy_result: Result,
):
    """Raise error if result_path is a file without extension and overwrite is true."""

    result_dir = tmp_path / "testresult"
    result_dir.touch()

    with pytest.raises(ValueError, match="The path '.+?' is not a directory."):
        save_result(
            result_path=str(result_dir),
            format_name="folder",
            result=dummy_result,
            allow_overwrite=True,
        )
