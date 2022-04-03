from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from glotaran.deprecation import GlotaranApiDeprecationWarning
from glotaran.io import save_result
from glotaran.optimization.optimize import optimize
from glotaran.project.result import Result
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""
    print(SCHEME.data["dataset_1"])
    yield optimize(SCHEME, raise_exception=True)


@pytest.mark.parametrize("format_name", ("folder", "legacy"))
def test_save_result_folder(
    tmp_path: Path,
    dummy_result: Result,
    format_name: Literal["folder", "legacy"],
):
    """Check all files exist."""

    result_dir = tmp_path / "testresult"
    assert not result_dir.exists()
    with pytest.warns(UserWarning) as record:
        save_paths = save_result(
            result_path=str(result_dir), format_name=format_name, result=dummy_result
        )

    assert len(record) == 1
    assert Path(record[0].filename) == Path(__file__)
    if format_name == "legacy":
        assert record[0].category == GlotaranApiDeprecationWarning
    else:
        assert record[0].category == UserWarning

    wanted_files = [
        "result.md",
        "initial_parameters.csv",
        "optimized_parameters.csv",
        "parameter_history.csv",
        "dataset_1.nc",
    ]
    if format_name == "legacy":
        wanted_files += ["scheme.yml", "model.yml", "result.yml"]
    for wanted in wanted_files:
        assert (result_dir / wanted).exists()
        assert (result_dir / wanted).as_posix() in save_paths


@pytest.mark.parametrize("format_name", ("folder", "legacy"))
def test_save_result_folder_error_path_is_file(
    tmp_path: Path,
    dummy_result: Result,
    format_name: Literal["folder", "legacy"],
):
    """Raise error if result_path is a file without extension and overwrite is true."""

    result_dir = tmp_path / "testresulterror"
    result_dir.touch()

    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="The path '.+?' is not a directory."):
            save_result(
                result_path=str(result_dir),
                format_name=format_name,
                result=dummy_result,
                allow_overwrite=True,
            )
