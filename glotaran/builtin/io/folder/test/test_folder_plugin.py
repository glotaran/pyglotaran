from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glotaran.analysis.optimize import optimize
from glotaran.examples.sequential import scheme
from glotaran.io import save_result

if TYPE_CHECKING:

    from py.path import local as TmpDir

    from glotaran.project.result import Result


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""
    scheme.maximum_number_function_evaluations = 1
    yield optimize(scheme)


def test_save_result_folder(
    tmpdir: TmpDir,
    dummy_result: Result,
):
    """Check all files exist."""

    result_dir = tmpdir / "test_result"
    save_result(dummy_result, str(result_dir), format_name="folder")

    assert result_dir.exists()

    wanted_files = [
        "result.md",
        "glotaran_result.yml",
        "scheme.yml",
        "model.yml",
        "initial_parameters.csv",
        "optimized_parameters.csv",
        "dataset_1.nc",
    ]
    for wanted in wanted_files:
        assert (result_dir / wanted).exists()


#  @pytest.mark.parametrize("format_name", ("folder", "legacy"))
#  def test_save_result_folder_error_path_is_file(
#      tmpdir: TmpDir,
#      dummy_result: Result,
#      format_name: Literal["folder", "legacy"],
#  ):
#      """Raise error if result_path is a file without extension and overwrite is true."""
#
#      result_dir = Path(tmpdir / "testresult")
#      result_dir.touch()
#
#      with pytest.raises(ValueError, match="The path '.+?' is not a directory."):
#          save_result(
#              result_path=str(result_dir),
#              format_name=format_name,
#              result=dummy_result,
#              allow_overwrite=True,
#          )