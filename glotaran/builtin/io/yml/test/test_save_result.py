from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

from glotaran import __version__
from glotaran.io import save_result
from glotaran.project.test.test_result import dummy_result  # noqa: F401

if TYPE_CHECKING:

    from glotaran.project.result import Result


def test_save_result_yml(
    tmp_path: Path,
    dummy_result: Result,  # noqa: F811
):
    """Check all files exist."""
    expected = dedent(
        f"""\
        number_of_function_evaluations: 1
        success: true
        termination_reason: The maximum number of function evaluations is exceeded.
        glotaran_version: {__version__}
        free_parameter_labels:
          - '1'
          - '2'
        scheme: scheme.yml
        initial_parameters: initial_parameters.csv
        optimized_parameters: optimized_parameters.csv
        parameter_history: parameter_history.csv
        data:
          dataset1: dataset1.nc
          dataset2: dataset2.nc
          dataset3: dataset3.nc
        """
    )

    result_dir = tmp_path / "testresult"
    save_result(result_path=result_dir / "result.yml", result=dummy_result)

    assert (result_dir / "result.md").exists()
    assert (result_dir / "scheme.yml").exists()
    assert (result_dir / "result.yml").exists()
    assert (result_dir / "initial_parameters.csv").exists()
    assert (result_dir / "optimized_parameters.csv").exists()
    assert (result_dir / "dataset1.nc").exists()
    assert (result_dir / "dataset2.nc").exists()
    assert (result_dir / "dataset3.nc").exists()
    # We can't check equality due to numerical fluctuations
    assert expected in (result_dir / "result.yml").read_text()
