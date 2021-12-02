from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import pytest

from glotaran import __version__
from glotaran.analysis.optimize import optimize
from glotaran.examples.sequential_spectral_decay import SCHEME
from glotaran.io import save_result
from glotaran.project.result import Result


@pytest.fixture(scope="session")
def dummy_result():
    """Dummy result for testing."""
    scheme = replace(SCHEME, maximum_number_function_evaluations=1)
    print(scheme.data["dataset_1"])
    yield optimize(scheme, raise_exception=True)


def test_save_result_yml(
    tmp_path: Path,
    dummy_result: Result,
):
    """Check all files exist."""
    expected = dedent(
        """\
        number_of_function_evaluations: 1
        success: true
        termination_reason: The maximum number of function evaluations is exceeded.
        glotaran_version: 0.5.0
        free_parameter_labels:
          - rates.species_1
          - rates.species_2
          - rates.species_3
          - irf.center
          - irf.width
        scheme: scheme.yml
        initial_parameters: initial_parameters.csv
        optimized_parameters: optimized_parameters.csv
        parameter_history: parameter_history.csv
        data:
          dataset_1: dataset_1.nc
        """
    )

    result_dir = tmp_path / "testresult"
    save_result(result_path=result_dir / "result.yml", result=dummy_result)

    assert (result_dir / "result.md").exists()
    assert (result_dir / "scheme.yml").exists()
    assert (result_dir / "result.yml").exists()
    assert (result_dir / "initial_parameters.csv").exists()
    assert (result_dir / "optimized_parameters.csv").exists()
    assert (result_dir / "dataset_1.nc").exists()

    # We can't check equality due to numerical fluctuations
    assert expected in (result_dir / "result.yml").read_text()
