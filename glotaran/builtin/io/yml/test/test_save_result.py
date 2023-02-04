from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from textwrap import dedent

import pytest
from pandas.testing import assert_frame_equal

from glotaran import __version__
from glotaran.io import load_result
from glotaran.io import save_dataset
from glotaran.io import save_result
from glotaran.optimization.optimize import optimize
from glotaran.project.result import Result
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME
from glotaran.utils.io import chdir_context


@pytest.fixture
def dummy_result(tmp_path: Path):
    """Dummy result for testing."""

    @lru_cache
    def create_result():
        scheme = replace(SCHEME, maximum_number_function_evaluations=1)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        save_dataset(scheme.data["dataset_1"], tmp_path / "data/ds1.nc")
        return optimize(scheme, raise_exception=True)

    yield create_result()


@pytest.mark.parametrize("path_is_absolute", (True, False))
def test_save_result_yml(tmp_path: Path, dummy_result: Result, path_is_absolute: bool):
    """Check all files exist."""
    expected_result = dedent(
        f"""\
        number_of_function_evaluations: 1
        success: true
        termination_reason: The maximum number of function evaluations is exceeded.
        glotaran_version: {__version__}
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
        optimization_history: optimization_history.csv
        data:
          dataset_1: dataset_1.nc
        """
    )
    expected_scheme = dedent(
        """\
        model: model.yml
        parameters: initial_parameters.csv
        data:
          dataset_1: dataset_1.nc
        clp_link_tolerance: 0.0
        clp_link_method: nearest
        maximum_number_function_evaluations: 1
        add_svd: true
        ftol: 1e-08
        gtol: 1e-08
        xtol: 1e-08
        optimization_method: TrustRegionReflection
        result_path: null
        """
    )
    if path_is_absolute is True:
        result_dir = tmp_path / "testresult"
    else:
        result_dir = Path("testresult")

    assert (
        dummy_result.scheme.data["dataset_1"].source_path == (tmp_path / "data/ds1.nc").as_posix()
    )

    result_path = result_dir / "result.yml"
    with chdir_context("." if path_is_absolute is True else tmp_path):
        save_result(result_path=result_path, result=dummy_result)

        assert dummy_result.source_path == result_path.as_posix()

        assert (result_dir / "result.md").exists()
        assert (result_dir / "scheme.yml").exists()
        assert (result_dir / "scheme.yml").read_text() == expected_scheme
        assert result_path.exists()
        assert (result_dir / "initial_parameters.csv").exists()
        assert (result_dir / "optimized_parameters.csv").exists()
        assert (result_dir / "optimization_history.csv").exists()
        assert (result_dir / "dataset_1.nc").exists()
        # Original scheme object isn't changed
        assert (
            dummy_result.scheme.data["dataset_1"].source_path
            == (tmp_path / "data/ds1.nc").as_posix()
        )

        # We can't check equality due to numerical fluctuations
        got = result_path.read_text()
        print(got)
        assert expected_result in got


def test_save_result_yml_result_path_is_folder(tmp_path: Path, dummy_result: Result):
    """Save the result passing a folder instead of a file path"""
    result_folder = tmp_path / "testresult"
    save_result(result_path=result_folder, result=dummy_result)

    assert (result_folder / "result.yml").is_file()

    load_result(result_folder)


@pytest.mark.parametrize("path_is_absolute", (True, False))
def test_save_result_yml_roundtrip(tmp_path: Path, dummy_result: Result, path_is_absolute: bool):
    """Save and reloaded Result should be the same."""
    if path_is_absolute is True:
        result_dir = tmp_path / "testresult"
    else:
        result_dir = Path("testresult")
    result_path = result_dir / "result.yml"

    with chdir_context("." if path_is_absolute is True else tmp_path):
        save_result(result_path=result_path, result=dummy_result)
        result_round_tripped = load_result(result_path)

        assert dummy_result.source_path == result_path.as_posix()
        assert result_round_tripped.source_path == result_path.as_posix()

        assert_frame_equal(
            dummy_result.initial_parameters.to_dataframe(),
            result_round_tripped.initial_parameters.to_dataframe(),
        )
        assert_frame_equal(
            dummy_result.optimized_parameters.to_dataframe(),
            result_round_tripped.optimized_parameters.to_dataframe(),
        )
        assert_frame_equal(
            dummy_result.parameter_history.to_dataframe(),
            result_round_tripped.parameter_history.to_dataframe(),
        )
        assert_frame_equal(
            dummy_result.optimization_history.data, result_round_tripped.optimization_history.data
        )
