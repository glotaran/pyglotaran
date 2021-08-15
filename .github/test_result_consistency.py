""""Tests to ensure result consistency."""
from __future__ import annotations

import os
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Protocol
from warnings import warn

import numpy as np
import pandas as pd
import pytest
import xarray as xr

if TYPE_CHECKING:
    from xarray.core.coordinates import DataArrayCoordinates


REPO_ROOT = Path(__file__).parent.parent
RUN_EXAMPLES_MSG = (
    "run 'python scripts/run_examples.py run-all --headless' "
    "in the 'pyglotaran-examples' repo root."
)

# in general this list should be empty, but for none stable examples this might be needed
EXAMPLE_BLOCKLIST = [
    "study_transient_absorption_two_dataset_analysis_result_2d_co_co2",
    "ex_spectral_guidance",
]


class AllCloseFixture(Protocol):
    def __call__(
        self,
        a: np.ndarray | xr.DataArray,
        b: np.ndarray | xr.DataArray,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        xtol: int = 0,
        equal_nan: bool = False,
        print_fail: int = 5,
        record_rmse: bool = True,
    ) -> bool:
        ...


class GitError(Exception):
    """Error raised when a git interaction didn't exit with a 0 returncode."""


def get_compare_results_path() -> Path:
    """Ensure that the comparison-results exist, are up to date and return their path."""
    compare_result_folder = REPO_ROOT / "comparison-results"
    example_repo = "git@github.com:glotaran/pyglotaran-examples.git"
    if not compare_result_folder.exists():
        proc_clone = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "-b",
                "comparison-results",
                example_repo,
                str(compare_result_folder),
            ],
            capture_output=True,
        )
        if proc_clone.returncode != 0:
            raise GitError(f"Error cloning {example_repo}:\n{proc_clone.stderr.decode()}")
    if "GITHUB" not in os.environ:
        proc_fetch = subprocess.run(
            [
                "git",
                "-C",
                compare_result_folder.as_posix(),
                "fetch",
                "--depth",
                "1",
                "origin",
                "comparison-results",
            ],
            capture_output=True,
        )
        if proc_fetch.returncode != 0:
            raise GitError(f"Error fetching {example_repo}:\n{proc_fetch.stderr.decode()}")
        proc_reset = subprocess.run(
            [
                "git",
                "-C",
                compare_result_folder.as_posix(),
                "reset",
                "--hard",
                "origin/comparison-results",
            ],
            capture_output=True,
        )
        if proc_reset.returncode != 0:
            raise GitError(f"Error resetting {example_repo}:\n{proc_reset.stderr.decode()}")
    return compare_result_folder


def get_current_result_path() -> Path:
    """Get the path of the current results."""
    local_path = Path.home() / "pyglotaran_examples_results"
    ci_path = REPO_ROOT / "comparison-results-current"
    if local_path.exists():
        return local_path
    elif ci_path.exists():
        return ci_path
    else:
        raise ValueError(f"No current results present, {RUN_EXAMPLES_MSG}")


def coord_test(
    expected_coords: DataArrayCoordinates,
    current_coords: DataArrayCoordinates,
    file_name: str,
    allclose: AllCloseFixture,
    exact_match=False,
    data_var_name: str = "unknown",
):
    """Tests that coordinates are exactly equal if exact match or string coords or close."""
    for expected_coord_name, expected_coord_value in expected_coords.items():
        assert (
            expected_coord_name in current_coords.keys()
        ), f"Missing coordinate: {expected_coord_name!r} in {file_name!r}, data_var {data_var_name!r}"

        if exact_match or expected_coord_value.data.dtype == object:
            assert np.array_equal(
                expected_coord_value, current_coords[expected_coord_name]
            ), f"Coordinate value mismatch in {file_name!r}, data_var {data_var_name!r}"
        else:
            assert allclose(
                expected_coord_value, current_coords[expected_coord_name], rtol=1e-5, print_fail=20
            ), f"Coordinate value mismatch in {file_name!r}, data_var {data_var_name!r}"


def map_result_files(file_glob_pattern: str) -> dict[str, list[tuple[Path, Path]]]:
    """Load all datasets and map them in a dict."""
    result_map = defaultdict(list)
    compare_results_path = get_compare_results_path()
    current_result_path = get_current_result_path()
    for expected_result_file in compare_results_path.rglob(file_glob_pattern):
        key = (
            expected_result_file.relative_to(compare_results_path)
            .parent.as_posix()
            .replace("/", "_")
        )
        if key in EXAMPLE_BLOCKLIST:
            continue
        current_result_file = current_result_path / expected_result_file.relative_to(
            compare_results_path
        )
        if current_result_file.exists():
            result_map[key].append((expected_result_file, current_result_file))
        else:
            warn(
                UserWarning(
                    f"No current result for: {expected_result_file.as_posix()}, {RUN_EXAMPLES_MSG}"
                )
            )
    return result_map


@lru_cache(maxsize=1)
def map_result_data() -> dict[str, list[tuple[xr.Dataset, xr.Dataset, str]]]:
    """Load all datasets and map them in a dict."""
    result_map = defaultdict(list)
    result_file_map = map_result_files(file_glob_pattern="*.nc")
    for key, path_list in result_file_map.items():
        for expected_result_file, current_result_file in path_list:
            result_map[key].append(
                (
                    xr.open_dataset(expected_result_file),
                    xr.open_dataset(current_result_file),
                    expected_result_file.name,
                )
            )
    return result_map


@lru_cache(maxsize=1)
def map_result_parameters() -> dict[str, list[pd.DataFrame]]:
    """Load all optimized parameter files and map them in a dict."""

    result_map = defaultdict(list)
    result_file_map = map_result_files(file_glob_pattern="*.csv")
    for key, path_list in result_file_map.items():
        for expected_result_file, current_result_file in path_list:
            compare_df = pd.DataFrame(
                {
                    "expected": pd.read_csv(expected_result_file, index_col="label")["value"],
                    "current": pd.read_csv(current_result_file, index_col="label")["value"],
                }
            )
            result_map[key].append(compare_df)
    return result_map


@pytest.mark.parametrize("result_name", map_result_data().keys())
def test_original_data_exact_consistency(
    allclose: AllCloseFixture,
    result_name: str,
):
    """The original data need to be exactly the same."""
    for expected_result, current_result, file_name in map_result_data()[result_name]:
        assert np.array_equal(
            expected_result.data.data, current_result.data.data
        ), f"Original data mismatch: {result_name!r} in {file_name!r}"
        coord_test(
            expected_result.data.coords,
            current_result.data.coords,
            file_name,
            allclose,
            exact_match=True,
            data_var_name="data",
        )


@pytest.mark.parametrize("result_name", map_result_parameters().keys())
def test_result_parameter_consistency(
    allclose: AllCloseFixture,
    result_name: str,
):
    """Optimized parameters need to be approximately the same"""
    for compare_df in map_result_parameters()[result_name]:
        assert allclose(
            compare_df["expected"].values, compare_df["current"].values, print_fail=20
        ), f"Parameter Mismatch: {compare_df.index}"


@pytest.mark.parametrize("result_name", map_result_data().keys())
def test_result_attr_consistency(
    allclose: AllCloseFixture,
    result_name: str,
):
    """Resultdataset attributes need to be approximately the same."""
    for expected, current, file_name in map_result_data()[result_name]:
        for expected_attr_name, expected_attr_value in expected.attrs.items():

            assert (
                expected_attr_name in current.attrs.keys()
            ), f"Missing result attribute: {expected_attr_name!r} in {file_name!r}"

            assert allclose(
                expected_attr_value, current.attrs[expected_attr_name], rtol=1e-4, print_fail=20
            ), f"Result attr value mismatch: {expected_attr_name!r} in {file_name!r}"


@pytest.mark.parametrize("result_name", map_result_data().keys())
def test_result_data_var_consistency(
    allclose: AllCloseFixture,
    result_name: str,
):
    """Result dataset data variables need to be approximately the same."""
    for expected_result, current_result, file_name in map_result_data()[result_name]:
        for expected_var_name, expected_var_value in expected_result.data_vars.items():
            if expected_var_name != "data":

                # weighted_data were always calculated and now will only be calculated when weights are applied
                if (
                    expected_var_name == "weighted_data"
                    and expected_var_name not in current_result.data_vars
                ):
                    continue

                assert (
                    expected_var_name in current_result.data_vars
                ), f"Missing data_var: {expected_var_name!r} in {file_name!r}"
                current_data = current_result.data_vars[expected_var_name]

                abs_tol = 1e-8  # default value

                # due to platform specific differences in the low level code that generates the SVD
                # the values might differ from expected which were generated on linux
                if "singular_vectors" in expected_var_name:  # type:ignore[operator]
                    abs_tol = 1e-5

                assert allclose(
                    expected_var_value.data,
                    current_data.data,
                    atol=abs_tol,
                    rtol=1e-3,
                    print_fail=20,
                ), f"Result data_var data mismatch: {expected_var_name!r} in {file_name!r}"

                coord_test(
                    expected_var_value.coords,
                    current_data.coords,
                    file_name,
                    allclose,
                    data_var_name=expected_var_name,
                )
