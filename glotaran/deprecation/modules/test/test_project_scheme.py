"""Test deprecated functionality in 'glotaran.project.schmeme'."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import xarray as xr

from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.project.scheme import Scheme
from glotaran.testing.simulated_data.parallel_spectral_decay import DATASET
from glotaran.testing.simulated_data.parallel_spectral_decay import MODEL
from glotaran.testing.simulated_data.parallel_spectral_decay import PARAMETER

if TYPE_CHECKING:
    from pathlib import Path


def test_scheme_from_yaml_file_method(tmp_path: Path):
    """Create Scheme from file."""
    scheme_path = tmp_path / "scheme.yml"

    model_yml_str = """
    megacomplex:
        m1:
            type: decay
            k_matrix: []
    dataset:
        dataset1:
            megacomplex: [m1]
    """
    model_path = tmp_path / "model.yml"
    model_path.write_text(model_yml_str)

    parameter_path = tmp_path / "parameters.yml"
    parameter_path.write_text("[1.0, 67.0]")

    dataset_path = tmp_path / "dataset.nc"
    xr.DataArray([[1, 2, 3]], coords=[("e", [1]), ("c", [1, 2, 3])]).to_dataset(
        name="data"
    ).to_netcdf(dataset_path)

    scheme_path.write_text(
        f"""
        model: {model_path}
        parameters: {parameter_path}
        maximum_number_function_evaluations: 42
        data:
            dataset1: {dataset_path}"""
    )

    _, result = deprecation_warning_on_call_test_helper(
        Scheme.from_yaml_file, args=[str(scheme_path)], raise_exception=True
    )

    assert isinstance(result, Scheme)


def test_scheme_group_tolerance():
    """Argument ``group_tolerance`` raises deprecation and maps to ``clp_link_tolerance``."""
    model, parameters, dataset = MODEL, PARAMETER, DATASET

    warnings, result = deprecation_warning_on_call_test_helper(
        Scheme,
        args=(model, parameters, {"dataset": dataset}),
        kwargs={"group_tolerance": 1},
        raise_exception=True,
    )

    assert isinstance(result, Scheme)
    assert result.clp_link_tolerance == 1
    assert "glotaran.project.Scheme(..., clp_link_tolerance=...)" in warnings[0].message.args[0]


@pytest.mark.parametrize(
    "group",
    (True, False),
)
def test_scheme_group(group: bool):
    """Argument ``group`` raises deprecation and maps to ``dataset_groups.default.link_clp``."""
    model, parameters, dataset = MODEL, PARAMETER, DATASET

    warnings, result = deprecation_warning_on_call_test_helper(
        Scheme,
        args=(model, parameters, {"dataset": dataset}),
        kwargs={"group": group},
        raise_exception=True,
    )

    assert isinstance(result, Scheme)
    assert result.model.dataset_group_models["default"].link_clp == group
    assert "<model_file>dataset_groups.default.link_clp" in warnings[0].message.args[0]


@pytest.mark.parametrize(
    "non_negative_least_squares, expected",
    ((True, "non_negative_least_squares"), (False, "variable_projection")),
)
def test_scheme_non_negative_least_squares(non_negative_least_squares: bool, expected: str):
    """Argument ``non_negative_least_squares`` raises deprecation and maps to
    ``dataset_groups.default.residual_function``.
    """
    model, parameters, dataset = MODEL, PARAMETER, DATASET

    warnings, result = deprecation_warning_on_call_test_helper(
        Scheme,
        args=(model, parameters, {"dataset": dataset}),
        kwargs={"non_negative_least_squares": non_negative_least_squares},
        raise_exception=True,
    )

    assert isinstance(result, Scheme)
    assert result.model.dataset_group_models["default"].residual_function == expected
    assert "<model_file>dataset_groups.default.residual_function" in warnings[0].message.args[0]
