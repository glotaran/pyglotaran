"""Test deprecated functionality in 'glotaran.project.schmeme'."""
from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from glotaran.deprecation.modules.test import deprecation_warning_on_call_test_helper
from glotaran.project.scheme import Scheme

if TYPE_CHECKING:
    from pathlib import Path


def test_Scheme_from_yaml_file_method(tmp_path: Path):
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
        model_file: {model_path}
        parameters_file: {parameter_path}
        maximum_number_function_evaluations: 42
        data_files:
            dataset1: {dataset_path}"""
    )

    _, result = deprecation_warning_on_call_test_helper(
        Scheme.from_yaml_file, args=[str(scheme_path)], raise_exception=True
    )

    assert isinstance(result, Scheme)
