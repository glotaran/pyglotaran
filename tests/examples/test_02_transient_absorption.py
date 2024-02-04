"""Multi dataset transient absorption spectroscopy analysis integration tests."""

from __future__ import annotations

from pathlib import Path

from pyreporoot import project_root
from pytest import approx

from glotaran.io import load_dataset
from glotaran.io import load_parameters
from glotaran.io import load_scheme

repo_root = Path(project_root(__file__, root_files=["pyproject.toml"]))
case_folder = repo_root / "examples/case_studies/02-transient_absorption"


def test_target_analysis_single_dataset():
    """Test"""
    fit_scheme = load_scheme(case_folder / "scheme.yml", format_name="yml")
    parameters = load_parameters(case_folder / "parameters.yml")
    data_path = case_folder / "data/demo_data_Hippius_etal_JPCC2007_111_13988_Figs5_9.ascii"
    fit_scheme.load_data({"dataset1": load_dataset(data_path)})
    result = fit_scheme.optimize(parameters=parameters, maximum_number_function_evaluations=3)
    assert result is not None
    assert result.data["dataset1"].root_mean_square_error == approx(0.0007, abs=0.0001)


def test_target_analysis_two_datasets():
    """Test"""
    scheme = load_scheme(case_folder / "scheme_2d_co_co2.yml")
    parameters = load_parameters(case_folder / "parameters_2d_co_co2.yml")
    dataset1 = load_dataset(case_folder / "data/2016co_tol.ascii")
    dataset2 = load_dataset(case_folder / "data/2016c2o_tol.ascii")
    scheme.load_data(
        {
            "dataset1": dataset1,
            "dataset2": dataset2,
        }
    )
    result = scheme.optimize(parameters=parameters, maximum_number_function_evaluations=3)
    assert result is not None
    assert result.data["dataset1"].root_mean_square_error < 1  # TBD
    # TODO: define some meaningful assertion for the result


if __name__ == "__main__":
    test_target_analysis_two_datasets()
    test_target_analysis_single_dataset()
