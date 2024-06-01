"""Single dataset time-resolved fluorescence spectroscopy analysis integration tests."""

from __future__ import annotations

from pathlib import Path

from pyreporoot import project_root
from pytest import approx

from glotaran.io import load_dataset
from glotaran.io import load_parameters
from glotaran.io import load_scheme

repo_root = Path(project_root(__file__, root_files=["pyproject.toml"]))
example_folder = repo_root / "examples/case_studies/01-fluorescence"


def test_target_analysis():
    fit_scheme = load_scheme(example_folder / "scheme.yaml", format_name="yml")
    parameters = load_parameters(example_folder / "parameters.yaml")
    data_path = example_folder / "data/data.ascii"
    fit_scheme.load_data({"dataset1": load_dataset(data_path)})
    result = fit_scheme.optimize(parameters=parameters)
    assert result is not None
    assert result.data["dataset1"].root_mean_square_error == approx(43.98496093)


if __name__ == "__main__":
    test_target_analysis()
