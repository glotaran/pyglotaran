from __future__ import annotations

from pathlib import Path

from pyreporoot import project_root
from pytest import approx

from glotaran.io import load_dataset
from glotaran.io import load_parameters
from glotaran.io import load_scheme

repo_root = Path(project_root(__file__, root_files=[".git", "pyproject.toml"]))
example_folder = repo_root / "examples/01-fluorescence"


def test_01_fluorescence():
    fit_scheme = load_scheme(example_folder / "scheme.yaml", format_name="yml")
    parameters = load_parameters(example_folder / "parameters.yaml")
    data_path = example_folder / "data/data.ascii"
    fit_scheme.load_data(
        {"dataset1": load_dataset(data_path), "dataset2": load_dataset(data_path)}
    )
    result = fit_scheme.optimize(parameters=parameters)
    assert result is not None
    assert result.data["dataset1"].root_mean_square_error == approx(43.98496093)

    # Compare the 'SAS' (EAS / DAS) for the both cases:
    # result.data['dataset1'].species_associated_estimation
    # result.data['dataset2'].species_associated_estimation
    # Currently seems incorrect.


if __name__ == "__main__":
    test_01_fluorescence()
