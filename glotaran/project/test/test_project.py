import os
from pathlib import Path

import pytest

from glotaran import __version__ as gta_version
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.project.project import TEMPLATE
from glotaran.project.project import Project
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET as example_dataset
from glotaran.testing.simulated_data.sequential_spectral_decay import (
    MODEL_YML as example_model_yml,
)
from glotaran.testing.simulated_data.sequential_spectral_decay import (
    PARAMETERS as example_parameter,
)


@pytest.fixture(scope="module")
def project_folder(tmpdir_factory):
    return str(tmpdir_factory.mktemp("test_project"))


@pytest.fixture(scope="module")
def project_file(project_folder):
    return Path(project_folder) / "project.gta"


@pytest.fixture(scope="module")
def test_data(tmpdir_factory):
    path = Path(tmpdir_factory.mktemp("test_project")) / "dataset_1.nc"
    save_dataset(example_dataset, path)
    return path


def test_create(project_folder, project_file):
    Project.create("testproject", project_folder)
    assert project_file.exists()
    assert project_file.read_text(encoding="utf-8") == TEMPLATE.format(
        gta_version=gta_version, name="testproject"
    )


def test_open(project_folder, project_file):
    project_from_folder = Project.open(project_folder)

    project_from_file = Project.open(project_file)

    assert project_from_folder == project_from_file

    project = project_from_file

    assert project.name == "testproject"
    assert project.version == gta_version
    assert not project.has_models
    assert not project.has_data
    assert not project.has_parameters
    assert not project.has_results


def test_generate_model(project_folder, project_file):
    project = Project.open(project_file)

    project.generate_model("test_model", "decay_parallel", {"nr_compartments": 5})

    model_folder = Path(project_folder) / "models"
    assert model_folder.exists()

    model_file = model_folder / "test_model.yml"
    assert model_file.exists()

    assert project.has_models

    model = project.load_model("test_model")
    assert "megacomplex_parallel_decay" in model.megacomplex


@pytest.mark.parametrize("name", ["test_parameter", None])
@pytest.mark.parametrize("fmt", ["yml", "yaml", "csv"])
def test_generate_parameters(project_folder, project_file, name, fmt):
    project = Project.open(project_file)

    assert project.has_models

    project.generate_parameters("test_model", name=name, fmt=fmt)

    parameter_folder = Path(project_folder) / "parameters"
    assert parameter_folder.exists()

    parameter_file_name = f"{'test_model_parameters' if name is None else name}.{fmt}"
    parameter_file = parameter_folder / parameter_file_name
    assert parameter_file.exists()

    assert project.has_parameters

    model = project.load_model("test_model")
    parameters = project.load_parameters("test_model_parameters" if name is None else name)

    for parameter in model.get_parameter_labels():
        assert parameters.has(parameter)
    os.remove(parameter_file)


@pytest.mark.parametrize("name", ["test_data", None])
def test_import_data(project_folder, project_file, test_data, name):
    project = Project.open(project_file)

    project.import_data(test_data, name=name)

    data_folder = Path(project_folder) / "data"
    assert data_folder.exists()

    data_file_name = f"{'dataset_1' if name is None else name}.nc"
    data_file = data_folder / data_file_name
    assert data_file.exists()

    assert project.has_data

    data = project.load_data("dataset_1" if name is None else name)
    assert data == example_dataset


def test_create_scheme(project_folder, project_file):
    project = Project.open(project_file)

    project.generate_parameters("test_model", name="test_parameters")
    scheme = project.create_scheme(
        model="test_model", parameters="test_parameters", maximum_number_function_evaluations=1
    )

    assert "dataset_1" in scheme.data
    assert "dataset_1" in scheme.model.dataset
    assert scheme.maximum_number_function_evaluations == 1


@pytest.mark.parametrize("name", ["test_run", None])
def test_run_optimization(project_folder, project_file, name):
    project_folder = Path(project_folder)
    project = Project.open(project_file)

    model_file = project_folder / "models" / "sequential.yml"
    with open(model_file, "w") as f:
        f.write(example_model_yml)

    parameters_file = project_folder / "parameters" / "sequential.csv"
    save_parameters(example_parameter, parameters_file)

    data_folder = project_folder / "data"
    assert data_folder.exists()
    data_file = data_folder / "dataset_1.nc"
    os.remove(data_file)
    save_dataset(example_dataset, data_file)

    assert project.has_models
    assert project.has_parameters
    assert project.has_data

    project.optimize(
        model="sequential",
        parameters="sequential",
        maximum_number_function_evaluations=1,
        name=name,
    )
    assert project.has_results
    name = name or "sequential_run_0"
    assert (project_folder / "results" / name).exists()
    os.remove(model_file)
    os.remove(parameters_file)
