from pathlib import Path

import pytest

from glotaran import __version__ as gta_version
from glotaran.builtin.io.yml.utils import load_dict
from glotaran.io import load_parameters
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.project.project import Project
from glotaran.project.result import Result
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET as example_dataset
from glotaran.testing.simulated_data.sequential_spectral_decay import (
    MODEL_YML as example_model_yml,
)
from glotaran.testing.simulated_data.sequential_spectral_decay import (
    PARAMETERS as example_parameter,
)


@pytest.fixture(scope="module")
def project_folder(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("test_project"))


@pytest.fixture(scope="module")
def project_file(project_folder):
    return Path(project_folder) / "project.gta"


@pytest.fixture(scope="module")
def test_data(tmpdir_factory):
    path = Path(tmpdir_factory.mktemp("test_project")) / "dataset_1.nc"
    save_dataset(example_dataset, path)
    return path


def test_create(project_folder):
    Project.create(project_folder)
    with pytest.raises(FileExistsError):
        assert Project.create(project_folder)


def test_open(project_folder, project_file):
    project_from_folder = Project.open(project_folder)

    project_from_file = Project.open(project_file)

    assert project_from_folder == project_from_file

    project = project_from_file

    assert project.version == gta_version
    assert not project.has_models
    assert not project.has_data
    assert not project.has_parameters
    assert not project.has_results


def test_generate_model(project_folder, project_file):
    project = Project.open(project_file)

    project.generate_model("test_model", "decay_parallel", {"nr_compartments": 5})

    model_folder = project_folder / "models"
    assert model_folder.exists()

    project.generate_model(
        "test_model", "decay_parallel", {"nr_compartments": 5}, ignore_existing=True
    )

    model_file = model_folder / "test_model.yml"
    assert model_file.exists()

    assert project.has_models

    model = project.load_model("test_model")
    assert "megacomplex_parallel_decay" in model.megacomplex

    comapartments = load_dict(model_file, is_file=True)["megacomplex"][
        "megacomplex_parallel_decay"
    ]["compartments"]

    assert len(comapartments) == 5


@pytest.mark.parametrize("name", ["test_parameter", None])
@pytest.mark.parametrize("fmt", ["yml", "yaml", "csv"])
def test_generate_parameters(project_folder, project_file, name, fmt):
    project = Project.open(project_file)

    assert project.has_models

    project.generate_parameters("test_model", name=name, fmt=fmt)

    parameter_folder = project_folder / "parameters"
    assert parameter_folder.exists()

    project.generate_parameters("test_model", name=name, fmt=fmt, ignore_existing=True)

    parameter_file_name = f"{'test_model_parameters' if name is None else name}.{fmt}"
    parameter_file = parameter_folder / parameter_file_name
    assert parameter_file.exists()

    assert project.has_parameters

    model = project.load_model("test_model")
    parameters = project.load_parameters("test_model_parameters" if name is None else name)

    for parameter in model.get_parameter_labels():
        assert parameters.has(parameter)

    assert len(list(filter(lambda p: p[0].startswith("rates"), parameters.all()))) == 5
    parameter_file.unlink()


@pytest.mark.parametrize("name", ["test_data", None])
def test_import_data(project_folder, project_file, test_data, name):
    project = Project.open(project_file)

    project.import_data(test_data, name=name)
    with pytest.raises(FileExistsError):
        project.import_data(test_data, name=name)

    project.import_data(test_data, name=name, allow_overwrite=True)
    project.import_data(test_data, name=name, ignore_existing=True)

    data_folder = project_folder / "data"
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


@pytest.mark.parametrize("name", ["test", None])
def test_run_optimization(project_folder, project_file, name):
    project = Project.open(project_file)

    model_file = project_folder / "models" / "sequential.yml"
    model_file.write_text(example_model_yml)

    parameters_file = project_folder / "parameters" / "sequential.csv"
    save_parameters(example_parameter, parameters_file, allow_overwrite=True)

    data_folder = project_folder / "data"
    assert data_folder.exists()
    data_file = data_folder / "dataset_1.nc"
    data_file.unlink()
    save_dataset(example_dataset, data_file)

    assert project.has_models
    assert project.has_parameters
    assert project.has_data

    name = name or "sequential"

    for i in range(2):
        project.optimize(
            model="sequential",
            parameters="sequential",
            maximum_number_function_evaluations=1,
            name=name,
        )
        assert project.has_results
        result_name = f"{name}_run_{i}"
        assert (project_folder / "results" / result_name).exists()
    model_file.unlink()
    parameters_file.unlink()


def test_load_result(project_folder, project_file):
    project = Project.open(project_file)

    assert project_folder / "results" / "test_run_0" == project.get_result_path("test_run_0")

    result = project.load_result("test_run")
    assert isinstance(result, Result)


def test_generators_allow_overwrite(project_folder, project_file):
    """Overwrite doesn't throw an exception.

    This is the last test not to interfer with other tests.
    """
    project = Project.open(project_file)

    model_file = project_folder / "models/test_model.yml"
    assert model_file.is_file()

    parameter_file = project_folder / "parameters/test_parameters.csv"
    assert parameter_file.is_file()

    parameters = load_parameters(parameter_file)

    assert len(list(filter(lambda p: p[0].startswith("rates"), parameters.all()))) == 5

    project.generate_model(
        "test_model", "decay_parallel", {"nr_compartments": 3}, allow_overwrite=True
    )
    new_model = project.load_model("test")
    assert "megacomplex_parallel_decay" in new_model.megacomplex

    comapartments = load_dict(model_file, is_file=True)["megacomplex"][
        "megacomplex_parallel_decay"
    ]["compartments"]

    assert len(comapartments) == 3

    project.generate_parameters("test", allow_overwrite=True)
    parameters = load_parameters(parameter_file)

    assert len(list(filter(lambda p: p[0].startswith("rates"), parameters.all()))) == 3
