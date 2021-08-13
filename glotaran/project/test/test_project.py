import os
from pathlib import Path

import pytest

from glotaran import __version__ as gta_version
from glotaran.project.project import TEMPLATE
from glotaran.project.project import Project
from glotaran.project.test.test_result import dummy_data  # noqa F401


@pytest.fixture(scope="module")
def project_folder(tmpdir_factory):
    return str(tmpdir_factory.mktemp("test_project"))


@pytest.fixture(scope="module")
def project_file(project_folder):
    return Path(project_folder) / "project.gta"


@pytest.fixture(scope="module")
def dummy_data_path(tmpdir_factory, dummy_data):  # noqa F811
    path = Path(tmpdir_factory.mktemp("test_project")) / "dummydata.nc"
    dummy_data["dataset1"].to_netcdf(path)
    return path


def test_create(project_folder, project_file):
    print(project_folder)  # noqa T001
    Project.create("testproject", project_folder)
    assert project_file.exists()
    assert project_file.read_text(encoding="utf-8") == TEMPLATE.format(
        gta_version=gta_version, name="testproject"
    )


def test_open(project_folder, project_file):
    print(project_folder)  # noqa T001
    project_from_folder = Project.open(project_folder)

    project_from_file = Project.open(project_file)

    assert project_from_folder == project_from_file

    project = project_from_file

    assert project.name == "testproject"
    assert project.version == gta_version
    assert not project.has_models
    assert not project.has_data
    assert not project.has_parameters


def test_generate_model(project_folder, project_file):
    project = Project.open(project_file)

    project.generate_model("test_model", "decay-parallel", {"nr_species": 5})

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

    for parameter in model.get_parameters():
        assert parameters.has(parameter)
    os.remove(parameter_file)


@pytest.mark.parametrize("name", ["test_data", None])
def test_import_data(project_folder, project_file, dummy_data, dummy_data_path, name):  # noqa F811
    project = Project.open(project_file)

    project.import_data(dummy_data_path, name=name)

    data_folder = Path(project_folder) / "data"
    assert data_folder.exists()

    data_file_name = f"{'dummydata' if name is None else name}.nc"
    data_file = data_folder / data_file_name
    assert data_file.exists()

    assert project.has_data

    data = project.load_data("dummydata" if name is None else name)
    assert data == dummy_data["dataset1"]
