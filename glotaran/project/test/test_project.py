from pathlib import Path

import pytest

from glotaran import __version__ as gta_version
from glotaran.project.project import TEMPLATE
from glotaran.project.project import Project


@pytest.fixture(scope="module")
def project_folder(tmpdir_factory):
    return str(tmpdir_factory.mktemp("test_project"))


@pytest.fixture(scope="module")
def project_file(project_folder):
    return Path(project_folder) / "project.gta"


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


def test_generate_model(project_file):
    project = Project.open(project_file)

    assert not project.has_models

    project.generate_model("test_model", "decay-parallel", {"nr_species": 5})

    assert project.has_models

    model = project.load_model("test_model")
    assert "megacomplex_parallel_decay" in model.megacomplex
