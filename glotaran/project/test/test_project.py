from __future__ import annotations

import sys
from importlib.metadata import distribution
from pathlib import Path
from shutil import rmtree
from textwrap import dedent
from typing import Literal

import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.recwarn import WarningsRecorder
from IPython.core.formatters import format_display_data

from glotaran import __version__ as gta_version
from glotaran.builtin.io.yml.utils import load_dict
from glotaran.io import load_parameters
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.project.project import Project
from glotaran.project.project_registry import AmbiguousNameWarning
from glotaran.project.result import Result
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET as example_dataset
from glotaran.testing.simulated_data.sequential_spectral_decay import (
    MODEL_YML as example_model_yml,
)
from glotaran.testing.simulated_data.sequential_spectral_decay import (
    PARAMETERS as example_parameter,
)
from glotaran.utils.io import chdir_context


@pytest.fixture(scope="module")
def project_folder(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("test_project"))


@pytest.fixture(scope="module")
def project_file(project_folder: Path):
    return Path(project_folder) / "project.gta"


@pytest.fixture(scope="module")
def test_data(tmpdir_factory):
    path = Path(tmpdir_factory.mktemp("test_project")) / "dataset_1.nc"
    save_dataset(example_dataset, path)
    return path


def test_init(project_folder: Path, project_file: Path):
    """Init project directly."""
    file_only_init = Project(project_file)
    file_and_folder_init = Project(project_file)

    assert file_only_init == file_and_folder_init


def test_create(project_folder: Path):
    rmtree(project_folder)
    Project.create(project_folder)
    with pytest.raises(FileExistsError):
        assert Project.create(project_folder)


def test_open(project_folder: Path, project_file: Path):
    rmtree(project_folder)
    project_from_folder = Project.open(project_folder)

    project_from_file = Project.open(project_file)

    assert project_from_folder == project_from_file

    project = project_from_file

    assert project.version == gta_version
    assert not project.has_models
    assert not project.has_data
    assert not project.has_parameters
    assert not project.has_results

    # Will cause following tests to fails on bad fuzzy matching due to higher string sort order
    (project_folder / "data/dataset_1-bad.nc").touch()
    (project_folder / "models/test_model-bad.yml").touch()
    (project_folder / "parameters/test_parameters-bad.yml").touch()


def test_open_diff_version(tmp_path: Path):
    """Loading from file overwrites current version."""
    project_file = tmp_path / "project.gta"
    project_file.write_text("version: 0.1.0")

    project = Project.open(project_file)

    assert project.version == "0.1.0"


def test_generate_model(project_folder: Path, project_file: Path):
    project = Project.open(project_file)

    project.generate_model("test_model", "decay_parallel", {"nr_compartments": 5})

    model_folder = project_folder / "models"
    assert model_folder.is_dir()

    project.generate_model(
        "test_model", "decay_parallel", {"nr_compartments": 5}, ignore_existing=True
    )

    assert project.get_models_directory() == model_folder

    model_file = model_folder / "test_model.yml"
    assert model_file.is_file()

    assert project.has_models

    model = project.load_model("test_model")
    assert "megacomplex_parallel_decay" in model.megacomplex

    comapartments = load_dict(model_file, is_file=True)["megacomplex"][
        "megacomplex_parallel_decay"
    ]["compartments"]

    assert len(comapartments) == 5

    with pytest.raises(FileExistsError) as exc_info:
        project.generate_model("test_model", "decay_parallel", {"nr_compartments": 5})

    assert str(exc_info.value) == "Model 'test_model' already exists and `allow_overwrite=False`."


@pytest.mark.parametrize("name", ["test_parameter", None])
@pytest.mark.parametrize("fmt", ["yml", "yaml", "csv"])
def test_generate_parameters(
    project_folder: Path, project_file: Path, name: str | None, fmt: Literal["yml", "yaml", "csv"]
):
    project = Project.open(project_file)

    assert project.has_models

    project.generate_parameters("test_model", parameters_name=name, format_name=fmt)

    parameter_folder = project_folder / "parameters"
    assert parameter_folder.is_dir()

    project.generate_parameters(
        "test_model", parameters_name=name, format_name=fmt, ignore_existing=True
    )

    parameter_file_name = f"{'test_model_parameters' if name is None else name}.{fmt}"
    parameter_file = parameter_folder / parameter_file_name
    assert parameter_file.is_file()
    assert project.get_parameters_directory() == parameter_folder

    assert project.has_parameters

    parameters_key = "test_model_parameters" if name is None else name
    model = project.load_model("test_model")
    parameters = project.load_parameters(parameters_key)

    for parameter in model.get_parameter_labels():
        assert parameters.has(parameter)

    assert len(list(filter(lambda p: p.label.startswith("rates"), parameters.all()))) == 5

    with pytest.raises(FileExistsError) as exc_info:
        project.generate_parameters("test_model", parameters_name=name, format_name=fmt)

    assert (
        str(exc_info.value)
        == f"Parameters '{parameters_key}' already exists and `allow_overwrite=False`."
    )

    parameter_file.unlink()


@pytest.mark.parametrize("name", ["test_data", None])
def test_import_data(project_folder: Path, project_file: Path, test_data: Path, name: str | None):
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


def test_create_scheme(project_file: Path):
    project = Project.open(project_file)

    project.generate_parameters("test_model", parameters_name="test_parameters")
    scheme = project.create_scheme(
        model_name="test_model",
        parameters_name="test_parameters",
        maximum_number_function_evaluations=1,
    )

    assert "dataset_1" in scheme.data
    assert "dataset_1" in scheme.model.dataset
    assert scheme.maximum_number_function_evaluations == 1


@pytest.mark.parametrize("name", ["test", None])
def test_run_optimization(project_folder: Path, project_file: Path, name: str | None):
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
            model_name="sequential",
            parameters_name="sequential",
            maximum_number_function_evaluations=1,
            result_name=name,
        )
        assert project.has_results
        result_name = f"{name}_run_000{i}"
        assert (project_folder / "results" / result_name).exists()
    model_file.unlink()
    parameters_file.unlink()


def test_load_result(project_folder: Path, project_file: Path, recwarn: WarningsRecorder):
    """No warnings if name contains run specifier or latest is true."""
    project = Project.open(project_file)

    assert project_folder / "results" / "test_run_0000" == project.get_result_path("test_run_0000")

    result = project.load_result("test_run_0000")
    assert isinstance(result, Result)

    assert project_folder / "results" / "test_run_0001" == project.get_result_path(
        "test", latest=True
    )

    assert isinstance(project.load_result("test", latest=True), Result)

    assert project_folder / "results" / "test_run_0001" == project.get_latest_result_path("test")

    assert isinstance(project.load_latest_result("test"), Result)

    assert len(recwarn) == 0


def test_load_result_warnings(project_folder: Path, project_file: Path):
    """Warn when using fallback to latest result."""
    project = Project.open(project_file)

    expected_warning_text = (
        "Result name 'test' is missing the run specifier, "
        "falling back to try getting latest result. "
        "Use latest=True to mute this warning."
    )

    with pytest.warns(UserWarning) as recwarn:
        assert project_folder / "results" / "test_run_0001" == project.get_result_path("test")

        assert len(recwarn) == 1
        assert Path(recwarn[0].filename).samefile(__file__)
        assert recwarn[0].message.args[0] == expected_warning_text

    with pytest.warns(UserWarning) as recwarn:
        assert isinstance(project.load_result("test"), Result)

        assert len(recwarn) == 1
        assert Path(recwarn[0].filename).samefile(__file__)
        assert recwarn[0].message.args[0] == expected_warning_text


def test_getting_items(project_file: Path):
    """Warn when using fallback to latest result."""
    project = Project.open(project_file)

    assert "dataset_1" in project.data
    assert project.data["dataset_1"].is_file()

    assert "test_model" in project.models
    assert project.models["test_model"].is_file()

    assert "test_parameters" in project.parameters
    assert project.parameters["test_parameters"].is_file()

    assert "test_run_0000" in project.results
    assert project.results["test_run_0000"].is_dir()
    assert "test_run_0001" in project.results
    assert project.results["test_run_0001"].is_dir()


def test_generators_allow_overwrite(project_folder: Path, project_file: Path):
    """Overwrite doesn't throw an exception.

    This is the last test not to interfere with other tests.
    """
    project = Project.open(project_file)

    model_file = project_folder / "models/test_model.yml"
    assert model_file.is_file()

    parameter_file = project_folder / "parameters/test_parameters.csv"
    assert parameter_file.is_file()

    parameters = load_parameters(parameter_file)

    assert len(list(filter(lambda p: p.label.startswith("rates"), parameters.all()))) == 5

    project.generate_model(
        "test_model", "decay_parallel", {"nr_compartments": 3}, allow_overwrite=True
    )
    new_model = project.load_model("test_model")
    assert "megacomplex_parallel_decay" in new_model.megacomplex

    comapartments = load_dict(model_file, is_file=True)["megacomplex"][
        "megacomplex_parallel_decay"
    ]["compartments"]

    assert len(comapartments) == 3

    project.generate_parameters("test_model", "test_parameters", allow_overwrite=True)
    parameters = load_parameters(parameter_file)

    assert len(list(filter(lambda p: p.label.startswith("rates"), parameters.all()))) == 3


def test_import_data_relative_paths_script_folder_not_cwd(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    """Import data using relative paths where cwd and script folder differ."""
    script_folder = tmp_path / "project"
    script_folder.mkdir(parents=True, exist_ok=True)
    with chdir_context(tmp_path), monkeypatch.context() as m:
        # Pretend the currently running script is located at ``script_folder / "script.py"``
        m.setattr(
            sys.modules[globals()["__name__"]],
            "__file__",
            (script_folder / "script.py").as_posix(),
        )
        cwd_data_import_path = "import_data.nc"
        save_dataset(example_dataset, cwd_data_import_path)
        assert (tmp_path / cwd_data_import_path).is_file() is True

        script_folder_data_import_path = "original/import_data.nc"
        save_dataset(example_dataset, script_folder / script_folder_data_import_path)
        assert (script_folder / script_folder_data_import_path).is_file() is True

        project = Project.open("project.gta")
        assert (script_folder / "project.gta").is_file() is True

        project.import_data(f"../{cwd_data_import_path}", name="dataset_1")
        assert (script_folder / "data/dataset_1.nc").is_file() is True

        project.import_data(script_folder_data_import_path, name="dataset_2")
        assert (script_folder / "data/dataset_2.nc").is_file() is True


@pytest.mark.parametrize("file_extension", ("ascii", "nc", "sdt"))
def test_data_plugin_system_integration(tmp_path: Path, file_extension: str):
    """Find data file for all builtin plugins that support ``load_dataset``."""
    data_file = tmp_path / f"data/test_data.{file_extension}"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.touch()
    project = Project.open(tmp_path)

    assert len(project.data) == 1
    assert project.data["test_data"].samefile(data_file)


@pytest.mark.parametrize("file_extension", ("yml", "yaml"))
def test_model_plugin_system_integration(tmp_path: Path, file_extension: str):
    """Find model file for all builtin plugins that support ``load_model``."""
    model_file = tmp_path / f"models/test_model.{file_extension}"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.touch()
    project = Project.open(tmp_path)

    assert len(project.models) == 1
    assert project.models["test_model"].samefile(model_file)


@pytest.mark.parametrize("file_extension", ("yml", "yaml", "ods", "tsv", "xlsx", "ods"))
def test_parameters_plugin_system_integration(tmp_path: Path, file_extension: str):
    """Find parameters file for all builtin plugins that support ``load_parameters``."""
    parameter_file = tmp_path / f"parameters/test_parameter.{file_extension}"
    parameter_file.parent.mkdir(parents=True, exist_ok=True)
    parameter_file.touch()
    project = Project.open(tmp_path)

    assert len(project.parameters) == 1
    assert project.parameters["test_parameter"].samefile(parameter_file)


def test_data_subfolder_folders(tmp_path: Path):
    """Models in sub folders are found."""
    data_file = tmp_path / "data/subfolder/test_data.ascii"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.touch()
    project = Project.open(tmp_path)

    assert len(project.data) == 1
    assert project.data["subfolder/test_data"].samefile(data_file)

    data_file2 = tmp_path / "data/subfolder/test_data.nc"
    data_file2.touch()

    with pytest.warns(AmbiguousNameWarning) as records:
        assert len(project.data) == 2
        assert project.data["subfolder/test_data"].samefile(data_file)
        assert project.data["subfolder/test_data.nc"].samefile(data_file2)
        # One warning pre accessing project.models
        assert len(records) == 3, [r.message for r in records]
        for record in records:
            assert Path(record.filename) == Path(__file__)
            assert str(record.message) == (
                "The Dataset name 'subfolder/test_data' is ambiguous since it could "
                "refer to the following files: ['data/subfolder/test_data.ascii', "
                "'data/subfolder/test_data.nc']\n"
                "The file 'data/subfolder/test_data.nc' will be accessible by the name "
                "'subfolder/test_data.nc'. \n"
                "While 'subfolder/test_data' refers to the file "
                "'data/subfolder/test_data.ascii'.\n"
                "Rename the files with unambiguous names to silence this warning."
            )


def test_model_subfolder_folders(tmp_path: Path):
    """Models in sub folders are found."""
    model_file = tmp_path / "models/subfolder/test_model.yaml"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.touch()
    project = Project.open(tmp_path)

    assert len(project.models) == 1
    assert project.models["subfolder/test_model"].samefile(model_file)

    model_file2 = tmp_path / "models/subfolder/test_model.yml"
    model_file2.touch()

    with pytest.warns(AmbiguousNameWarning) as records:
        assert len(project.models) == 2
        assert project.models["subfolder/test_model"].samefile(model_file)
        assert project.models["subfolder/test_model.yml"].samefile(model_file2)
        # One warning pre accessing project.models
        assert len(records) == 3, [r.message for r in records]
        for record in records:
            assert Path(record.filename) == Path(__file__)
            assert str(record.message) == (
                "The Model name 'subfolder/test_model' is ambiguous since it could "
                "refer to the following files: ['models/subfolder/test_model.yaml', "
                "'models/subfolder/test_model.yml']\n"
                "The file 'models/subfolder/test_model.yml' will be accessible by the name "
                "'subfolder/test_model.yml'. \n"
                "While 'subfolder/test_model' refers to the file "
                "'models/subfolder/test_model.yaml'.\n"
                "Rename the files with unambiguous names to silence this warning."
            )


def test_parameters_subfolder_folders(tmp_path: Path):
    """Parameters in sub folders are found."""
    parameter_file = tmp_path / "parameters/subfolder/test_parameter.yaml"
    parameter_file.parent.mkdir(parents=True, exist_ok=True)
    parameter_file.touch()
    project = Project.open(tmp_path)

    assert len(project.parameters) == 1
    assert project.parameters["subfolder/test_parameter"].samefile(parameter_file)

    parameter_file2 = tmp_path / "parameters/subfolder/test_parameter.yml"
    parameter_file2.touch()

    with pytest.warns(AmbiguousNameWarning) as records:
        assert len(project.parameters) == 2
        assert project.parameters["subfolder/test_parameter"].samefile(parameter_file)
        assert project.parameters["subfolder/test_parameter.yml"].samefile(parameter_file2)
        # One warning pre accessing project.parameters
        assert len(records) == 3, [r.message for r in records]
        for record in records:
            assert Path(record.filename) == Path(__file__)
            assert str(record.message) == (
                "The Parameters name 'subfolder/test_parameter' is ambiguous since it could "
                "refer to the following files: ['parameters/subfolder/test_parameter.yaml', "
                "'parameters/subfolder/test_parameter.yml']\n"
                "The file 'parameters/subfolder/test_parameter.yml' will be accessible by the "
                "name 'subfolder/test_parameter.yml'. \n"
                "While 'subfolder/test_parameter' refers to the file "
                "'parameters/subfolder/test_parameter.yaml'.\n"
                "Rename the files with unambiguous names to silence this warning."
            )


def test_missing_file_errors(tmp_path: Path, project_folder: Path):
    """Error when accessing non existing files."""
    with pytest.raises(FileNotFoundError) as exc_info:
        Project.open(tmp_path, create_if_not_exist=False)

    assert (
        str(exc_info.value)
        == f"Project file {(tmp_path/'project.gta').as_posix()} does not exists."
    )

    project = Project.open(project_folder)

    with pytest.raises(ValueError) as exc_info:
        project.load_data("not-existing")

    assert str(exc_info.value) == (
        "Dataset 'not-existing' does not exist. "
        "Known Datasets are: ['dataset_1', 'dataset_1-bad', 'test_data']"
    )

    with pytest.raises(ValueError) as exc_info:
        project.load_model("not-existing")

    assert str(exc_info.value) == (
        "Model 'not-existing' does not exist. "
        "Known Models are: ['test_model', 'test_model-bad']"
    )

    with pytest.raises(ValueError) as exc_info:
        project.load_parameters("not-existing")

    assert str(exc_info.value) == (
        "Parameters 'not-existing' does not exist. "
        "Known Parameters are: ['test_parameters', 'test_parameters-bad']"
    )

    with pytest.raises(ValueError) as exc_info:
        project.load_result("not-existing_run_0000")

    expected_known_results = (
        "Known Results are: "
        "['sequential_run_0000', 'sequential_run_0001', 'test_run_0000', 'test_run_0001']"
    )

    assert str(exc_info.value) == (
        f"Result 'not-existing_run_0000' does not exist. {expected_known_results}"
    )

    with pytest.raises(ValueError) as exc_info:
        project.load_latest_result("not-existing")

    assert str(exc_info.value) == (
        f"Result 'not-existing' does not exist. {expected_known_results}"
    )

    with pytest.raises(ValueError) as exc_info:
        project.get_result_path("not-existing_run_0000")

    assert str(exc_info.value) == (
        f"Result 'not-existing_run_0000' does not exist. {expected_known_results}"
    )

    with pytest.raises(ValueError) as exc_info:
        project.get_latest_result_path("not-existing")

    assert str(exc_info.value) == (
        f"Result 'not-existing' does not exist. {expected_known_results}"
    )


def test_markdown_repr(project_folder: Path, project_file: Path):
    """calling markdown directly and via ipython."""
    project = Project.open(project_file)

    expected = f"""\
        # Project _{project_folder.as_posix()}_

        pyglotaran version: {distribution('pyglotaran').version}

        ## Data

        * dataset_1
        * dataset_1-bad
        * test_data


        ## Model

        * test_model
        * test_model-bad


        ## Parameters

        * test_parameters
        * test_parameters-bad


        ## Results

        * sequential_run_0000
        * sequential_run_0001
        * test_run_0000
        * test_run_0001

        """

    assert str(project.markdown()) == dedent(expected)

    rendered_result = format_display_data(project)[0]

    assert "text/markdown" in rendered_result
    assert rendered_result["text/markdown"] == dedent(expected)
