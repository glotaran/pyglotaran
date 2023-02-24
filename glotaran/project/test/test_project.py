from __future__ import annotations

import sys
from importlib.metadata import distribution
from pathlib import Path
from shutil import copytree
from textwrap import dedent
from typing import Literal

import pytest
import xarray as xr
from _pytest.monkeypatch import MonkeyPatch
from _pytest.recwarn import WarningsRecorder
from IPython.core.formatters import format_display_data
from pandas.testing import assert_frame_equal

from glotaran import __version__ as gta_version
from glotaran.builtin.io.yml.utils import load_dict
from glotaran.io import load_dataset
from glotaran.io import load_parameters
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.project.project import Project
from glotaran.project.project_registry import AmbiguousNameWarning
from glotaran.project.project_registry import ItemMapping
from glotaran.project.result import Result
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET as example_dataset
from glotaran.testing.simulated_data.sequential_spectral_decay import (
    MODEL_YML as example_model_yml,
)
from glotaran.testing.simulated_data.sequential_spectral_decay import (
    PARAMETERS as example_parameter,
)
from glotaran.typing.types import LoadableDataset
from glotaran.utils.io import chdir_context


def create_populated_project(root_path: Path) -> Project:
    """Helper function to create"""
    project_folder = root_path / "test_project"
    project = Project.open(project_folder)

    input_data = project_folder / "data/dataset_1.nc"
    save_dataset(example_dataset, input_data)

    model_file = project_folder / "models/test_model.yml"
    model_file.write_text(example_model_yml)

    parameters_file = project_folder / "parameters/test_parameters.csv"
    save_parameters(example_parameter, parameters_file, allow_overwrite=True)

    # Will cause tests to fails on bad fuzzy matching due to higher string sort order
    (project_folder / "data/dataset_1-bad.nc").touch()
    (project_folder / "models/test_model-bad.yml").touch()
    (project_folder / "parameters/test_parameters-bad.yml").touch()
    return project


@pytest.fixture
def existing_project(tmp_path: Path):
    return create_populated_project(tmp_path)


@pytest.fixture(scope="session")
def persistent_result(tmp_path_factory: pytest.TempPathFactory):
    """Create a dummy result once per session."""
    root_path = tmp_path_factory.mktemp("tmp_project")
    project = create_populated_project(root_path)
    project.optimize(
        model_name="test_model",
        parameters_name="test_parameters",
        maximum_number_function_evaluations=1,
        result_name="tmp",
    )
    return root_path / "test_project/results/tmp_run_0000"


@pytest.fixture
def dummy_results(persistent_result: Path, existing_project: Project):
    """Create same results as if ``test_run_optimization`` was run with persistence."""
    for dummy_result_folder in (
        "sequential_run_0000",
        "sequential_run_0001",
        "test_run_0000",
        "test_run_0001",
    ):
        copytree(persistent_result, existing_project.folder / f"results/{dummy_result_folder}")


def test_item_mapping():
    """Test all protocol methods work as expected"""
    data = {"foo": Path("foo"), "bar": Path("bar")}
    item_mapping = ItemMapping(data, "test")

    assert len(item_mapping) == 2

    assert repr(item_mapping) == repr({"bar": Path("bar"), "foo": Path("foo")})

    assert item_mapping["foo"] == Path("foo")

    assert item_mapping == ItemMapping(data, "test")
    assert item_mapping == ItemMapping(data, "test2")
    assert item_mapping == data

    assert "foo" in item_mapping

    assert dict(item_mapping.items()) == data

    with pytest.raises(ValueError) as exc_info:
        item_mapping["baz"]

    assert str(exc_info.value) == ("test 'baz' does not exist. Known tests are: ['bar', 'foo']")


def test_init(tmp_path: Path):
    """Init project directly."""
    file_only_init = Project(tmp_path / "project.gta")
    file_and_folder_init = Project(tmp_path / "project.gta")

    assert file_only_init == file_and_folder_init


def test_create(tmp_path: Path):
    Project.create(tmp_path)
    with pytest.raises(FileExistsError):
        assert Project.create(tmp_path)


def test_open(tmp_path: Path):
    project_from_folder = Project.open(tmp_path)

    project_from_file = Project.open(tmp_path / "project.gta")

    assert project_from_folder == project_from_file

    project = project_from_file

    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "parameters").is_dir()

    assert project.version == gta_version
    assert not project.has_models
    assert not project.has_data
    assert not project.has_parameters
    assert not project.has_results


def test_open_diff_version(tmp_path: Path):
    """Loading from file overwrites current version."""
    project_file = tmp_path / "project.gta"
    project_file.write_text("version: 0.1.0")

    project = Project.open(project_file)

    assert project.version == "0.1.0"


@pytest.mark.filterwarnings(
    "ignore::glotaran.deprecation.deprecation_utils.GlotaranApiDeprecationWarning"
)
def test_generate_model(tmp_path: Path):
    project = Project.open(tmp_path / "test_project")

    project.generate_model("generated_test_model", "decay_parallel", {"nr_compartments": 5})

    model_folder = tmp_path / "test_project/models"
    assert model_folder.is_dir()

    project.generate_model(
        "generated_test_model", "decay_parallel", {"nr_compartments": 5}, ignore_existing=True
    )

    assert project.get_models_directory() == model_folder

    model_file = model_folder / "generated_test_model.yml"
    assert model_file.is_file()

    assert project.has_models

    model = project.load_model("generated_test_model")
    assert "megacomplex_parallel_decay" in model.megacomplex

    compartments = load_dict(model_file, is_file=True)["megacomplex"][
        "megacomplex_parallel_decay"
    ]["compartments"]

    assert len(compartments) == 5

    with pytest.raises(FileExistsError) as exc_info:
        project.generate_model("generated_test_model", "decay_parallel", {"nr_compartments": 5})

    assert (
        str(exc_info.value)
        == "Model 'generated_test_model' already exists and `allow_overwrite=False`."
    )


@pytest.mark.filterwarnings(
    "ignore::glotaran.deprecation.deprecation_utils.GlotaranApiDeprecationWarning"
)
@pytest.mark.parametrize("name", ["test_parameter", None])
@pytest.mark.parametrize("fmt", ["yml", "yaml", "csv"])
def test_generate_parameters(tmp_path: Path, name: str | None, fmt: Literal["yml", "yaml", "csv"]):
    project = Project.open(tmp_path / "test_project")

    project.generate_model("generated_test_model", "decay_parallel", {"nr_compartments": 5})

    assert project.has_models

    project.generate_parameters("generated_test_model", parameters_name=name, format_name=fmt)

    parameter_folder = tmp_path / "test_project/parameters"
    assert parameter_folder.is_dir()

    project.generate_parameters(
        "generated_test_model", parameters_name=name, format_name=fmt, ignore_existing=True
    )

    parameter_file_name = f"{'generated_test_model_parameters' if name is None else name}.{fmt}"
    parameter_file = parameter_folder / parameter_file_name
    assert parameter_file.is_file()
    assert project.get_parameters_directory() == parameter_folder

    assert project.has_parameters

    parameters_key = "generated_test_model_parameters" if name is None else name
    model = project.load_model("generated_test_model")
    parameters = project.load_parameters(parameters_key)

    for parameter in model.get_parameter_labels():
        assert parameters.has(parameter)

    assert len(list(filter(lambda p: p.label.startswith("rates"), parameters.all()))) == 5

    with pytest.raises(FileExistsError) as exc_info:
        project.generate_parameters("generated_test_model", parameters_name=name, format_name=fmt)

    assert (
        str(exc_info.value)
        == f"Parameters '{parameters_key}' already exists and `allow_overwrite=False`."
    )


@pytest.mark.parametrize("name", ["test_data", None])
def test_import_data(tmp_path: Path, name: str | None):
    project = Project.open(tmp_path / "test_project")

    test_data = tmp_path / "import_data.nc"
    save_dataset(example_dataset, test_data)

    project.import_data(test_data, dataset_name=name)
    project.import_data(test_data, dataset_name=name)

    with pytest.raises(FileExistsError):
        project.import_data(test_data, dataset_name=name, ignore_existing=False)

    data_folder = tmp_path / "test_project/data"

    data_file_name = f"{'import_data' if name is None else name}.nc"
    data_file = data_folder / data_file_name
    assert data_file.exists()

    assert project.has_data

    data = project.load_data("import_data" if name is None else name)
    assert data.equals(example_dataset)

    data_with_svd = project.load_data("import_data" if name is None else name, add_svd=True)
    assert "data_left_singular_vectors" in data_with_svd
    assert "data_singular_values" in data_with_svd
    assert "data_right_singular_vectors" in data_with_svd


@pytest.mark.parametrize(
    "data",
    (
        xr.DataArray([1]),
        xr.Dataset({"data": xr.DataArray([1])}),
    ),
)
def test_import_data_xarray(tmp_path: Path, data: xr.Dataset | xr.DataArray):
    """Loaded data are always a dataset."""
    project = Project.open(tmp_path)
    project.import_data(data, dataset_name="test_data")

    assert (tmp_path / "data/test_data.nc").is_file() is True

    assert project.load_data("test_data").equals(xr.Dataset({"data": xr.DataArray([1])}))


def test_import_data_allow_overwrite(existing_project: Project):
    """Overwrite data when ``allow_overwrite==True``."""

    dummy_data = xr.Dataset({"data": xr.DataArray([1])})

    assert not existing_project.load_data("dataset_1").equals(dummy_data)

    existing_project.import_data(dummy_data, dataset_name="dataset_1", allow_overwrite=True)

    assert existing_project.load_data("dataset_1").equals(dummy_data)


@pytest.mark.parametrize(
    "data",
    (
        xr.DataArray([1]),
        xr.Dataset({"data": xr.DataArray([1])}),
    ),
)
def test_import_data_mapping(tmp_path: Path, data: xr.Dataset | xr.DataArray):
    """Import data as a mapping"""
    project = Project.open(tmp_path)

    test_data = tmp_path / "import_data.nc"
    save_dataset(example_dataset, test_data)

    project.import_data({"test_data_1": data, "test_data_2": test_data})

    assert (tmp_path / "data/test_data_1.nc").is_file() is True
    assert (tmp_path / "data/test_data_2.nc").is_file() is True

    assert project.load_data("test_data_1").equals(xr.Dataset({"data": xr.DataArray([1])}))
    assert project.load_data("test_data_2").equals(load_dataset(test_data))


def test_create_scheme(existing_project: Project):
    scheme = existing_project.create_scheme(
        model_name="test_model",
        parameters_name="test_parameters",
        maximum_number_function_evaluations=1,
    )

    assert "dataset_1" in scheme.data
    assert "dataset_1" in scheme.model.dataset
    assert scheme.maximum_number_function_evaluations == 1


@pytest.mark.parametrize(
    "data",
    (xr.DataArray([1]), xr.Dataset({"data": xr.DataArray([1])}), "file"),
)
def test_create_scheme_data_lookup_override(
    tmp_path: Path, existing_project: Project, data: LoadableDataset
):
    """Test data_lookup_override functionality."""

    if data == "file":
        data = tmp_path / "file_data.nc"
        save_dataset(xr.Dataset({"data": xr.DataArray([1])}), data)

    scheme = existing_project.create_scheme(
        model_name="test_model",
        parameters_name="test_parameters",
        data_lookup_override={"dataset_1": data},
    )

    assert len(scheme.data) == 1
    assert "dataset_1" in scheme.data
    assert "dataset_1" in scheme.model.dataset
    assert scheme.data["dataset_1"].equals(xr.Dataset({"data": xr.DataArray([1])}))


@pytest.mark.parametrize("result_name", ["test", None])
def test_run_optimization(existing_project: Project, result_name: str | None):
    assert existing_project.has_models
    assert existing_project.has_parameters
    assert existing_project.has_data

    result_name = result_name or "sequential"

    for i in range(2):
        result = existing_project.optimize(
            model_name="test_model",
            parameters_name="test_parameters",
            maximum_number_function_evaluations=1,
            result_name=result_name,
        )
        assert isinstance(result, Result)
        assert existing_project.has_results
        assert (existing_project.folder / f"results/{result_name}_run_000{i}").exists()


@pytest.mark.usefixtures("dummy_results")
def test_load_result(existing_project: Project, recwarn: WarningsRecorder):
    """No warnings if name contains run specifier or latest is true."""

    assert existing_project.folder / "results/test_run_0000" == existing_project.get_result_path(
        "test_run_0000"
    )

    result = existing_project.load_result("test_run_0000")
    assert isinstance(result, Result)

    assert existing_project.folder / "results/test_run_0001" == existing_project.get_result_path(
        "test", latest=True
    )

    assert isinstance(existing_project.load_result("test", latest=True), Result)

    assert (
        existing_project.folder / "results/test_run_0001"
        == existing_project.get_latest_result_path("test")
    )

    assert isinstance(existing_project.load_latest_result("test"), Result)

    assert len(recwarn) == 0


@pytest.mark.usefixtures("dummy_results")
def test_load_result_warnings(existing_project: Project):
    """Warn when using fallback to latest result."""

    expected_warning_text = (
        "Result name 'test' is missing the run specifier, "
        "falling back to try getting latest result. "
        "Use latest=True to mute this warning."
    )

    with pytest.warns(UserWarning) as recwarn:
        assert (
            existing_project.folder / "results/test_run_0001"
            == existing_project.get_result_path("test")
        )

        assert len(recwarn) == 1
        assert Path(recwarn[0].filename).samefile(__file__)
        assert recwarn[0].message.args[0] == expected_warning_text

    with pytest.warns(UserWarning) as recwarn:
        assert isinstance(existing_project.load_result("test"), Result)

        assert len(recwarn) == 1
        assert Path(recwarn[0].filename).samefile(__file__)
        assert recwarn[0].message.args[0] == expected_warning_text


@pytest.mark.usefixtures("dummy_results")
def test_getting_items(existing_project: Project):
    """Warn when using fallback to latest result."""

    assert "dataset_1" in existing_project.data
    assert existing_project.data["dataset_1"].is_file()

    assert "test_model" in existing_project.models
    assert existing_project.models["test_model"].is_file()

    assert "test_parameters" in existing_project.parameters
    assert existing_project.parameters["test_parameters"].is_file()

    assert "test_run_0000" in existing_project.results
    assert existing_project.results["test_run_0000"].is_dir()
    assert "test_run_0001" in existing_project.results
    assert existing_project.results["test_run_0001"].is_dir()


def test_validate(existing_project: Project):
    """Validation works"""

    assert str(existing_project.validate("test_model")) == "Your model is valid."
    assert (
        str(existing_project.validate("test_model", "test_parameters")) == "Your model is valid."
    )

    bad_parameters_path = existing_project.folder / "parameters/bad_parameters.yml"
    bad_parameters_path.write_text("pure_list: [1.0]")
    assert str(existing_project.validate("test_model", "bad_parameters")).startswith(
        "Your model has 5 problems"
    )
    bad_parameters_path.unlink()


def test_show_model_definition(tmp_path: Path):
    """Syntax is correct inferred and expected file content."""
    project = Project.open(tmp_path)
    file_content = "foo:\n  bar"

    (tmp_path / "models/yml_model.yml").write_text(file_content)

    assert str(project.show_model_definition("yml_model")) == "```yaml\nfoo:\n  bar\n```"


def test_show_parameters_definition(tmp_path: Path):
    """Syntax is correct inferred and expected file content and dataframes are returned when
    ``as_dataframe=True`` or file is excel format."""
    project = Project.open(tmp_path)
    file_content = "pure_list: [1.0]"

    (tmp_path / "parameters/yml_parameters.yml").write_text(file_content)

    assert (
        str(project.show_parameters_definition("yml_parameters"))
        == "```yaml\npure_list: [1.0]\n```"
    )

    dummy_parameters = load_parameters("pure_list: [1.0]", format_name="yml_str")

    assert_frame_equal(
        project.show_parameters_definition("yml_parameters", as_dataframe=True),
        dummy_parameters.to_dataframe(),
    )

    save_parameters(dummy_parameters, (tmp_path / "parameters/excel_parameters.xlsx"))

    assert_frame_equal(
        project.show_parameters_definition("excel_parameters"),
        dummy_parameters.to_dataframe(),
    )


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

        project.import_data(f"../{cwd_data_import_path}", dataset_name="dataset_1")
        assert (script_folder / "data/dataset_1.nc").is_file() is True

        project.import_data(script_folder_data_import_path, dataset_name="dataset_2")
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


@pytest.mark.filterwarnings(
    "ignore::glotaran.deprecation.deprecation_utils.GlotaranApiDeprecationWarning"
)
def test_generators_allow_overwrite(existing_project: Project):
    """Overwrite doesn't throw an exception.
    This is the last test not to interfere with other tests.
    """

    model_file = existing_project.folder / "models/test_model.yml"
    assert model_file.is_file()

    parameter_file = existing_project.folder / "parameters/test_parameters.csv"
    assert parameter_file.is_file()

    parameters = load_parameters(parameter_file)

    assert len(list(filter(lambda p: p.label.startswith("rates"), parameters.all()))) == 3

    existing_project.generate_model(
        "test_model", "decay_parallel", {"nr_compartments": 5}, allow_overwrite=True
    )
    new_model = existing_project.load_model("test_model")
    assert "megacomplex_parallel_decay" in new_model.megacomplex

    compartments = load_dict(model_file, is_file=True)["megacomplex"][
        "megacomplex_parallel_decay"
    ]["compartments"]

    assert len(compartments) == 5

    existing_project.generate_parameters("test_model", "test_parameters", allow_overwrite=True)
    parameters = load_parameters(parameter_file)

    assert len(list(filter(lambda p: p.label.startswith("rates"), parameters.all()))) == 5


@pytest.mark.usefixtures("dummy_results")
def test_missing_file_errors(tmp_path: Path, existing_project: Project):
    """Error when accessing non existing files."""
    with pytest.raises(FileNotFoundError) as exc_info:
        Project.open(tmp_path, create_if_not_exist=False)

    assert (
        str(exc_info.value)
        == f"Project file {(tmp_path/'project.gta').as_posix()} does not exists."
    )

    with pytest.raises(ValueError) as exc_info:
        existing_project.import_data(xr.Dataset({"data": [1]}))

    assert str(exc_info.value) == (
        "When importing data from a 'xarray.Dataset' or 'xarray.DataArray' "
        "it is required to also pass a ``dataset_name``."
    )

    with pytest.raises(ValueError) as exc_info:
        existing_project.import_data(xr.DataArray([1]))

    assert str(exc_info.value) == (
        "When importing data from a 'xarray.Dataset' or 'xarray.DataArray' "
        "it is required to also pass a ``dataset_name``."
    )

    no_exist_data_error_msg = (
        "Dataset 'not-existing' does not exist. "
        "Known Datasets are: ['dataset_1', 'dataset_1-bad']"
    )

    with pytest.raises(ValueError) as exc_info:
        existing_project.data["not-existing"]

    assert str(exc_info.value) == no_exist_data_error_msg

    with pytest.raises(ValueError) as exc_info:
        existing_project.load_data("not-existing")

    assert str(exc_info.value) == no_exist_data_error_msg

    no_exist_model_error_msg = (
        "Model 'not-existing' does not exist. "
        "Known Models are: ['test_model', 'test_model-bad']"
    )

    with pytest.raises(ValueError) as exc_info:
        existing_project.models["not-existing"]

    assert str(exc_info.value) == no_exist_model_error_msg

    with pytest.raises(ValueError) as exc_info:
        existing_project.load_model("not-existing")

    assert str(exc_info.value) == no_exist_model_error_msg

    no_exist_parameters_error_msg = (
        "Parameters 'not-existing' does not exist. "
        "Known Parameters are: ['test_parameters', 'test_parameters-bad']"
    )

    with pytest.raises(ValueError) as exc_info:
        existing_project.parameters["not-existing"]

    assert str(exc_info.value) == no_exist_parameters_error_msg

    with pytest.raises(ValueError) as exc_info:
        existing_project.load_parameters("not-existing")

    assert str(exc_info.value) == no_exist_parameters_error_msg

    expected_known_results = (
        "Known Results are: "
        "['sequential_run_0000', 'sequential_run_0001', 'test_run_0000', 'test_run_0001']"
    )

    no_exist_full_result_name_error_msg = (
        f"Result 'not-existing_run_0000' does not exist. {expected_known_results}"
    )

    with pytest.raises(ValueError) as exc_info:
        existing_project.results["not-existing_run_0000"]

    assert str(exc_info.value) == no_exist_full_result_name_error_msg

    with pytest.raises(ValueError) as exc_info:
        existing_project.load_result("not-existing_run_0000")

    assert str(exc_info.value) == no_exist_full_result_name_error_msg

    with pytest.raises(ValueError) as exc_info:
        existing_project.load_latest_result("not-existing")

    assert str(exc_info.value) == (
        f"Result 'not-existing' does not exist. {expected_known_results}"
    )

    with pytest.raises(ValueError) as exc_info:
        existing_project.get_result_path("not-existing_run_0000")

    assert str(exc_info.value) == (
        f"Result 'not-existing_run_0000' does not exist. {expected_known_results}"
    )

    with pytest.raises(ValueError) as exc_info:
        existing_project.get_latest_result_path("not-existing")

    assert str(exc_info.value) == (
        f"Result 'not-existing' does not exist. {expected_known_results}"
    )


@pytest.mark.usefixtures("dummy_results")
def test_markdown_repr(existing_project: Project):
    """calling markdown directly and via ipython."""

    expected = f"""\
        # Project (_test_project_)

        pyglotaran version: `{distribution('pyglotaran').version}`

        ## Data

        * dataset_1
        * dataset_1-bad


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

    assert str(existing_project.markdown()) == dedent(expected)

    rendered_result = format_display_data(existing_project)[0]

    assert "text/markdown" in rendered_result
    assert rendered_result["text/markdown"] == dedent(expected)
