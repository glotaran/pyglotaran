from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from glotaran.builtin.io.csv.csv import CsvProjectIo
from glotaran.builtin.io.yml.yml import YmlProjectIo
from glotaran.io import ProjectIoInterface
from glotaran.parameter import ParameterGroup
from glotaran.plugin_system.base_registry import PluginOverwriteWarning
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.project_io_registration import get_project_io
from glotaran.plugin_system.project_io_registration import get_project_io_method
from glotaran.plugin_system.project_io_registration import is_known_project_format
from glotaran.plugin_system.project_io_registration import known_project_formats
from glotaran.plugin_system.project_io_registration import load_model
from glotaran.plugin_system.project_io_registration import load_parameters
from glotaran.plugin_system.project_io_registration import load_result
from glotaran.plugin_system.project_io_registration import load_scheme
from glotaran.plugin_system.project_io_registration import project_io_plugin_table
from glotaran.plugin_system.project_io_registration import register_project_io
from glotaran.plugin_system.project_io_registration import save_model
from glotaran.plugin_system.project_io_registration import save_parameters
from glotaran.plugin_system.project_io_registration import save_result
from glotaran.plugin_system.project_io_registration import save_scheme
from glotaran.plugin_system.project_io_registration import set_project_plugin
from glotaran.plugin_system.project_io_registration import show_project_io_method_help

if TYPE_CHECKING:
    from os import PathLike
    from typing import Any
    from typing import Callable

    from _pytest.capture import CaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from glotaran.model import Model
    from glotaran.project import Result
    from glotaran.project import Scheme


class MockProjectIo(ProjectIoInterface):
    # TODO: Investigate why write methods raises an [override] type error and load functions don't
    def load_model(self, file_name: str | PathLike[str], **kwargs: Any) -> Model:
        """This docstring is just for help testing of 'load_model'."""
        return {"file_name": file_name, **kwargs}  # type:ignore[return-value]

    def save_model(  # type:ignore[override]
        self,
        model: Model,
        file_name: str | PathLike[str],
        *,
        result_container: dict[str, Any],
        **kwargs: Any,
    ):
        result_container.update(
            **{
                "file_name": file_name,
                "data_object": model,
                **kwargs,
            }
        )

    def load_parameters(self, file_name: str | PathLike[str], **kwargs: Any) -> ParameterGroup:
        return {"file_name": file_name, **kwargs}  # type:ignore[return-value]

    def save_parameters(  # type:ignore[override]
        self,
        parameters: ParameterGroup,
        file_name: str | PathLike[str],
        *,
        result_container: dict[str, Any],
        **kwargs: Any,
    ):
        result_container.update(
            **{
                "file_name": file_name,
                "data_object": parameters,
                **kwargs,
            }
        )

    def load_scheme(self, file_name: str | PathLike[str], **kwargs: Any) -> Scheme:
        return {"file_name": file_name, **kwargs}  # type:ignore[return-value]

    def save_scheme(  # type:ignore[override]
        self,
        scheme: Scheme,
        file_name: str | PathLike[str],
        *,
        result_container: dict[str, Any],
        **kwargs: Any,
    ):
        result_container.update(
            **{
                "file_name": file_name,
                "data_object": scheme,
                **kwargs,
            }
        )

    def load_result(self, result_path: str | PathLike[str], **kwargs: Any) -> Result:
        return {"file_name": result_path, **kwargs}  # type:ignore[return-value]

    def save_result(  # type:ignore[override]
        self,
        result: Result,
        result_path: str | PathLike[str],
        *,
        result_container: dict[str, Any],
        **kwargs: Any,
    ):
        result_container.update(
            **{
                "file_name": result_path,
                "data_object": result,
                **kwargs,
            }
        )


@pytest.fixture
def mocked_registry(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        __PluginRegistry,
        "project_io",
        {
            "foo": ProjectIoInterface("foo"),
            "mock": MockProjectIo("bar"),
            "test_project_io_registration.MockProjectIo_bar": MockProjectIo("bar"),
        },
    )


@pytest.mark.usefixtures("mocked_registry")
def test_register_project_io():
    """Registered project_io plugin is in registry"""

    @register_project_io("dummy")
    class Dummy(ProjectIoInterface):
        pass

    @register_project_io(["dummy2", "dummy3"])
    class Dummy2(ProjectIoInterface):
        pass

    for format_name, plugin_class in [("dummy", Dummy), ("dummy2", Dummy2), ("dummy3", Dummy2)]:
        assert format_name in __PluginRegistry.project_io
        assert isinstance(__PluginRegistry.project_io[format_name], plugin_class)
        assert isinstance(
            __PluginRegistry.project_io[
                f"test_project_io_registration.{plugin_class.__name__}_{format_name}"
            ],
            plugin_class,
        )
        assert __PluginRegistry.project_io[format_name].format == format_name


@pytest.mark.usefixtures("mocked_registry")
def test_register_project_io_warning():
    """PluginOverwriteWarning raised pointing to correct file."""

    with pytest.warns(PluginOverwriteWarning, match="Dummy.+dummy.+Dummy2") as record:

        @register_project_io("dummy")
        class Dummy(ProjectIoInterface):
            pass

        @register_project_io("dummy")
        class Dummy2(ProjectIoInterface):
            pass

        assert len(record) == 1
        assert Path(record[0].filename) == Path(__file__)


@pytest.mark.usefixtures("mocked_registry")
def test_known_project_format():
    """Known format in mocked register"""
    assert is_known_project_format("foo")
    assert is_known_project_format("mock")
    assert not is_known_project_format("baz")


def test_known_project_format_actual_register():
    """Builtins in are in actual register."""
    assert is_known_project_format("yml")
    assert is_known_project_format("yaml")
    assert is_known_project_format("yml_str")
    assert is_known_project_format("csv")
    assert is_known_project_format("glotaran.builtin.io.yml.yml.YmlProjectIo_yml")
    assert is_known_project_format("glotaran.builtin.io.yml.yml.YmlProjectIo_yaml")
    assert is_known_project_format("glotaran.builtin.io.yml.yml.YmlProjectIo_yml_str")
    assert is_known_project_format("glotaran.builtin.io.csv.csv.CsvProjectIo_csv")


@pytest.mark.parametrize(
    "format_name, io_class",
    (
        ("yml", YmlProjectIo),
        ("yaml", YmlProjectIo),
        ("yml_str", YmlProjectIo),
        ("csv", CsvProjectIo),
    ),
)
def test_get_project_io(format_name: str, io_class: type[ProjectIoInterface]):
    """Get the right instance"""
    assert isinstance(get_project_io(format_name), io_class)
    assert get_project_io(format_name).format == format_name


@pytest.mark.usefixtures("mocked_registry")
def test_known_project_formats():
    """Known formats are the same as mocked register keys"""
    assert known_project_formats() == ["foo", "mock"]


@pytest.mark.usefixtures("mocked_registry")
def test_set_project_plugin():
    """Set Change Plugin used for format foo"""
    assert isinstance(get_project_io("foo"), ProjectIoInterface)
    set_project_plugin("foo", "test_project_io_registration.MockProjectIo_bar")
    assert isinstance(get_project_io("foo"), MockProjectIo)


@pytest.mark.parametrize(
    "load_function",
    (
        load_model,
        load_parameters,
        load_scheme,
        load_result,
    ),
)
@pytest.mark.usefixtures("mocked_registry")
def test_load_functions(tmp_path: Path, load_function: Callable[..., Any]):
    """All args and kwargs are passes correctly."""
    file_path = tmp_path / "model.mock"
    file_path.touch()

    result = load_function(file_path, dummy_arg="baz")

    assert result == {"file_name": str(file_path), "dummy_arg": "baz"}


@pytest.mark.parametrize(
    "save_function",
    (
        save_model,
        save_parameters,
        save_scheme,
        save_result,
    ),
)
@pytest.mark.usefixtures("mocked_registry")
def test_write_functions(tmp_path: Path, save_function: Callable[..., Any]):
    """All args and kwargs are passes correctly."""
    file_path = tmp_path / "model.mock"
    result: dict[str, Any] = {}

    save_function(
        "data_object",  # type:ignore
        file_path,
        "mock",
        result_container=result,
        dummy_arg="baz",
    )

    assert result == {
        "file_name": str(file_path),
        "data_object": "data_object",
        "dummy_arg": "baz",
    }


@pytest.mark.parametrize(
    "load_function, error_regex",
    (
        (load_model, "read models"),
        (load_parameters, "read parameters"),
        (load_scheme, "read scheme"),
        (load_result, "read result"),
    ),
)
@pytest.mark.usefixtures("mocked_registry")
def test_load_functions_value_error(
    tmp_path: Path, load_function: Callable[..., Any], error_regex: str
):
    """Raise ValueError if load method isn't implemented."""
    file_path = tmp_path / "dummy.foo"

    with pytest.raises(ValueError, match=f"Cannot {error_regex} with format 'foo'"):
        load_function(file_path, "foo")


@pytest.mark.parametrize(
    "save_function, error_regex",
    (
        (save_model, "save models"),
        (save_parameters, "save parameters"),
        (save_scheme, "save scheme"),
        (save_result, "save result"),
    ),
)
@pytest.mark.usefixtures("mocked_registry")
def test_save_functions_value_error(
    tmp_path: Path, save_function: Callable[..., Any], error_regex: str
):
    """Raise ValueError if save method isn't implemented."""
    file_path = tmp_path / "dummy.foo"

    with pytest.raises(ValueError, match=f"Cannot {error_regex} with format 'foo'"):
        save_function("bar", file_path)


@pytest.mark.parametrize(
    "function",
    (save_model, save_parameters, save_scheme, save_result),
)
@pytest.mark.usefixtures("mocked_registry")
def test_protect_from_overwrite_save_functions(tmp_path: Path, function: Callable[..., Any]):
    """Raise FileExistsError if file exists."""

    file_path = tmp_path / "dummy.foo"
    file_path.touch()

    with pytest.raises(FileExistsError, match="The file .+? already exists"):
        function("foo", file_path, "bar")


@pytest.mark.usefixtures("mocked_registry")
def test_get_project_io_method():
    """Methods have the same code."""
    io = get_project_io("mock")
    result = get_project_io_method("mock", "load_model")

    assert result.__code__ == io.load_model.__code__


@pytest.mark.usefixtures("mocked_registry")
def test_show_project_io_method_help(capsys: CaptureFixture):
    """Same help as when called directly."""
    plugin = MockProjectIo("foo")
    help(plugin.load_model)
    original_help, _ = capsys.readouterr()

    show_project_io_method_help(format_name="mock", method_name="load_model")
    result, _ = capsys.readouterr()

    assert "This docstring is just for help testing of 'load_model'." in result
    assert result == original_help


@pytest.mark.usefixtures("mocked_registry")
def test_project_io_plugin_table():
    """Plugin foo supports no function and mock supports all"""
    expected = dedent(
        """\
        |  __Format name__  |  __load_model__  |  __save_model__  |  __load_parameters__  |  __save_parameters__  |  __load_scheme__  |  __save_scheme__  |  __load_result__  |  __save_result__  |
        |-------------------|------------------|------------------|-----------------------|-----------------------|-------------------|-------------------|-------------------|-------------------|
        |       `foo`       |        /         |        /         |           /           |           /           |         /         |         /         |         /         |         /         |
        |      `mock`       |        *         |        *         |           *           |           *           |         *         |         *         |         *         |         *         |
        """  # noqa: E501
    )

    assert f"{project_io_plugin_table()}\n" == expected


@pytest.mark.usefixtures("mocked_registry")
def test_project_io_plugin_table_full():
    """Full Table with all extras"""
    expected = dedent(
        """\
        |                 __Format name__                  |  __load_model__  |  __save_model__  |  __load_parameters__  |  __save_parameters__  |  __load_scheme__  |  __save_scheme__  |  __load_result__  |  __save_result__  |                  __Plugin name__                  |
        |--------------------------------------------------|------------------|------------------|-----------------------|-----------------------|-------------------|-------------------|-------------------|-------------------|---------------------------------------------------|
        |                      `foo`                       |        /         |        /         |           /           |           /           |         /         |         /         |         /         |         /         |  `glotaran.io.interface.ProjectIoInterface_foo`   |
        |                      `mock`                      |        *         |        *         |           *           |           *           |         *         |         *         |         *         |         *         | `test_project_io_registration.MockProjectIo_mock` |
        | `test_project_io_registration.MockProjectIo_bar` |        *         |        *         |           *           |           *           |         *         |         *         |         *         |         *         | `test_project_io_registration.MockProjectIo_bar`  |
        """  # noqa: E501
    )

    assert f"{project_io_plugin_table(plugin_names=True,full_names=True)}\n" == expected
