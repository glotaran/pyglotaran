from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest
import xarray as xr

from glotaran.builtin.io.ascii.wavelength_time_explicit_file import AsciiDataIo
from glotaran.builtin.io.netCDF.netCDF import NetCDFDataIo
from glotaran.builtin.io.sdt.sdt_file_reader import SdtDataIo
from glotaran.io import DataIoInterface
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.data_io_registration import data_io_plugin_table
from glotaran.plugin_system.data_io_registration import get_data_io
from glotaran.plugin_system.data_io_registration import get_dataloader
from glotaran.plugin_system.data_io_registration import get_datawriter
from glotaran.plugin_system.data_io_registration import is_known_data_format
from glotaran.plugin_system.data_io_registration import known_data_formats
from glotaran.plugin_system.data_io_registration import load_dataset
from glotaran.plugin_system.data_io_registration import register_data_io
from glotaran.plugin_system.data_io_registration import show_data_io_method_help
from glotaran.plugin_system.data_io_registration import write_dataset

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from _pytest.capture import CaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from glotaran.project import SavingOptions


class MockDataIO(DataIoInterface):
    def load_dataset(self, file_name: str, **kwargs: Any) -> xr.Dataset | xr.DataArray:
        """This docstring is just for help testing of 'load_dataset'."""
        return {"file_name": file_name, **kwargs}  # type:ignore

    # TODO: Investigate why this raises an [override] type error and read_dataset doesn't
    def write_dataset(  # type:ignore[override]
        self,
        file_name: str,
        dataset: xr.Dataset | xr.DataArray,
        saving_options: SavingOptions | None = None,
        *,
        result_container: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        result_container.update(
            **{
                "file_name": file_name,
                "dataset": dataset,
                "saving_options": saving_options,
                **kwargs,
            }
        )


@pytest.fixture(scope="function")
def mocked_registry(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        __PluginRegistry, "data_io", {"foo": DataIoInterface("foo"), "mock": MockDataIO("bar")}
    )


@pytest.mark.usefixtures("mocked_registry")
def test_register_data_io():
    """Registered data_io plugin is in registry"""

    @register_data_io("dummy")
    class Dummy(DataIoInterface):
        pass

    @register_data_io(["dummy2", "dummy3"])
    class Dummy2(DataIoInterface):
        pass

    for format_name, plugin_class in [("dummy", Dummy), ("dummy2", Dummy2), ("dummy3", Dummy2)]:
        assert format_name in __PluginRegistry.data_io
        assert isinstance(__PluginRegistry.data_io[format_name], plugin_class)
        assert __PluginRegistry.data_io[format_name].format == format_name


@pytest.mark.usefixtures("mocked_registry")
def test_known_data_format():
    """Known format in mocked register"""
    assert is_known_data_format("foo")
    assert is_known_data_format("mock")
    assert not is_known_data_format("baz")


def test_known_data_format_actual_register():
    """Builtins in are in actual register."""
    assert is_known_data_format("sdt")
    assert is_known_data_format("ascii")
    assert is_known_data_format("nc")


@pytest.mark.parametrize(
    "format_name, io_class",
    (
        ("sdt", SdtDataIo),
        ("ascii", AsciiDataIo),
        ("nc", NetCDFDataIo),
    ),
)
def test_get_data_io(format_name: str, io_class: type[DataIoInterface]):
    """Get the right instance"""
    assert isinstance(get_data_io(format_name), io_class)
    assert get_data_io(format_name).format == format_name


@pytest.mark.usefixtures("mocked_registry")
def test_known_data_formats():
    """Known formats are the same as mocked register keys"""
    assert known_data_formats() == ["foo", "mock"]


@pytest.mark.parametrize(
    "format_name, io_class",
    (
        ("sdt", SdtDataIo),
        ("ascii", AsciiDataIo),
        ("nc", NetCDFDataIo),
    ),
)
def test_get_dataloader(format_name: str, io_class: type[DataIoInterface]):
    """Code of the dataloader is the same as original classes method code"""
    dataloader = get_dataloader(format_name)
    assert dataloader.__code__ == io_class.load_dataset.__code__


@pytest.mark.parametrize(
    "format_name, io_class",
    (
        ("sdt", SdtDataIo),
        ("ascii", AsciiDataIo),
        ("nc", NetCDFDataIo),
    ),
)
def test_get_datawriter(format_name: str, io_class: type[DataIoInterface]):
    """Code of the datawriter is the same as original classes method code"""
    datawriter = get_datawriter(format_name)
    assert datawriter.__code__ == io_class.write_dataset.__code__


@pytest.mark.usefixtures("mocked_registry")
def test_load_dataset(tmp_path: Path):
    """All args and kwargs are passes correctly."""
    file_path = tmp_path / "dummy.mock"
    file_path.write_text("mock")

    result = load_dataset(str(file_path), dummy_arg="baz")

    assert result == {"file_name": str(file_path), "dummy_arg": "baz"}


@pytest.mark.usefixtures("mocked_registry")
def test_protect_from_overwrite_write_functions(tmp_path: Path):
    """Raise FileExistsError if file exists."""

    file_path = tmp_path / "dummy.foo"
    file_path.touch()

    with pytest.raises(FileExistsError, match="The file .+? already exists"):
        write_dataset(str(file_path), "foo", "")  # type:ignore


@pytest.mark.usefixtures("mocked_registry")
def test_write_dataset(tmp_path: Path):
    """All args and kwargs are passes correctly."""
    file_path = tmp_path / "dummy.ascii"

    result: dict[str, Any] = {}
    write_dataset(
        str(file_path),
        "mock",
        "no_dataset",  # type:ignore
        "no_option",  # type:ignore
        result_container=result,
        dummy_arg="baz",
    )

    assert result == {
        "file_name": str(file_path),
        "dataset": "no_dataset",
        "saving_options": "no_option",
        "dummy_arg": "baz",
    }


@pytest.mark.usefixtures("mocked_registry")
def test_write_dataset_error(tmp_path: Path):
    """Raise ValueError if method isn't implemented in the DataIo class."""
    file_path = tmp_path / "dummy.foo"

    with pytest.raises(ValueError, match="Cannot write data with format: 'foo'"):
        write_dataset(str(file_path), "foo", "")  # type:ignore

    file_path.touch()

    with pytest.raises(ValueError, match="Cannot read data with format: 'foo'"):
        load_dataset(str(file_path))


@pytest.mark.usefixtures("mocked_registry")
def test_show_data_io_method_help(capsys: CaptureFixture):
    """Same help as when called directly."""
    plugin = MockDataIO("foo")
    help(plugin.load_dataset)
    original_help, _ = capsys.readouterr()

    show_data_io_method_help(format_name="mock", method_name="load_dataset")
    result, _ = capsys.readouterr()

    assert "This docstring is just for help testing of 'load_dataset'." in result
    assert result == original_help


@pytest.mark.usefixtures("mocked_registry")
def test_data_io_plugin_table():
    """Plugin foo supports no function and mock supports all"""
    expected = dedent(
        """\
        |  __Plugin__  |  __load_dataset__  |  __write_dataset__  |
        |--------------|--------------------|---------------------|
        |     foo      |         /          |          /          |
        |     mock     |         *          |          *          |
        """
    )

    assert f"{data_io_plugin_table()}\n" == expected
