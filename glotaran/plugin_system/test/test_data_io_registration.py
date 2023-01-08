from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.io.ascii.wavelength_time_explicit_file import AsciiDataIo
from glotaran.builtin.io.netCDF.netCDF import NetCDFDataIo
from glotaran.builtin.io.sdt.sdt_file_reader import SdtDataIo
from glotaran.io import DataIoInterface
from glotaran.plugin_system.base_registry import PluginOverwriteWarning
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.data_io_registration import data_io_plugin_table
from glotaran.plugin_system.data_io_registration import get_data_io
from glotaran.plugin_system.data_io_registration import get_dataloader
from glotaran.plugin_system.data_io_registration import get_datasaver
from glotaran.plugin_system.data_io_registration import is_known_data_format
from glotaran.plugin_system.data_io_registration import known_data_formats
from glotaran.plugin_system.data_io_registration import load_dataset
from glotaran.plugin_system.data_io_registration import register_data_io
from glotaran.plugin_system.data_io_registration import save_dataset
from glotaran.plugin_system.data_io_registration import set_data_plugin
from glotaran.plugin_system.data_io_registration import show_data_io_method_help
from glotaran.plugin_system.data_io_registration import supported_file_extensions_data_io
from glotaran.testing.plugin_system import monkeypatch_plugin_registry_data_io

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from _pytest.capture import CaptureFixture

    from glotaran.typing import StrOrPath


class MockDataIo(DataIoInterface):
    def load_dataset(  # type:ignore[override]
        self, file_name: StrOrPath, *, result_container: dict[str, Any], **kwargs: Any
    ) -> xr.Dataset | xr.DataArray:
        """This docstring is just for help testing of 'load_dataset'."""
        result_container |= {"file_name": file_name, **kwargs}
        return xr.DataArray([1, 2])

    # TODO: Investigate why this raises an [override] type error and read_dataset doesn't
    def save_dataset(  # type:ignore[override]
        self,
        file_name: StrOrPath,
        dataset: xr.Dataset | xr.DataArray,
        *,
        result_container: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        result_container.update(
            **{
                "file_name": file_name,
                "dataset": dataset,
                **kwargs,
            }
        )


class MockDataIoPartial(DataIoInterface):
    def load_dataset(self, file_name: StrOrPath) -> xr.Dataset | xr.DataArray:
        return xr.DataArray([1, 2])


MOCK_REGISTRY_VALUES = {
    "foo": DataIoInterface("foo"),
    "mock": MockDataIo("bar"),
    "test_data_io_registration.MockDataIo_bar": MockDataIo("bar"),
}


@pytest.fixture(scope="function")
def mocked_registry():
    with monkeypatch_plugin_registry_data_io(MOCK_REGISTRY_VALUES, create_new_registry=True):
        yield


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
        assert isinstance(
            __PluginRegistry.data_io[
                f"test_data_io_registration.{plugin_class.__name__}_{format_name}"
            ],
            plugin_class,
        )


@pytest.mark.usefixtures("mocked_registry")
def test_register_data_io_warning():
    """PluginOverwriteWarning raised pointing to correct file."""

    with pytest.warns(PluginOverwriteWarning, match="Dummy.+dummy.+Dummy2") as record:

        @register_data_io("dummy")
        class Dummy(DataIoInterface):
            pass

        @register_data_io("dummy")
        class Dummy2(DataIoInterface):
            pass

        assert len(record) == 1
        assert Path(record[0].filename) == Path(__file__)


@pytest.mark.usefixtures("mocked_registry")
def test_is_known_data_format():
    """Known format in mocked register"""
    assert is_known_data_format("foo")
    assert is_known_data_format("mock")
    assert not is_known_data_format("baz")


def test_known_data_format_actual_register():
    """Builtins in are in actual register."""
    assert is_known_data_format("sdt")
    assert is_known_data_format("ascii")
    assert is_known_data_format("nc")
    assert is_known_data_format("glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo_sdt")
    assert is_known_data_format(
        "glotaran.builtin.io.ascii.wavelength_time_explicit_file.AsciiDataIo_ascii"
    )
    assert is_known_data_format("glotaran.builtin.io.netCDF.netCDF.NetCDFDataIo_nc")


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
    assert sorted(known_data_formats()) == sorted(["foo", "mock"])


@pytest.mark.usefixtures("mocked_registry")
def test_set_data_plugin():
    """Set Change Plugin used for format foo"""
    assert isinstance(get_data_io("foo"), DataIoInterface)
    set_data_plugin("foo", "test_data_io_registration.MockDataIo_bar")
    assert isinstance(get_data_io("foo"), MockDataIo)


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
    datawriter = get_datasaver(format_name)
    assert datawriter.__code__ == io_class.save_dataset.__code__


@pytest.mark.usefixtures("mocked_registry")
def test_load_dataset(tmp_path: Path):
    """All args and kwargs are passes correctly."""
    file_path = tmp_path / "dummy.mock"
    file_path.write_text("mock")
    result: dict[str, Any] = {}

    dataset = load_dataset(file_path, result_container=result, dummy_arg="baz")

    assert result == {"file_name": file_path.as_posix(), "dummy_arg": "baz"}
    assert np.all(dataset.data == xr.DataArray([1, 2]).to_dataset(name="data").data)
    assert dataset.source_path == file_path.as_posix()


@pytest.mark.usefixtures("mocked_registry")
def test_protect_from_overwrite_write_functions(tmp_path: Path):
    """Raise FileExistsError if file exists."""

    file_path = tmp_path / "dummy.foo"
    file_path.touch()

    with pytest.raises(FileExistsError, match="The file .+? already exists"):
        save_dataset(xr.DataArray([1, 2]), str(file_path))


@pytest.mark.parametrize("sub_dir", ("", "sub_dir"))
@pytest.mark.usefixtures("mocked_registry")
def test_write_dataset(tmp_path: Path, sub_dir: str):
    """All args and kwargs are passes correctly."""
    file_path = tmp_path / sub_dir / "dummy.mock"

    result: dict[str, Any] = {}
    save_dataset(
        xr.DataArray([1, 2]),
        file_path,
        result_container=result,
        dummy_arg="baz",
    )

    assert len(result) == 3
    assert result["file_name"] == file_path.as_posix()
    assert result["dummy_arg"] == "baz"
    assert np.all(result["dataset"] == xr.DataArray([1, 2]))


@pytest.mark.usefixtures("mocked_registry")
def test_write_dataset_error(tmp_path: Path):
    """Raise ValueError if method isn't implemented in the DataIo class."""
    file_path = tmp_path / "dummy.foo"

    with pytest.raises(ValueError, match="Cannot save data with format: 'foo'"):
        save_dataset(xr.DataArray([1, 2]), str(file_path), "foo")

    file_path.touch()

    with pytest.raises(ValueError, match="Cannot read data with format: 'foo'"):
        load_dataset(str(file_path))


@pytest.mark.usefixtures("mocked_registry")
def test_show_data_io_method_help(capsys: CaptureFixture):
    """Same help as when called directly."""
    plugin = MockDataIo("foo")
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
        |  __Format name__  |  __load_dataset__  |  __save_dataset__  |
        |-------------------|--------------------|--------------------|
        |       `foo`       |         /          |         /          |
        |      `mock`       |         *          |         *          |
        """
    )

    assert f"{data_io_plugin_table()}\n" == expected


@pytest.mark.usefixtures("mocked_registry")
def test_data_io_plugin_table_full():
    """Full Table with all extras"""
    expected = dedent(
        """\
        |              __Format name__               |  __load_dataset__  |  __save_dataset__  |               __Plugin name__               |
        |--------------------------------------------|--------------------|--------------------|---------------------------------------------|
        |                   `foo`                    |         /          |         /          | `glotaran.io.interface.DataIoInterface_foo` |
        |                   `mock`                   |         *          |         *          | `test_data_io_registration.MockDataIo_mock` |
        | `test_data_io_registration.MockDataIo_bar` |         *          |         *          | `test_data_io_registration.MockDataIo_bar`  |
        """  # noqa: E501
    )

    assert f"{data_io_plugin_table(plugin_names=True,full_names=True)}\n" == expected


@pytest.mark.parametrize(
    "method_names, expected",
    (
        (
            "load_dataset",
            [".mock", ".mock_partial"],
        ),
        (
            "save_dataset",
            [".mock"],
        ),
        (
            ["load_dataset", "save_dataset"],
            [".mock"],
        ),
    ),
)
def test_supported_file_extensions_data_io(method_names: str | Sequence[str], expected: list[str]):
    """Extension don't list full plugin name and omit extension that don't support all methods."""
    with monkeypatch_plugin_registry_data_io(
        test_data_io={**MOCK_REGISTRY_VALUES, "mock_partial": MockDataIoPartial},
        create_new_registry=True,
    ):
        assert list(supported_file_extensions_data_io(method_names)) == expected
