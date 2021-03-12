from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import xarray as xr

from glotaran.builtin.io.ascii.wavelength_time_explicit_file import AsciiDataIo
from glotaran.builtin.io.netCDF.netCDF import NetCDFDataIo
from glotaran.builtin.io.sdt.sdt_file_reader import SdtDataIo
from glotaran.io import DataIoInterface
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.data_io_registration import get_data_io
from glotaran.plugin_system.data_io_registration import get_dataloader
from glotaran.plugin_system.data_io_registration import get_datawriter
from glotaran.plugin_system.data_io_registration import known_data_format
from glotaran.plugin_system.data_io_registration import known_data_formats
from glotaran.plugin_system.data_io_registration import load_dataset
from glotaran.plugin_system.data_io_registration import register_data_io
from glotaran.plugin_system.data_io_registration import write_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


DUMMY_ASCII_CONTENT = """\
# Skip the filename when comparing

Time explicit
Intervalnr 2
0.0\t0.00999999978
4.00000e+02\t9.04130e-02\t-8.30020e-02
4.04000e+02\t-3.17128e-02\t-2.86548e-02
"""

DUMMY_DATASET = xr.DataArray(
    [[9.04130e-02, -3.17128e-02], [-8.30020e-02, -2.86548e-02]],
    coords=[("time", [0, 9.99999978e-03]), ("spectral", [400, 404])],
)


@pytest.fixture
def mocked_registry(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        __PluginRegistry, "data_io", {"foo": DataIoInterface("foo"), "bar": DataIoInterface("bar")}
    )


def test_register_data_io(mocked_registry):
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


def test_known_data_format(mocked_registry):
    """Known format in mocked register"""
    assert known_data_format("foo")
    assert known_data_format("bar")
    assert not known_data_format("baz")


def test_known_data_format_actual_register():
    """Builtins in are in actual register."""
    assert known_data_format("sdt")
    assert known_data_format("ascii")
    assert known_data_format("nc")


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


def test_known_data_formats(mocked_registry):
    """Known formats are the same as mocked register keys"""
    assert known_data_formats() == ["foo", "bar"]


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
    assert dataloader.__code__ == io_class.read_dataset.__code__


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


def test_load_dataset(tmp_path: Path):
    """Used reduced ascii testcase for loading data."""
    file_path = tmp_path / "dummy.ascii"
    file_path.write_text(DUMMY_ASCII_CONTENT)

    result = load_dataset(str(file_path))

    assert result == DUMMY_DATASET


def test_write_dataset(tmp_path: Path):
    """Despite the comments the written content is the same as the original"""
    file_path = tmp_path / "dummy.ascii"

    write_dataset(
        str(file_path),
        "ascii",
        DUMMY_DATASET,
        number_format="%.5e",
    )
    result = file_path.read_text()

    assert result.splitlines()[2:] == DUMMY_ASCII_CONTENT.splitlines()[2:]


def test_write_dataset_error(tmp_path: Path):
    """Raise ValueError if method isn't implemented in the DataIo class."""
    file_path = tmp_path / "dummy.sdt"

    with pytest.raises(ValueError, match="Cannot write data with format: 'sdt'"):
        write_dataset(str(file_path), "sdt", DUMMY_DATASET)
