from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glotaran.builtin.io.csv.csv import CsvProjectIo
from glotaran.builtin.io.yml.yml import YmlProjectIo
from glotaran.builtin.models.kinetic_spectrum import KineticSpectrumModel
from glotaran.builtin.models.kinetic_spectrum.test.test_kinetic_spectrum_model import MODEL_1C_BASE
from glotaran.builtin.models.kinetic_spectrum.test.test_kinetic_spectrum_model import (
    PARAMETERS_3C_BASE,
)
from glotaran.io import ProjectIoInterface
from glotaran.parameter import ParameterGroup
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.project_io_registration import get_project_io
from glotaran.plugin_system.project_io_registration import is_known_project_format
from glotaran.plugin_system.project_io_registration import known_project_formats
from glotaran.plugin_system.project_io_registration import load_model
from glotaran.plugin_system.project_io_registration import load_parameters
from glotaran.plugin_system.project_io_registration import load_result
from glotaran.plugin_system.project_io_registration import load_scheme
from glotaran.plugin_system.project_io_registration import register_project_io
from glotaran.plugin_system.project_io_registration import write_model
from glotaran.plugin_system.project_io_registration import write_parameters
from glotaran.plugin_system.project_io_registration import write_result
from glotaran.plugin_system.project_io_registration import write_scheme

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any
    from typing import Callable

    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def mocked_registry(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        __PluginRegistry,
        "project_io",
        {"foo": ProjectIoInterface("foo"), "bar": ProjectIoInterface("bar")},
    )


def test_register_project_io(mocked_registry):
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
        assert __PluginRegistry.project_io[format_name].format == format_name


def test_known_project_format(mocked_registry):
    """Known format in mocked register"""
    assert is_known_project_format("foo")
    assert is_known_project_format("bar")
    assert not is_known_project_format("baz")


def test_known_project_format_actual_register():
    """Builtins in are in actual register."""
    assert is_known_project_format("yml")
    assert is_known_project_format("yaml")
    assert is_known_project_format("yml_str")
    assert is_known_project_format("csv")


@pytest.mark.parametrize(
    "format_name, io_class",
    (
        ("yml", YmlProjectIo),
        ("yaml", YmlProjectIo),
        ("yml_str", YmlProjectIo),
        ("csv", CsvProjectIo),
    ),
)
def test_get_data_io(format_name: str, io_class: type[ProjectIoInterface]):
    """Get the right instance"""
    assert isinstance(get_project_io(format_name), io_class)
    assert get_project_io(format_name).format == format_name


def test_known_project_formats(mocked_registry):
    """Known formats are the same as mocked register keys"""
    assert known_project_formats() == ["foo", "bar"]


def test_load_model(tmp_path: Path):
    """Right kind of model from yml definition"""
    model_path = tmp_path / "model.yml"
    model_path.write_text(MODEL_1C_BASE)

    model = load_model(str(model_path))

    assert model.validate() == "Your model is valid."
    assert isinstance(model, KineticSpectrumModel)


def test_load_parameters(tmp_path: Path):
    """Parameters have irf"""
    parameters_path = tmp_path / "parameters.yml"
    parameters_path.write_text(PARAMETERS_3C_BASE)
    parameters = load_parameters(str(parameters_path))
    assert isinstance(parameters, ParameterGroup)
    assert parameters.has("irf.center")
    assert parameters.has("irf.width")


@pytest.mark.parametrize(
    "function, error_regex",
    (
        (load_model, "read models"),
        (load_parameters, "read parameters"),
        (load_scheme, "read scheme"),
        (load_result, "read result"),
    ),
)
def test_value_error_load_functions(
    mocked_registry, tmp_path: Path, function: Callable[..., Any], error_regex: str
):
    """Raise ValueError if load method isn't implemented."""

    file_path = tmp_path / "dummy.foo"

    with pytest.raises(ValueError, match=f"Cannot {error_regex} with format 'foo'"):
        function(str(file_path), "foo")


@pytest.mark.parametrize(
    "function, error_regex",
    (
        (write_model, "write models"),
        (write_parameters, "write parameters"),
        (write_scheme, "write scheme"),
        (write_result, "write result"),
    ),
)
def test_value_error_write_functions(
    mocked_registry, tmp_path: Path, function: Callable[..., Any], error_regex: str
):
    """Raise ValueError if write method isn't implemented."""

    file_path = tmp_path / "dummy.foo"

    with pytest.raises(ValueError, match=f"Cannot {error_regex} with format 'foo'"):
        function(str(file_path), "foo", "bar")


@pytest.mark.parametrize(
    "function, error_regex",
    (
        (write_model, "write models"),
        (write_parameters, "write parameters"),
        (write_scheme, "write scheme"),
        (write_result, "write result"),
    ),
)
def test_protect_from_overwrite_write_functions(
    mocked_registry, tmp_path: Path, function: Callable[..., Any], error_regex: str
):
    """Raise FileExistsError if file exists."""

    file_path = tmp_path / "dummy.foo"
    file_path.touch()

    with pytest.raises(FileExistsError, match="The file .+? already exists"):
        function(str(file_path), "foo", "bar")
