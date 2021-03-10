from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import MutableMapping
from typing import Type
from typing import cast
from warnings import warn

import pandas as pd
import pytest

from glotaran.builtin.io.sdt.sdt_file_reader import SdtDataIo
from glotaran.builtin.io.yml.yml import YmlProjectIo
from glotaran.builtin.models.kinetic_image import KineticImageModel
from glotaran.io.interface import DataIoInterface
from glotaran.io.interface import ProjectIoInterface
from glotaran.model.base_model import Model
from glotaran.plugin_system.base_registry import PluginOverwriteWarning
from glotaran.plugin_system.base_registry import add_instantiated_plugin_to_registry
from glotaran.plugin_system.base_registry import add_plugin_to_registry
from glotaran.plugin_system.base_registry import extend_conflicting_plugin_key
from glotaran.plugin_system.base_registry import full_plugin_name
from glotaran.plugin_system.base_registry import get_plugin_from_registry
from glotaran.plugin_system.base_registry import is_registered_plugin
from glotaran.plugin_system.base_registry import registered_plugins

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

    from glotaran.plugin_system.base_registry import _PluginInstantiableType
    from glotaran.plugin_system.base_registry import _PluginType


class MockPlugin:
    pass


mock_registry_data_io = cast(
    MutableMapping[str, DataIoInterface],
    {
        "sdt": SdtDataIo("sdt"),
        "yml": YmlProjectIo("yml"),
        "kinetic-image": KineticImageModel,
        "mock_plugin": MockPlugin,
        "imported_plugin": MockPlugin,
    },
)

mock_registry_project_io = cast(
    MutableMapping[str, ProjectIoInterface], copy(mock_registry_data_io)
)
mock_registry_model = cast(MutableMapping[str, Type[Model]], copy(mock_registry_data_io))


@pytest.mark.parametrize(
    "plugin, expected",
    (
        (SdtDataIo, "glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo"),
        (SdtDataIo("sdt"), "glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo"),
    ),
)
def test_full_plugin_name(plugin: object | type[object], expected: str):
    """Full name of class and instance is retrieved correctly."""
    assert full_plugin_name(plugin) == expected


def test_PluginOverwriteWarning():
    """Ignore the values this is just for testing, even if it doesn't make sense."""

    expected = (
        "The plugin 'glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo' tried to "
        "overwrite the plugin 'glotaran.builtin.io.yml.yml.YmlProjectIo', "
        "with the access_name 'yml'. "
        "Use 'yml_gta' to access 'glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo'."
    )

    with pytest.warns(PluginOverwriteWarning) as record:
        warn(
            PluginOverwriteWarning(
                old_key="yml", new_key="yml_gta", old_plugin=YmlProjectIo, new_plugin=SdtDataIo
            )
        )

        assert len(record) == 1
        assert record[0].message.args[0] == expected  # type: ignore


@pytest.mark.parametrize(
    "key, plugin, expected",
    (
        ("sdt", SdtDataIo("sdt"), "sdt_gta"),
        ("yml", YmlProjectIo("yml"), "yml_gta"),
        ("kinetic-image", KineticImageModel, "kinetic-image_gta"),
        ("mock_plugin", MockPlugin, "mock_plugin_test_base_registry"),
        ("mock_plugin", MockPlugin(), "mock_plugin_test_base_registry"),
        ("imported_plugin", pd.DataFrame, "imported_plugin_pandas"),
        ("imported_plugin", pd.DataFrame(), "imported_plugin_pandas"),
    ),
)
def test_extend_conflicting_plugin_key(key: str, plugin: object | type[object], expected: str):
    """Keys get proper postfix if there isn't a fixed key already."""
    assert extend_conflicting_plugin_key(key, plugin, mock_registry_data_io) == expected


def test_extend_conflicting_plugin_key_with_fixed_entry(monkeypatch: MonkeyPatch):
    """Keys get proper postfix if there isn't a fixed key already."""
    monkeypatch.setitem(mock_registry_data_io, "sdt_gta", SdtDataIo("sdt"))
    assert (
        extend_conflicting_plugin_key("sdt", SdtDataIo("sdt"), mock_registry_data_io)
        == "sdt_gta_1"
    )


@pytest.mark.parametrize(
    "plugin_register_key, plugin, plugin_registy",
    (
        ("sdt_new", SdtDataIo("sdt"), copy(mock_registry_data_io)),
        ("yml_new", YmlProjectIo("yml"), copy(mock_registry_project_io)),
        ("kinetic-image_new", KineticImageModel, copy(mock_registry_model)),
    ),
)
def test_add_plugin_to_register(
    plugin_register_key: str,
    plugin: _PluginType,
    plugin_registy: MutableMapping[str, _PluginType],
):
    """Add plugin with one key"""
    add_plugin_to_registry(plugin_register_key, plugin, plugin_registy)
    assert plugin_register_key in plugin_registy
    assert plugin_registy[plugin_register_key] == plugin


def test_add_plugin_to_register_existing_plugin():
    """Try plugin_overwrite"""

    plugin_registy = copy(mock_registry_data_io)
    with pytest.warns(PluginOverwriteWarning, match="SdtDataIo.+sdt.+sdt_gta") as record:
        add_plugin_to_registry("sdt", SdtDataIo("sdt"), plugin_registy)
        assert len(record) == 1
    assert "sdt_gta" in plugin_registy
    assert plugin_registy["sdt_gta"].format == "sdt"


@pytest.mark.parametrize(
    "plugin_register_key, plugin, plugin_registy",
    (
        ("sdt_new", SdtDataIo, copy(mock_registry_data_io)),
        ("yml_new", YmlProjectIo, copy(mock_registry_project_io)),
    ),
)
def test_add_instantiated_plugin_to_register(
    plugin_register_key: str,
    plugin: type[_PluginInstantiableType],
    plugin_registy: MutableMapping[str, _PluginInstantiableType],
):
    """Add instantiated plugin"""
    add_instantiated_plugin_to_registry(plugin_register_key, plugin, plugin_registy)
    assert plugin_register_key in plugin_registy
    assert plugin_registy[plugin_register_key].format == plugin_register_key


@pytest.mark.parametrize(
    "plugin_register_keys, plugin, plugin_registy",
    (
        (["sdt_new", "sdt_new2"], SdtDataIo, copy(mock_registry_data_io)),
        (["yml_new", "yaml_new"], YmlProjectIo, copy(mock_registry_project_io)),
    ),
)
def test_add_instantiated_plugin_to_register_multiple_keys(
    plugin_register_keys: list[str],
    plugin: type[_PluginInstantiableType],
    plugin_registy: MutableMapping[str, _PluginInstantiableType],
):
    """Add instantiated plugin with multiple keys"""
    add_instantiated_plugin_to_registry(plugin_register_keys, plugin, plugin_registy)
    for plugin_register_key in plugin_register_keys:
        assert plugin_register_key in plugin_registy
        assert plugin_registy[plugin_register_key].format == plugin_register_key


def test_registered_plugins():
    """List of registered names"""
    result = [
        "sdt",
        "yml",
        "kinetic-image",
        "mock_plugin",
        "imported_plugin",
    ]
    assert registered_plugins(mock_registry_data_io) == result


def test_is_registered_plugin():
    """Registered name in registry"""
    assert is_registered_plugin("sdt", mock_registry_data_io)
    assert not is_registered_plugin("not-registered", mock_registry_data_io)


def test_get_plugin_from_register():
    """Retrieve plugin from registry"""
    plugin = get_plugin_from_registry("sdt", mock_registry_data_io, "something went wrong")
    assert plugin.format == "sdt"


def test_get_plugin_from_register_not_found():
    """Error when Plugin wasn't found"""
    with pytest.raises(ValueError, match="something went wrong"):
        get_plugin_from_registry("not-registered", mock_registry_data_io, "something went wrong")
