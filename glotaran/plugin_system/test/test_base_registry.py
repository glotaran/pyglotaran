from __future__ import annotations

from collections.abc import MutableMapping
from copy import copy
from typing import TYPE_CHECKING
from typing import cast
from warnings import warn

import pytest

from glotaran.builtin.io.sdt.sdt_file_reader import SdtDataIo
from glotaran.builtin.io.yml.yml import YmlProjectIo
from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.io.interface import DataIoInterface
from glotaran.io.interface import ProjectIoInterface
from glotaran.model.megacomplex import Megacomplex
from glotaran.plugin_system.base_registry import PluginOverwriteWarning
from glotaran.plugin_system.base_registry import add_instantiated_plugin_to_registry
from glotaran.plugin_system.base_registry import add_plugin_to_registry
from glotaran.plugin_system.base_registry import full_plugin_name
from glotaran.plugin_system.base_registry import get_method_from_plugin
from glotaran.plugin_system.base_registry import get_plugin_from_registry
from glotaran.plugin_system.base_registry import is_registered_plugin
from glotaran.plugin_system.base_registry import methods_differ_from_baseclass
from glotaran.plugin_system.base_registry import methods_differ_from_baseclass_table
from glotaran.plugin_system.base_registry import registered_plugins
from glotaran.plugin_system.base_registry import set_plugin
from glotaran.plugin_system.base_registry import show_method_help
from glotaran.plugin_system.base_registry import supported_file_extensions

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture

    from glotaran.plugin_system.base_registry import _PluginInstantiableType
    from glotaran.plugin_system.base_registry import _PluginType


class MockPlugin:
    format_name = "mock"

    def some_method(self):
        """This docstring is just for help testing of 'some_method'."""
        return f"got the method of {self.format_name}"

    def some_other_method(self):
        pass


class MockPluginSubclassPartial(MockPlugin):
    def some_method(self):
        return "different implementation"


class MockPluginSubclassFull(MockPlugin):
    def some_method(self):
        return "different implementation"

    def some_other_method(self):
        return "different implementation"


class MockPluginSubclassStr(MockPlugin):
    def some_method(self):
        return "different implementation"


def get_mock_plugin_function(plugin_registry_key: str):
    if plugin_registry_key == "base":
        return MockPlugin
    elif plugin_registry_key == "sub_class_partial":
        return MockPluginSubclassPartial
    elif plugin_registry_key == "sub_class_full":
        return MockPluginSubclassFull
    elif plugin_registry_key == "sub_class_str":
        return MockPluginSubclassStr
    raise ValueError(f"No mock plugin with name {plugin_registry_key!r}")


mock_registry_data_io = cast(
    MutableMapping[str, DataIoInterface],
    {
        "sdt": SdtDataIo("sdt"),
        "yml": YmlProjectIo("yml"),
        "glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo": SdtDataIo("sdt"),
        "decay": DecayMegacomplex,
        "mock_plugin": MockPlugin,
        "imported_plugin": MockPlugin,
    },
)

mock_registry_project_io = cast(
    MutableMapping[str, ProjectIoInterface], copy(mock_registry_data_io)
)
mock_registry_model = cast(MutableMapping[str, type[Megacomplex]], copy(mock_registry_data_io))


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

    with pytest.warns(PluginOverwriteWarning) as record:
        warn(
            PluginOverwriteWarning(
                old_key="yml",
                old_plugin=YmlProjectIo,
                new_plugin=SdtDataIo,
                plugin_set_func_name="set_plugin",
            )
        )

        assert len(record) == 1
        expected = (
            "The plugin 'glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo' tried to "
            "overwrite the plugin 'glotaran.builtin.io.yml.yml.YmlProjectIo', "
            "with the access_name 'yml'. "
            "Use set_plugin('yml', 'glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo') "
            "to use 'glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo' instead."
        )

        assert record[0].message.args[0] == expected


@pytest.mark.parametrize(
    "plugin_register_key, plugin, plugin_registry",
    (
        ("sdt_new", SdtDataIo("sdt"), copy(mock_registry_data_io)),
        ("yml_new", YmlProjectIo("yml"), copy(mock_registry_project_io)),
        ("decay_new", DecayMegacomplex, copy(mock_registry_model)),
    ),
)
def test_add_plugin_to_register(
    plugin_register_key: str,
    plugin: _PluginType,
    plugin_registry: MutableMapping[str, _PluginType],
):
    """Add plugin with one key"""
    add_plugin_to_registry(plugin_register_key, plugin, plugin_registry, "set_plugin")
    assert plugin_register_key in plugin_registry
    assert plugin_registry[plugin_register_key] == plugin


def test_add_plugin_to_register_naming_error():
    """Raise error if pluginkey contains '.'"""
    with pytest.raises(
        ValueError,
        match=(
            r"The character '\.' isn't allowed in the name of a plugin, "
            r"you provided the name 'bad\.name'."
        ),
    ):
        add_plugin_to_registry("bad.name", MockPlugin, mock_registry_model, "set_plugin")


def test_add_plugin_to_register_existing_plugin():
    """Warn if different plugin overwrites existing."""

    plugin_registry = copy(mock_registry_data_io)
    with pytest.warns(PluginOverwriteWarning, match="SdtDataIo.+sdt.+YmlProjectIo") as record:
        add_plugin_to_registry("sdt", YmlProjectIo("sdt"), plugin_registry, "set_plugin")
        assert len(record) == 1
    assert plugin_registry["glotaran.builtin.io.yml.yml.YmlProjectIo"].format == "sdt"


@pytest.mark.xfail(strict=True, reason="Should not warn")
def test_add_plugin_to_register_existing_plugin_self():
    """Don't warn if plugin overwrites itself."""

    plugin_registry = copy(mock_registry_data_io)
    with pytest.warns(PluginOverwriteWarning, match="SdtDataIo.+sdt.+SdtDataIo") as record:
        add_plugin_to_registry("sdt", SdtDataIo("sdt"), plugin_registry, "set_plugin")
        assert len(record) == 0


@pytest.mark.parametrize(
    "plugin_register_key, plugin, plugin_registry",
    (
        ("sdt_new", SdtDataIo, copy(mock_registry_data_io)),
        ("yml_new", YmlProjectIo, copy(mock_registry_project_io)),
    ),
)
def test_add_instantiated_plugin_to_register(
    plugin_register_key: str,
    plugin: type[_PluginInstantiableType],
    plugin_registry: MutableMapping[str, _PluginInstantiableType],
):
    """Add instantiated plugin"""
    add_instantiated_plugin_to_registry(plugin_register_key, plugin, plugin_registry, "set_plugin")
    assert plugin_register_key in plugin_registry
    assert plugin_registry[plugin_register_key].format == plugin_register_key


@pytest.mark.parametrize(
    "plugin_register_keys, plugin, plugin_registry",
    (
        (["sdt_new", "sdt_new2"], SdtDataIo, copy(mock_registry_data_io)),
        (["yml_new", "yaml_new"], YmlProjectIo, copy(mock_registry_project_io)),
    ),
)
def test_add_instantiated_plugin_to_register_multiple_keys(
    plugin_register_keys: list[str],
    plugin: type[_PluginInstantiableType],
    plugin_registry: MutableMapping[str, _PluginInstantiableType],
):
    """Add instantiated plugin with multiple keys"""
    add_instantiated_plugin_to_registry(
        plugin_register_keys, plugin, plugin_registry, "set_plugin"
    )
    for plugin_register_key in plugin_register_keys:
        assert plugin_register_key in plugin_registry
        assert plugin_registry[plugin_register_key].format == plugin_register_key


def test_set_plugin():
    """Overwrite existing plugin"""
    plugin_registry = copy(mock_registry_data_io)
    set_plugin("yml", "glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo", plugin_registry)
    assert isinstance(plugin_registry["yml"], SdtDataIo)


def test_set_plugin_dot_in_plugin_key():
    """Raise error if plugin_register_key contains '.'"""
    plugin_registry = copy(mock_registry_data_io)
    with pytest.raises(
        ValueError,
        match=r"The value of 'format_name' isn't allowed to contain the character '\.' \.",
    ):
        set_plugin("foo.bar", "glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo", plugin_registry)


def test_set_plugin_plugin_not_registered():
    """Raise error if the is not plugin"""
    plugin_registry = copy(mock_registry_data_io)
    with pytest.raises(
        ValueError,
        match=(
            r"There isn't a plugin registered under the full name 'mymodul\.NotRegistered'\.\n"
            r"Maybe you need to install a plugin\? Known plugins are:\n "
            r"\['glotaran\.builtin\.io\.sdt\.sdt_file_reader\.SdtDataIo'\]"
        ),
    ):
        set_plugin("yml", "mymodul.NotRegistered", plugin_registry)


def test_set_plugin_no_dot_full_name():
    """Raise error if full_plugin_name doesn't contains '.'"""
    plugin_registry = copy(mock_registry_data_io)
    with pytest.raises(
        ValueError,
        match=(
            r"There isn't a plugin registered under the full name 'sdt'\.\n"
            r"Maybe you need to install a plugin\? Known plugins are:\n "
            r"\['glotaran\.builtin\.io\.sdt\.sdt_file_reader\.SdtDataIo'\]"
        ),
    ):
        set_plugin("yml", "sdt", plugin_registry)


def test_registered_plugins():
    """List of registered names"""
    result = [
        "sdt",
        "yml",
        "decay",
        "mock_plugin",
        "imported_plugin",
    ]
    assert registered_plugins(mock_registry_data_io) == sorted(result)


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


def test_get_method_from_plugin():
    """Method works like a function."""
    plugin = MockPlugin()

    method = get_method_from_plugin(plugin, "some_method")

    assert method() == "got the method of mock"


def test_get_method_from_plugin_just_attribute():
    """Error if attribute does exists on plugin but is not a method."""
    plugin = MockPlugin()

    with pytest.raises(
        ValueError, match=r"The plugin '.+?\.MockPlugin' has no method 'format_name'"
    ):
        get_method_from_plugin(plugin, "format_name")


def test_get_method_from_plugin_missing_attribute():
    """Error if attribute does not exists on plugin."""
    plugin = MockPlugin()

    with pytest.raises(
        ValueError, match=r"The plugin '.+?\.MockPlugin' has no method 'not_even_an_attribute'"
    ):
        get_method_from_plugin(plugin, "not_even_an_attribute")


@pytest.mark.parametrize("plugin", (MockPlugin, MockPlugin()))
def test_show_method_help(capsys: CaptureFixture, plugin: MockPlugin | type[MockPlugin]):
    """Same help as when called directly.

    Note: help differs when called on class.method vs. instance.method
    """
    help(plugin.some_method)
    original_help, _ = capsys.readouterr()

    show_method_help(plugin, "some_method")
    result, _ = capsys.readouterr()

    assert "This docstring is just for help testing of 'some_method'." in result
    assert result == original_help


@pytest.mark.parametrize(
    "method_names,plugin,expected",
    (
        ("some_method", MockPluginSubclassPartial, [True]),
        ("some_method", MockPluginSubclassPartial(), [True]),
        (["some_method", "some_other_method"], MockPluginSubclassPartial, [True, False]),
        (["some_method", "some_other_method"], MockPluginSubclassPartial(), [True, False]),
        (["some_method", "some_other_method"], MockPluginSubclassFull, [True, True]),
        (["some_method", "some_other_method"], MockPluginSubclassFull(), [True, True]),
    ),
)
def test_methods_differ_from_baseclass(
    method_names: str | list[str], plugin: object | type[object], expected: list[bool]
):
    """Inherited methods are the same as base class and overwritten ones differ"""
    result = methods_differ_from_baseclass(method_names, plugin, MockPlugin)

    assert list(result) == expected


@pytest.mark.parametrize(
    "method_names,plugin_registry_keys,expected",
    (
        ("some_method", "base", [["`base`", False]]),
        ("some_method", "sub_class_partial", [["`sub_class_partial`", True]]),
        ("some_method", "sub_class_full", [["`sub_class_full`", True]]),
        (
            ["some_method", "some_other_method"],
            ["base", "sub_class_partial", "sub_class_full", "sub_class_str"],
            [
                ["`base`", False, False],
                ["`sub_class_partial`", True, False],
                ["`sub_class_full`", True, True],
                ["`sub_class_str`", True, False],
            ],
        ),
    ),
)
def test_methods_differ_from_baseclass_table(
    method_names: str | list[str],
    plugin_registry_keys: str | list[str],
    expected: list[list[str | bool]],
):
    """Inherited methods are the same as base class and overwritten ones differ"""

    result = methods_differ_from_baseclass_table(
        method_names, plugin_registry_keys, get_mock_plugin_function, MockPlugin
    )

    assert list(result) == expected


def test_methods_differ_from_baseclass_table_plugin_names():
    """Show plugin name"""

    result = methods_differ_from_baseclass_table(
        "some_method", "base", get_mock_plugin_function, MockPlugin, plugin_names=True
    )

    assert list(result) == [["`base`", False, "`test_base_registry.MockPlugin`"]]


@pytest.mark.parametrize(
    "method_names, expected",
    (
        ("some_method", [".sub_class_partial", ".sub_class_full"]),
        ("some_other_method", [".sub_class_full"]),
        (["some_method", "some_other_method"], [".sub_class_full"]),
    ),
)
def test_supported_file_extensions(
    method_names: str | list[str],
    expected: list[str],
):
    """Only extensions where the plugin supports all methods in ``method_names`` are returned."""
    result = supported_file_extensions(
        method_names,
        ["base", "sub_class_partial", "sub_class_full", "sub_class_str"],
        get_mock_plugin_function,
        MockPlugin,
    )

    assert list(result) == expected
