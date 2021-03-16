"""Functionality to register, initialize and retrieve glotaran plugins.

Since this module is imported at the root ``__init__.py`` file all other
glotaran imports should be used for typechecking only in the 'if TYPE_CHECKING' block.
This is to prevent issues with circular imports.
"""
from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import MutableMapping
    from typing import Sequence
    from typing import TypeVar

    from glotaran.io.interface import DataIoInterface
    from glotaran.io.interface import ProjectIoInterface
    from glotaran.model.base_model import Model

    _PluginType = TypeVar("_PluginType", type[Model], DataIoInterface, ProjectIoInterface)
    _PluginInstantiableType = TypeVar(
        "_PluginInstantiableType", DataIoInterface, ProjectIoInterface
    )

    GenericPluginInstance = TypeVar("GenericPluginInstance", bound=object)


class __PluginRegistry:
    """Central Plugin Registry.

    This is super private since if anyone messes with it, the pluginsystem could break.
    """

    model: MutableMapping[str, type[Model]] = {}
    data_io: MutableMapping[str, DataIoInterface] = {}
    project_io: MutableMapping[str, ProjectIoInterface] = {}


def full_plugin_name(plugin: object | type[object]) -> str:
    """Full name of a plugin instance/class similar to the ``repr``.

    Parameters
    ----------
    plugin : object | type[object]
        plugin instance/class

    Examples
    --------
    >>> from glotaran.builtin.io.sdt.sdt_file_reader import SdtDataIo
    >>> full_plugin_name(SdtDataIo)
    "glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo"
    >>> full_plugin_name(SdtDataIo("sdt"))
    "glotaran.builtin.io.sdt.sdt_file_reader.SdtDataIo"

    Returns
    -------
    str
        Full name of the plugin.
    """
    if isinstance(plugin, type):
        return f"{plugin.__module__}.{plugin.__name__}"
    else:
        return f"{plugin.__module__}.{type(plugin).__name__}"


class PluginOverwriteWarning(UserWarning):
    """Warning used if a plugin tries to overwrite and existing plugin."""

    def __init__(
        self,
        *args: Any,
        old_key: str,
        new_key: str,
        old_plugin: object | type[object],
        new_plugin: object | type[object],
    ):
        """Use old and new plugin and keys to give verbose warning message.

        Parameters
        ----------
        old_key : str
            Old registry key.
        new_key : str
            New none conflicting registry key.
        old_plugin :  object | type[object]
            Old plugin ('registry[old_key]').
        new_plugin :  object | type[object]
            New Plugin ('registry[new_key]').
        *args : Any
            Additional args passed to the super constructor.
        """
        old_plugin_name = full_plugin_name(old_plugin)
        new_plugin_name = full_plugin_name(new_plugin)
        message = (
            f"The plugin {new_plugin_name!r} tried to overwrite the plugin {old_plugin_name!r}, "
            f"with the access_name {old_key!r}. Use {new_key!r} to access {new_plugin_name!r}."
        )
        super().__init__(message, *args)


def load_plugins():
    """Initialize plugins registered under the entrypoint 'glotaran.plugins'.

    For an entry_point to be considered a glotaran plugin it just needs to start with
    'glotaran.plugins', which allows for an easy extendability.

    Currently used builtin entrypoints are:

    - ``glotaran.plugins.data_io``
    - ``glotaran.plugins.model``
    - ``glotaran.plugins.project_io``
    """
    for entry_point_name, entry_points in metadata.entry_points().items():
        if entry_point_name.startswith("glotaran.plugins"):
            for entry_point in entry_points:
                entry_point.load()


def extend_conflicting_plugin_key(
    plugin_registry_key: str,
    plugin: object | type[object],
    plugin_registry: MutableMapping[str, _PluginType],
    numerical_postfix: int = 0,
) -> str:
    """Postfix a plugin_registry_key with the module name the plugin originates from.

    Parameters
    ----------
    plugin_registry_key : str
        Original key a plugin tried to register itself with.
    plugin : object | type[object]
        Plugin instance or class to be added.
    plugin_registry : MutableMapping[str, _PluginType]
        Register the plugin should be added to.
    numerical_postfix : int
        Additional postfix if the fixed key already exists , by default 0

    Returns
    -------
    str
        New key for the plugin which isn't in the plugin register yet.

    See Also
    --------
    add_plugin_to_register
    """
    origin_module = plugin.__module__
    if origin_module.startswith("glotaran.builtin"):
        new_plugin_registry_key = f"{plugin_registry_key}_gta"
    else:
        origin_module, _, _ = origin_module.partition(".")
        new_plugin_registry_key = f"{plugin_registry_key}_{origin_module}"

    if numerical_postfix != 0:
        new_plugin_registry_key = f"{new_plugin_registry_key}_{numerical_postfix}"

    if new_plugin_registry_key not in plugin_registry:
        return new_plugin_registry_key
    else:
        return extend_conflicting_plugin_key(
            plugin_registry_key,
            plugin,
            plugin_registry,
            numerical_postfix + 1,
        )


def add_plugin_to_registry(
    plugin_register_key: str,
    plugin: _PluginType,
    plugin_registry: MutableMapping[str, _PluginType],
) -> None:
    """Add a plugin with name ``plugin_register_key`` to the given registry.

    Parameters
    ----------
    plugin_register_key : str
        Name of the plugin under which it is registered.
    plugin: _PluginType
        Plugin to be added to the registry.
    plugin_registry: MutableMapping[str, _PluginType]
        Registry the plugin should be added to.

    See Also
    --------
    add_instantiated_plugin_to_register
    """
    if plugin_register_key in plugin_registry:
        old_key = plugin_register_key
        plugin_register_key = extend_conflicting_plugin_key(
            plugin_register_key, plugin, plugin_registry
        )
        warn(
            PluginOverwriteWarning(
                old_key=old_key,
                new_key=plugin_register_key,
                old_plugin=plugin_registry[old_key],
                new_plugin=plugin,
            )
        )
    plugin_registry[plugin_register_key] = plugin


def add_instantiated_plugin_to_registry(
    plugin_register_keys: str | list[str],
    plugin_class: type[_PluginInstantiableType],
    plugin_registry: MutableMapping[str, _PluginInstantiableType],
) -> None:
    """Add instances of plugin_class to the given registry.

    Parameters
    ----------
    plugin_register_keys : str | list[str]
        Name/-s of the plugin under which it is registered.
    plugin_class : type[_PluginInstantiableType]
        Pluginclass which should be instantiated with ``plugin_register_keys``
        and added to the registry.
    plugin_registry : MutableMapping[str, _PluginInstantiableType]
        Registry the plugin should be added to.

    See Also
    --------
    add_plugin_to_register
    """
    if isinstance(plugin_register_keys, str):
        plugin_register_keys = [plugin_register_keys]
    for plugin_register_key in plugin_register_keys:
        # The type ignore is needed due to an issue with mypy
        # ``Cannot instantiate type "Type[Type[DataIoInterface]]"``
        add_plugin_to_registry(
            plugin_register_key,
            plugin_class(plugin_register_key),  # type:ignore[misc]
            plugin_registry,
        )


def registered_plugins(plugin_registry: MutableMapping[str, _PluginType]) -> list[str]:
    """Names of the plugins in the given registry.

    Parameters
    ----------
    plugin_registry : MutableMapping[str, _PluginType]
        Registry to search in.

    Returns
    -------
    list[str]
        List of plugin names in plugin_registry.
    """
    return list(plugin_registry.keys())


def is_registered_plugin(
    plugin_register_key: str, plugin_registry: MutableMapping[str, _PluginType]
) -> bool:
    """Check if a plugin with name ``plugin_register_key`` is registered in the given registry.

    Parameters
    ----------
    plugin_register_key : str
        Name of the plugin under which it is registered.
    plugin_registry : MutableMapping[str, _PluginType]
        Registry to search in.

    Returns
    -------
    bool
        Whether or not a plugin is in the registry.
    """
    return plugin_register_key in plugin_registry


def get_plugin_from_registry(
    plugin_register_key: str,
    plugin_registry: MutableMapping[str, _PluginType],
    not_found_error_message: str,
) -> _PluginType:
    """Retrieve a plugin with name ``plugin_register_key`` is registered in a given registry.

    Parameters
    ----------
    plugin_register_key : str
        Name of the plugin under which it is registered.
    plugin_registry : MutableMapping[str, _PluginType]
        Registry to search in.
    not_found_error_message : str
        Error message to be shown if the plugin wasn't found.

    Returns
    -------
    _PluginType
        Plugin from the plugin Registry.

    Raises
    ------
    ValueError
        If there was no plugin registered under the name ``plugin_register_key``.
    """
    if not is_registered_plugin(plugin_register_key, plugin_registry):
        raise ValueError(not_found_error_message)
    else:
        return plugin_registry[plugin_register_key]


def get_method_from_plugin(
    plugin: object | type[object],
    method_name: str,
) -> Callable[..., Any]:
    """Retrieve a method callabe from an class or instance plugin.

    Parameters
    ----------
    plugin : object | type[object],
        Plugin instance or class.
    method_name : str
        Method name, e.g. load_model.

    Returns
    -------
    Callable[..., Any]
        Method callable.

    Raises
    ------
    ValueError
        If plugin has an attribute with that name but it isn't callable.
    ValueError
        If plugin misses the attribute.
    """
    not_a_method_error_message = (
        f"The plugin {full_plugin_name(plugin)!r} has no method {method_name!r}"
    )
    try:
        possible_method = getattr(plugin, method_name)
        if callable(possible_method):
            return possible_method
        else:
            raise ValueError(not_a_method_error_message)
    except AttributeError:
        raise ValueError(not_a_method_error_message)


def show_method_help(
    plugin: object | type[object],
    method_name: str,
) -> None:
    """Show help on a method as if it was called directly on it.

    Parameters
    ----------
    plugin : object | type[object],
        Plugin instance or class.
    method_name : str
        Method name, e.g. load_model.
    """
    method = get_method_from_plugin(plugin, method_name)
    help(method)


def methods_differ_from_baseclass(
    method_names: str | Sequence[str],
    plugin: GenericPluginInstance | type[GenericPluginInstance],
    base_class: type[GenericPluginInstance],
) -> list[bool]:
    """Check if a plugins methods implementation differ from its baseclass.

    Based on the assumption that ``base_class`` didn't implement the methods
    (e.g. :class:`DataIoInterface` or :class:`ProjectIoInterface`), this can be
    used to to create a 'supported methods' list.

    Parameters
    ----------
    method_names : str | list[str]
        Name|s of the method|s
    plugin : GenericPluginInstance | type[GenericPluginInstance]
        Plugin class or instance.
    base_class : type[GenericPluginInstance]
        Base class the plugin inherited from.

    Returns
    -------
    list[bool]
        List containing whether or not a plugins method differs from the baseclasses.
    """
    differs_list = []
    if isinstance(method_names, str):
        method_names = [method_names]
    for method_name in method_names:
        plugin_method = get_method_from_plugin(plugin, method_name)
        base_class_method = get_method_from_plugin(base_class, method_name)
        differs_list.append(plugin_method.__code__ != base_class_method.__code__)
    return differs_list


def methods_differ_from_baseclass_table(
    method_names: str | Sequence[str],
    plugin_registry_keys: str | Sequence[str],
    get_plugin_function: Callable[[str], GenericPluginInstance | type[GenericPluginInstance]],
    base_class: type[GenericPluginInstance],
) -> list[list[str | bool]]:
    """Create table of which plugins methods differ from their baseclass.

    This uses the assumption that all plugins have the same ``base_class``.

    The main purpose of this function is to show the user which plugin implements
    which methods differently than its baseclass.

    Based on the assumption that ``base_class`` didn't implement the methods
    (e.g. :class:`DataIoInterface` or :class:`ProjectIoInterface`), this can be
    used to to create a 'supported methods' table.

    Parameters
    ----------
    method_names : str | list[str]
        Name|s of the method|s.
    plugin_registry_keys : str | list[str]
        Keys the plugins are registered under
        (e.g. return value of the implementation of func:`registered_plugins`)
    get_plugin_function: Callable[[str], GenericPluginInstance | type[GenericPluginInstance]]
        Function to get plugin from plugin registry.
    base_class : type[GenericPluginInstance]
        Base class the plugin inherited from.

    Returns
    -------
    list[list[str | bool]]
        Table like structure with the first value of each row being the
        ``plugin_registry_key`` and the others whether or not a plugins
        method differs from the baseclasses.

    See Also
    --------
    methods_differ_from_baseclass
    """
    differs_table: list[list[str | bool]] = []
    if isinstance(plugin_registry_keys, str):
        plugin_registry_keys = [plugin_registry_keys]
    for plugin_registry_key in plugin_registry_keys:
        plugin = get_plugin_function(plugin_registry_key)
        differs_list = methods_differ_from_baseclass(method_names, plugin, base_class)
        differs_table.append([plugin_registry_key, *differs_list])
    return differs_table
