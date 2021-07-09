"""Megacomplex registration convenience functions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from tabulate import tabulate

from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.base_registry import add_plugin_to_registry
from glotaran.plugin_system.base_registry import full_plugin_name
from glotaran.plugin_system.base_registry import get_plugin_from_registry
from glotaran.plugin_system.base_registry import is_registered_plugin
from glotaran.plugin_system.base_registry import registered_plugins
from glotaran.plugin_system.base_registry import set_plugin
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    from glotaran.model import Megacomplex


def register_megacomplex(megacomplex_type: str, megacomplex: type[Megacomplex]) -> None:
    """Add a megacomplex to the megacomplex registry.

    Parameters
    ----------
    megacomplex_type : str
        Name of the megacomplex under which it is registered.
    megacomplex : type[Megacomplex]
        megacomplex class to be registered.
    """
    add_plugin_to_registry(
        plugin_register_key=megacomplex_type,
        plugin=megacomplex,
        plugin_registry=__PluginRegistry.megacomplex,
        plugin_set_func_name="set_megacomplex_plugin",
    )


def is_known_megacomplex(megacomplex_type: str) -> bool:
    """Check if a megacomplex is in the megacomplex registry.

    Parameters
    ----------
    megacomplex_type : str
        Name of the megacomplex under which it is registered.

    Returns
    -------
    bool
        Whether or not the megacomplex is registered.
    """
    return is_registered_plugin(
        plugin_register_key=megacomplex_type, plugin_registry=__PluginRegistry.megacomplex
    )


def get_megacomplex(megacomplex_type: str) -> type[Megacomplex]:
    """Retrieve a megacomplex from the megacomplex registry.

    Parameters
    ----------
    megacomplex_type : str
        Name of the megacomplex under which it is registered.

    Returns
    -------
    type[Megacomplex]
        Megacomplex class
    """
    return get_plugin_from_registry(
        plugin_register_key=megacomplex_type,
        plugin_registry=__PluginRegistry.megacomplex,
        not_found_error_message=(
            f"Unknown megacomplex type {megacomplex_type!r}. "
            f"Known megacomplex types are: {known_megacomplex_names(full_names=True)}"
        ),
    )


def known_megacomplex_names(full_names: bool = False) -> list[str]:
    """Names of the registered megacomplexs.

    Parameters
    ----------
    full_names : bool
        Whether to display the full names the plugins are
        registered under as well.

    Returns
    -------
    list[str]
        List of registered megacomplexs.
    """
    return registered_plugins(__PluginRegistry.megacomplex, full_names=full_names)


def set_megacomplex_plugin(megacomplex_name: str, full_plugin_name: str) -> None:
    """Set the plugin used for a specific megacomplex name.

    This function is useful when you want to resolve conflicts of installed plugins
    or overwrite the plugin used for a specific megacomplex name.

    Effected functions:

    - :func:`optimize`

    Parameters
    ----------
    megacomplex_name : str
        Name of the megacomplex to use the plugin for.
    full_plugin_name : str
        Full name (import path) of the registered plugin.
    """
    set_plugin(
        plugin_register_key=megacomplex_name,
        full_plugin_name=full_plugin_name,
        plugin_registry=__PluginRegistry.megacomplex,
        plugin_register_key_name="megacomplex_name",
    )


def megacomplex_plugin_table(
    *, plugin_names: bool = False, full_names: bool = False
) -> MarkdownStr:
    """Return registered megacomplex plugins as markdown table.

    This is especially useful when you work with new plugins.

    Parameters
    ----------
    plugin_names : bool
        Whether or not to add the names of the plugins to the table.
    full_names : bool
        Whether to display the full names the plugins are
        registered under as well.

    Returns
    -------
    MarkdownStr
        Markdown table of megacomplexnames.
    """
    table_data = []
    megacomplex_names = known_megacomplex_names(full_names=full_names)
    header_values = ["Megacomplex name"]
    if plugin_names:
        header_values.append("Plugin name")
        for megacomplex_name in megacomplex_names:
            table_data.append(
                [
                    f"`{megacomplex_name}`",
                    f"`{full_plugin_name(get_megacomplex(megacomplex_name))}`",
                ]
            )
    else:
        table_data = [[f"`{megacomplex_name}`"] for megacomplex_name in megacomplex_names]
    headers = tuple(map(lambda x: f"__{x}__", header_values))
    return MarkdownStr(tabulate(table_data, tablefmt="github", headers=headers, stralign="center"))
