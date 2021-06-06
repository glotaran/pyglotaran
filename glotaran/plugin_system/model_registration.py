"""Model registration convenience functions."""
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
    from glotaran.model import Model


def register_model(model_type: str, model: type[Model]) -> None:
    """Add a model to the model registry.

    Parameters
    ----------
    model_type : str
        Name of the model under which it is registered.
    model : type[Model]
        model class to be registered.
    """
    add_plugin_to_registry(
        plugin_register_key=model_type,
        plugin=model,
        plugin_registry=__PluginRegistry.model,
        plugin_set_func_name="set_model_plugin",
    )


def is_known_model(model_type: str) -> bool:
    """Check if a model is in the model registry.

    Parameters
    ----------
    model_type : str
        Name of the model under which it is registered.

    Returns
    -------
    bool
        Whether or not the model is registered.
    """
    return is_registered_plugin(
        plugin_register_key=model_type, plugin_registry=__PluginRegistry.model
    )


def get_model(model_type: str) -> type[Model]:
    """Retrieve a model from the model registry.

    Parameters
    ----------
    model_type : str
        Name of the model under which it is registered.

    Returns
    -------
    type[Model]
        Model class
    """
    return get_plugin_from_registry(
        plugin_register_key=model_type,
        plugin_registry=__PluginRegistry.model,
        not_found_error_message=(
            f"Unknown model type {model_type!r}. "
            f"Known model types are: {known_model_names(full_names=True)}"
        ),
    )


def known_model_names(full_names: bool = False) -> list[str]:
    """Names of the registered models.

    Parameters
    ----------
    full_names : bool
        Whether to display the full names the plugins are
        registered under as well.

    Returns
    -------
    list[str]
        List of registered models.
    """
    return registered_plugins(__PluginRegistry.model, full_names=full_names)


def set_model_plugin(model_name: str, full_plugin_name: str) -> None:
    """Set the plugin used for a specific model name.

    This function is useful when you want to resolve conflicts of installed plugins
    or overwrite the plugin used for a specific model name.

    Effected functions:

    - :func:`optimize`

    Parameters
    ----------
    model_name : str
        Name of the model to use the plugin for.
    full_plugin_name : str
        Full name (import path) of the registered plugin.
    """
    set_plugin(
        plugin_register_key=model_name,
        full_plugin_name=full_plugin_name,
        plugin_registry=__PluginRegistry.model,
        plugin_register_key_name="model_name",
    )


def model_plugin_table(*, plugin_names: bool = False, full_names: bool = False) -> MarkdownStr:
    """Return registered model plugins as markdown table.

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
    str
        Markdown table of modelnames.
    """
    table_data = []
    model_names = known_model_names(full_names=full_names)
    header_values = ["Model name"]
    if plugin_names:
        header_values.append("Plugin name")
        for model_name in model_names:
            table_data.append([f"`{model_name}`", f"`{full_plugin_name(get_model(model_name))}`"])
    else:
        table_data = [[f"`{model_name}`"] for model_name in model_names]
    headers = tuple(map(lambda x: f"__{x}__", header_values))
    return MarkdownStr(tabulate(table_data, tablefmt="github", headers=headers, stralign="center"))
