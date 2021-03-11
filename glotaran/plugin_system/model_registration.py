"""Model registration convenience functions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.base_registry import add_plugin_to_registry
from glotaran.plugin_system.base_registry import get_plugin_from_registry
from glotaran.plugin_system.base_registry import is_registered_plugin
from glotaran.plugin_system.base_registry import registered_plugins

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
        plugin_register_key=model_type, plugin=model, plugin_registry=__PluginRegistry.model
    )


def known_model(model_type: str) -> bool:
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
            f"Unknown model type {model_type!r}. Known model types are: {known_model_names()}"
        ),
    )


def known_model_names() -> list[str]:
    """Names of the registered models.

    Returns
    -------
    list[str]
        List of registered models.
    """
    return registered_plugins(__PluginRegistry.model)
