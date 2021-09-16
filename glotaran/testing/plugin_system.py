"""Mock functionality for the plugin system."""
from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest import mock

from glotaran.plugin_system.base_registry import __PluginRegistry

if TYPE_CHECKING:
    from typing import Generator
    from typing import MutableMapping

    from glotaran.io.interface import DataIoInterface
    from glotaran.io.interface import ProjectIoInterface
    from glotaran.model.megacomplex import Megacomplex
    from glotaran.plugin_system.base_registry import _PluginType


@contextmanager
def _monkeypatch_plugin_registry(
    register_name: str,
    test_registry: MutableMapping[str, _PluginType] | None = None,
    create_new_registry: bool = False,
) -> Generator[None, None, None]:
    """Contextmanager to monkeypatch any Pluginregistry with name ``register_name``.

    Parameters
    ----------
    register_name : str
        Name of the register which should be patched.
    test_registry : MutableMapping[str, _PluginType]
        Registry to to update or replace the ``register_name`` registry with.
        , by default None
    create_new_registry : bool
        Whether to update the actual registry or create a new one from ``test_registry``
        , by default False

    Yields
    ------
    Generator[None, None, None]
        Just to keep the context alive.

    See Also
    --------
    monkeypatch_plugin_registry_megacomplex
    monkeypatch_plugin_registry_data_io
    monkeypatch_plugin_registry_project_io
    """
    if test_registry is not None:
        initila_plugins = (
            __PluginRegistry.__dict__[register_name] if not create_new_registry else {}
        )

        with mock.patch.object(
            __PluginRegistry, register_name, {**initila_plugins, **test_registry}
        ):
            yield
    else:
        yield


@contextmanager
def monkeypatch_plugin_registry_megacomplex(
    test_megacomplex: MutableMapping[str, type[Megacomplex]] | None = None,
    create_new_registry: bool = False,
) -> Generator[None, None, None]:
    """Monkeypatch the :class:`Megacomplex` registry.

    Parameters
    ----------
    test_megacomplex : MutableMapping[str, type[Megacomplex]], optional
        Registry to to update or replace the ``Megacomplex`` registry with.
        , by default None
    create_new_registry : bool
        Whether to update the actual registry or create a new one from ``test_megacomplex``
        , by default False

    Yields
    ------
    Generator[None, None, None]
        Just to keep the context alive.
    """
    with _monkeypatch_plugin_registry("megacomplex", test_megacomplex, create_new_registry):
        yield


@contextmanager
def monkeypatch_plugin_registry_data_io(
    test_data_io: MutableMapping[str, DataIoInterface] | None = None,
    create_new_registry: bool = False,
) -> Generator[None, None, None]:
    """Monkeypatch the :class:`DataIoInterface` registry.

    Parameters
    ----------
    test_data_io : MutableMapping[str, DataIoInterface], optional
        Registry to to update or replace the ``DataIoInterface`` registry with.
        , by default None
    create_new_registry : bool
        Whether to update the actual registry or create a new one from ``test_data_io``
        , by default False

    Yields
    ------
    Generator[None, None, None]
        Just to keep the context alive.
    """
    with _monkeypatch_plugin_registry("data_io", test_data_io, create_new_registry):
        yield


@contextmanager
def monkeypatch_plugin_registry_project_io(
    test_project_io: MutableMapping[str, ProjectIoInterface] | None = None,
    create_new_registry: bool = False,
) -> Generator[None, None, None]:
    """Monkeypatch the :class:`ProjectIoInterface` registry.

    Parameters
    ----------
    test_project_io : MutableMapping[str, ProjectIoInterface], optional
        Registry to to update or replace the ``ProjectIoInterface`` registry with.
        , by default None
    create_new_registry : bool
        Whether to update the actual registry or create a new one from ``test_data_io``
        , by default False

    Yields
    ------
    Generator[None, None, None]
        Just to keep the context alive.
    """
    with _monkeypatch_plugin_registry("project_io", test_project_io, create_new_registry):
        yield


@contextmanager
def monkeypatch_plugin_registry(
    *,
    test_megacomplex: MutableMapping[str, type[Megacomplex]] | None = None,
    test_data_io: MutableMapping[str, DataIoInterface] | None = None,
    test_project_io: MutableMapping[str, ProjectIoInterface] | None = None,
    create_new_registry: bool = False,
) -> Generator[None, None, None]:
    """Contextmanager to monkeypatch multiple plugin registries at once.

    Parameters
    ----------
    test_megacomplex : MutableMapping[str, type[Megacomplex]], optional
        Registry to to update or replace the ``Megacomplex`` registry with.
        , by default None
    test_data_io : MutableMapping[str, DataIoInterface], optional
        Registry to to update or replace the ``DataIoInterface`` registry with.
        , by default None
    test_project_io : MutableMapping[str, ProjectIoInterface], optional
        Registry to to update or replace the ``ProjectIoInterface`` registry with.
        , by default None
    create_new_registry : bool
        Whether to update the actual registry or create a new one from the arguments.
        , by default False

    Yields
    ------
    Generator[None, None, None]
        Just keeps all context manager alive

    See Also
    --------
    monkeypatch_plugin_registry_megacomplex
    monkeypatch_plugin_registry_data_io
    monkeypatch_plugin_registry_project_io
    """
    context_managers = [
        monkeypatch_plugin_registry_megacomplex(test_megacomplex, create_new_registry),
        monkeypatch_plugin_registry_data_io(test_data_io, create_new_registry),
        monkeypatch_plugin_registry_project_io(test_project_io, create_new_registry),
    ]

    with ExitStack() as stack:
        for context_manager in context_managers:
            stack.enter_context(context_manager)
        yield
