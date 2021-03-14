"""Project Io registration convenience functions.

Note:
-----
The [call-arg] type error would be raised since the base methods doesn't have a **kwargs argument,
but we rather ignore this error here, than adding **kwargs to the base method and
causing an [override] type error in the plugins implementation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.base_registry import add_instantiated_plugin_to_registry
from glotaran.plugin_system.base_registry import get_plugin_from_registry
from glotaran.plugin_system.base_registry import inferr_file_format
from glotaran.plugin_system.base_registry import is_registered_plugin
from glotaran.plugin_system.base_registry import not_implemented_to_value_error
from glotaran.plugin_system.base_registry import registered_plugins
from glotaran.project import SavingOptions

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable

    from glotaran.io.interface import ProjectIoInterface
    from glotaran.model import Model
    from glotaran.parameter import ParameterGroup
    from glotaran.project import Result
    from glotaran.project import Scheme


def register_project_io(
    format_names: str | list[str],
) -> Callable[[type[ProjectIoInterface]], type[ProjectIoInterface]]:
    """Register project io plugins to one or more formats.

    Decorate a project io plugin class with ``@register_project_io(format_name|[*format_names])``
    to add it to the registry.

    Parameters
    ----------
    format_names : str | list[str]
        Name of the project io plugin under which it is registered.

    Returns
    -------
    Callable[[type[ProjectIoInterface]], type[ProjectIoInterface]]
        Inner decorator function.

    Examples
    --------
    >>> @register_project_io("my_format_1")
    ... class MyProjectIo1(ProjectIoInterface):
    ...     pass

    >>> @register_project_io(["my_format_1", "my_format_1_alias"])
    ... class MyProjectIo2(ProjectIoInterface):
    ...     pass
    """

    def decorator(cls: type[ProjectIoInterface]) -> type[ProjectIoInterface]:
        add_instantiated_plugin_to_registry(
            plugin_register_keys=format_names,
            plugin_class=cls,
            plugin_registry=__PluginRegistry.project_io,
        )
        return cls

    return decorator


def is_known_project_format(format_name: str) -> bool:
    """Check if a data format is in the project_io registry.

    Parameters
    ----------
    format_name : str
        Name of the project io plugin under which it is registered.

    Returns
    -------
    bool
        Whether or not the data format is a registered project io plugin.
    """
    return is_registered_plugin(
        plugin_register_key=format_name, plugin_registry=__PluginRegistry.project_io
    )


def known_project_formats() -> list[str]:
    """Names of the registered project io plugins.

    Returns
    -------
    list[str]
        List of registered project io plugins.
    """
    return registered_plugins(plugin_registry=__PluginRegistry.project_io)


def get_project_io(format_name: str) -> ProjectIoInterface:
    """Retrieve a data io plugin from the project_io registry.

    Parameters
    ----------
    format_name : str
        Name of the data io plugin under which it is registered.

    Returns
    -------
    ProjectIoInterface
        Project io plugin instance.
    """
    return get_plugin_from_registry(
        plugin_register_key=format_name,
        plugin_registry=__PluginRegistry.project_io,
        not_found_error_message=(
            f"Unknown Project Io format {format_name!r}. "
            f"Known formats are: {known_project_formats()}"
        ),
    )


@not_implemented_to_value_error
def load_model(file_name: str, format_name: str = None, **kwargs: Any) -> Model:
    """Create a Model instance from the specs defined in a file.

    Parameters
    ----------
    file_name : str
        File containing the model specs.
    format_name : str
        Format the file is in, if not provided it will be inferred from the file extension.
    **kwargs: Any
        Additional keyword arguments passes to the ``read_model`` implementation
        of the project io plugin.

    Returns
    -------
    Model
        Model instance created from the file.
    """
    io = get_project_io(format_name or inferr_file_format(file_name))
    return io.read_model(file_name, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def write_model(file_name: str, format_name: str, model: Model, **kwargs: Any) -> None:
    """Write a :class:`Model` instance to a spec file.

    Parameters
    ----------
    file_name : str
        File to write the model specs to.
    format_name : str
        Format the file should be in.
    model: Model
        :class:`Model` instance to write to specs file.
    **kwargs: Any
        Additional keyword arguments passes to the ``write_model`` implementation
        of the project io plugin.
    """
    io = get_project_io(format_name)
    io.write_model(file_name=file_name, model=model, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def load_parameters(file_name: str, format_name: str = None, **kwargs) -> ParameterGroup:
    """Create a :class:`ParameterGroup` instance from the specs defined in a file.

    Parameters
    ----------
    file_name : str
        File containing the parameter specs.
    format_name : str
        Format the file is in, if not provided it will be inferred from the file extension.
    **kwargs: Any
        Additional keyword arguments passes to the ``read_parameters`` implementation
        of the project io plugin.

    Returns
    -------
    ParameterGroup
        :class:`ParameterGroup` instance created from the file.
    """
    io = get_project_io(format_name or inferr_file_format(file_name))
    return io.read_parameters(file_name, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def write_parameters(
    file_name: str, format_name: str, parameters: ParameterGroup, **kwargs: Any
) -> None:
    """Write a :class:`ParameterGroup` instance to a spec file.

    Parameters
    ----------
    file_name : str
        File to write the parameter specs to.
    format_name : str
        Format the file should be in.
    parameters : ParameterGroup
        :class:`ParameterGroup` instance to write to specs file.
    **kwargs: Any
        Additional keyword arguments passes to the ``write_parameters`` implementation
        of the project io plugin.
    """
    io = get_project_io(format_name)
    io.write_parameters(  # type: ignore[call-arg]
        file_name=file_name,
        parameters=parameters,
        **kwargs,
    )


@not_implemented_to_value_error
def load_scheme(file_name: str, format_name: str = None, **kwargs: Any) -> Scheme:
    """Create a :class:`Scheme` instance from the specs defined in a file.

    Parameters
    ----------
    file_name : str
        File containing the parameter specs.
    format_name : str
        Format the file is in, if not provided it will be inferred from the file extension.
    **kwargs: Any
        Additional keyword arguments passes to the ``read_scheme`` implementation
        of the project io plugin.

    Returns
    -------
    Scheme
        :class:`Scheme` instance created from the file.
    """
    io = get_project_io(format_name or inferr_file_format(file_name))
    return io.read_scheme(file_name, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def write_scheme(file_name: str, format_name: str, scheme: Scheme, **kwargs: Any) -> None:
    """Write a :class:`Scheme` instance to a spec file.

    Parameters
    ----------
    file_name : str
        File to write the scheme specs to.
    format_name : str
        Format the file should be in.
    scheme : Scheme
        :class:`Scheme` instance to write to specs file.
    **kwargs: Any
        Additional keyword arguments passes to the ``write_scheme`` implementation
        of the project io plugin.
    """
    io = get_project_io(format_name)
    io.write_scheme(file_name=file_name, scheme=scheme, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def load_result(result_path: str, format_name: str, **kwargs: Any) -> Result:
    """Create a :class:`Result` instance from the specs defined in a file.

    Parameters
    ----------
    result_path : str
        Path containing the result data.
    format_name : str
        Format the result should be saved in, if not provided and it is a file
        it will be inferred from the file extension.
    **kwargs: Any
        Additional keyword arguments passes to the ``read_result`` implementation
        of the project io plugin.

    Returns
    -------
    Result
        :class:`Result` instance created from the saved format.
    """
    io = get_project_io(format_name or inferr_file_format(result_path))
    return io.read_result(result_path, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def write_result(
    result_path: str,
    format_name: str,
    result: Result,
    saving_options: SavingOptions = SavingOptions(),
    **kwargs: Any,
) -> None:
    """Write a :class:`Result` instance to a spec file.

    Parameters
    ----------
    result_path : str
        Path to write the result data to.
    format_name : str
        Format the result should be saved in.
    saving_options : SavingOptions
        Options on how to save the result.
    result : Result
        :class:`Result` instance to write.
    **kwargs: Any
        Additional keyword arguments passes to the ``write_result`` implementation
        of the project io plugin.
    """
    io = get_project_io(format_name)
    io.write_result(  # type: ignore[call-arg]
        result_path=result_path,
        result=result,
        saving_options=saving_options,
        **kwargs,
    )
