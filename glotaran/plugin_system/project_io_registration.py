"""Project Io registration convenience functions.

Note
----
The [call-arg] type error would be raised since the base methods doesn't have a ``**kwargs``
argument, but we rather ignore this error here, than adding ``**kwargs`` to the base method
and causing an [override] type error in the plugins implementation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

from tabulate import tabulate

from glotaran.io.interface import ProjectIoInterface
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.base_registry import add_instantiated_plugin_to_registry
from glotaran.plugin_system.base_registry import get_method_from_plugin
from glotaran.plugin_system.base_registry import get_plugin_from_registry
from glotaran.plugin_system.base_registry import is_registered_plugin
from glotaran.plugin_system.base_registry import methods_differ_from_baseclass_table
from glotaran.plugin_system.base_registry import registered_plugins
from glotaran.plugin_system.base_registry import show_method_help
from glotaran.plugin_system.io_plugin_utils import bool_table_repr
from glotaran.plugin_system.io_plugin_utils import inferr_file_format
from glotaran.plugin_system.io_plugin_utils import not_implemented_to_value_error
from glotaran.plugin_system.io_plugin_utils import protect_from_overwrite

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Literal

    from glotaran.model import Model
    from glotaran.parameter import ParameterGroup
    from glotaran.project import Result
    from glotaran.project import Scheme

    ProjectIoMethods = TypeVar(
        "ProjectIoMethods",
        Literal["load_model"],
        Literal["save_model"],
        Literal["load_parameters"],
        Literal["save_parameters"],
        Literal["load_scheme"],
        Literal["save_scheme"],
        Literal["load_result"],
        Literal["save_result"],
    )


PROJECT_IO_METHODS = (
    "load_model",
    "save_model",
    "load_parameters",
    "save_parameters",
    "load_scheme",
    "save_scheme",
    "load_result",
    "save_result",
)


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

    def wrapper(cls: type[ProjectIoInterface]) -> type[ProjectIoInterface]:
        add_instantiated_plugin_to_registry(
            plugin_register_keys=format_names,
            plugin_class=cls,
            plugin_registry=__PluginRegistry.project_io,
        )
        return cls

    return wrapper


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
        Additional keyword arguments passes to the ``load_model`` implementation
        of the project io plugin.

    Returns
    -------
    Model
        Model instance created from the file.
    """
    io = get_project_io(format_name or inferr_file_format(file_name))
    return io.load_model(file_name, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def save_model(
    file_name: str,
    model: Model,
    format_name: str = None,
    *,
    allow_overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Save a :class:`Model` instance to a spec file.

    Parameters
    ----------
    file_name : str
        File to write the model specs to.
    model: Model
        :class:`Model` instance to save to specs file.
    format_name : str
        Format the file should be in, if not provided it will be inferred from the file extension.
    allow_overwrite : bool
        Whether or not to allow overwriting existing files, by default False
    **kwargs: Any
        Additional keyword arguments passes to the ``save_model`` implementation
        of the project io plugin.
    """
    protect_from_overwrite(file_name, allow_overwrite=allow_overwrite)
    io = get_project_io(format_name or inferr_file_format(file_name, needs_to_exist=False))
    io.save_model(file_name=file_name, model=model, **kwargs)  # type: ignore[call-arg]


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
        Additional keyword arguments passes to the ``load_parameters`` implementation
        of the project io plugin.

    Returns
    -------
    ParameterGroup
        :class:`ParameterGroup` instance created from the file.
    """
    io = get_project_io(format_name or inferr_file_format(file_name))
    return io.load_parameters(file_name, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def save_parameters(
    file_name: str,
    parameters: ParameterGroup,
    format_name: str = None,
    *,
    allow_overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Save a :class:`ParameterGroup` instance to a spec file.

    Parameters
    ----------
    file_name : str
        File to write the parameter specs to.
    parameters : ParameterGroup
        :class:`ParameterGroup` instance to save to specs file.
    format_name : str
        Format the file should be in, if not provided it will be inferred from the file extension.
    allow_overwrite : bool
        Whether or not to allow overwriting existing files, by default False
    **kwargs: Any
        Additional keyword arguments passes to the ``save_parameters`` implementation
        of the project io plugin.
    """
    protect_from_overwrite(file_name, allow_overwrite=allow_overwrite)
    io = get_project_io(format_name or inferr_file_format(file_name, needs_to_exist=False))
    io.save_parameters(  # type: ignore[call-arg]
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
        Additional keyword arguments passes to the ``load_scheme`` implementation
        of the project io plugin.

    Returns
    -------
    Scheme
        :class:`Scheme` instance created from the file.
    """
    io = get_project_io(format_name or inferr_file_format(file_name))
    return io.load_scheme(file_name, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def save_scheme(
    file_name: str,
    scheme: Scheme,
    format_name: str = None,
    *,
    allow_overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Save a :class:`Scheme` instance to a spec file.

    Parameters
    ----------
    file_name : str
        File to write the scheme specs to.
    scheme : Scheme
        :class:`Scheme` instance to save to specs file.
    format_name : str
        Format the file should be in, if not provided it will be inferred from the file extension.
    allow_overwrite : bool
        Whether or not to allow overwriting existing files, by default False
    **kwargs: Any
        Additional keyword arguments passes to the ``save_scheme`` implementation
        of the project io plugin.
    """
    protect_from_overwrite(file_name, allow_overwrite=allow_overwrite)
    io = get_project_io(format_name or inferr_file_format(file_name, needs_to_exist=False))
    io.save_scheme(file_name=file_name, scheme=scheme, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def load_result(result_path: str, format_name: str = None, **kwargs: Any) -> Result:
    """Create a :class:`Result` instance from the specs defined in a file.

    Parameters
    ----------
    result_path : str
        Path containing the result data.
    format_name : str
        Format the result is in, if not provided and it is a file
        it will be inferred from the file extension.
    **kwargs: Any
        Additional keyword arguments passes to the ``load_result`` implementation
        of the project io plugin.

    Returns
    -------
    Result
        :class:`Result` instance created from the saved format.
    """
    io = get_project_io(format_name or inferr_file_format(result_path))
    return io.load_result(result_path, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def save_result(
    result_path: str,
    result: Result,
    format_name: str = None,
    *,
    allow_overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Write a :class:`Result` instance to a spec file.

    Parameters
    ----------
    result_path : str
        Path to write the result data to.
    result : Result
        :class:`Result` instance to write.
    format_name : str
        Format the result should be saved in, if not provided and it is a file
        it will be inferred from the file extension.
    allow_overwrite : bool
        Whether or not to allow overwriting existing files, by default False
    **kwargs: Any
        Additional keyword arguments passes to the ``save_result`` implementation
        of the project io plugin.
    """
    protect_from_overwrite(result_path, allow_overwrite=allow_overwrite)
    io = get_project_io(format_name or inferr_file_format(result_path, needs_to_exist=False))
    io.save_result(  # type: ignore[call-arg]
        result_path=result_path,
        result=result,
        **kwargs,
    )


def get_project_io_method(format_name: str, method_name: ProjectIoMethods) -> Callable[..., Any]:
    """Retrieve implementation of project io functionality for the format 'format_name'.

    This allows to get the proper help and autocomplete for the function,
    which is especially valuable if the function provides additional options.

    Parameters
    ----------
    format_name : str
        Format the dataloader should be able to read.
    method_name : {'load_model', 'write_model', 'load_parameters', 'write_parameters',\
    'load_scheme', 'write_scheme', 'load_result', 'write_result'}
        Method name, e.g. load_model.

    Returns
    -------
    Callable[..., Any]
        The function which is called in the background by the convenience functions.


    .. # noqa: DAR103 method_name
    """
    io = get_project_io(format_name)
    return get_method_from_plugin(io, method_name)


def show_project_io_method_help(format_name: str, method_name: ProjectIoMethods) -> None:
    """Show help for the implementation of project io plugin methods.

    Parameters
    ----------
    format_name : str
        Format the method should support.
    method_name : {'load_model', 'write_model', 'load_parameters', 'write_parameters',\
    'load_scheme', 'write_scheme', 'load_result', 'write_result'}
        Method name.


    .. # noqa: DAR103 method_name
    .. # noqa: DAR101
    """
    io = get_project_io(format_name)
    show_method_help(io, method_name)


def project_io_plugin_table() -> str:
    """Return registered project io plugins and which functions they support as markdown table.

    This is especially useful when you work with new plugins.

    Returns
    -------
    str
        Markdown table of project io plugins.
    """
    table_data = methods_differ_from_baseclass_table(
        PROJECT_IO_METHODS, known_project_formats(), get_project_io, ProjectIoInterface
    )
    headers = tuple(map(lambda x: f"__{x}__", ["Plugin", *PROJECT_IO_METHODS]))
    return tabulate(
        bool_table_repr(table_data), tablefmt="github", headers=headers, stralign="center"
    )
