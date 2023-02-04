"""Project Io registration convenience functions.

Note
----
The [call-arg] type error would be raised since the base methods doesn't have a ``**kwargs``
argument, but we rather ignore this error here, than adding ``**kwargs`` to the base method
and causing an [override] type error in the plugins implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import TypeVar

from tabulate import tabulate

from glotaran.io.interface import SAVING_OPTIONS_DEFAULT
from glotaran.io.interface import ProjectIoInterface
from glotaran.io.interface import SavingOptions
from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.base_registry import add_instantiated_plugin_to_registry
from glotaran.plugin_system.base_registry import get_method_from_plugin
from glotaran.plugin_system.base_registry import get_plugin_from_registry
from glotaran.plugin_system.base_registry import is_registered_plugin
from glotaran.plugin_system.base_registry import methods_differ_from_baseclass_table
from glotaran.plugin_system.base_registry import registered_plugins
from glotaran.plugin_system.base_registry import set_plugin
from glotaran.plugin_system.base_registry import show_method_help
from glotaran.plugin_system.base_registry import supported_file_extensions
from glotaran.plugin_system.io_plugin_utils import bool_table_repr
from glotaran.plugin_system.io_plugin_utils import infer_file_format
from glotaran.plugin_system.io_plugin_utils import not_implemented_to_value_error
from glotaran.plugin_system.io_plugin_utils import protect_from_overwrite
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from collections.abc import Sequence
    from typing import Any
    from typing import Literal

    from glotaran.model import Model
    from glotaran.parameter import Parameters
    from glotaran.project import Result
    from glotaran.project import Scheme
    from glotaran.typing import StrOrPath

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
            plugin_set_func_name="set_project_plugin",
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


def known_project_formats(full_names: bool = False) -> list[str]:
    """Names of the registered project io plugins.

    Parameters
    ----------
    full_names : bool
        Whether to display the full names the plugins are
        registered under as well.

    Returns
    -------
    list[str]
        List of registered project io plugins.
    """
    return registered_plugins(plugin_registry=__PluginRegistry.project_io, full_names=full_names)


def set_project_plugin(
    format_name: str,
    full_plugin_name: str,
) -> None:
    """Set the plugin used for a specific project format.

    This function is useful when you want to resolve conflicts of installed plugins
    or overwrite the plugin used for a specific format.

    Effected functions:

    - :func:`load_model`
    - :func:`save_model`
    - :func:`load_parameters`
    - :func:`save_parameters`
    - :func:`load_scheme`
    - :func:`save_scheme`
    - :func:`load_result`
    - :func:`save_result`

    Parameters
    ----------
    format_name : str
        Format name used to refer to the plugin when used for ``save`` and ``load`` functions.
    full_plugin_name : str
        Full name (import path) of the registered plugin.
    """
    set_plugin(
        plugin_register_key=format_name,
        full_plugin_name=full_plugin_name,
        plugin_registry=__PluginRegistry.project_io,
    )


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
def load_model(file_name: StrOrPath, format_name: str | None = None, **kwargs: Any) -> Model:
    """Create a Model instance from the specs defined in a file.

    Parameters
    ----------
    file_name : StrOrPath
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
    io = get_project_io(format_name or infer_file_format(file_name))
    model = io.load_model(Path(file_name).as_posix(), **kwargs)
    model.source_path = Path(file_name).as_posix()
    return model


@not_implemented_to_value_error
def save_model(
    model: Model,
    file_name: StrOrPath,
    format_name: str | None = None,
    *,
    allow_overwrite: bool = False,
    update_source_path: bool = True,
    **kwargs: Any,
) -> None:
    """Save a :class:`Model` instance to a spec file.

    Parameters
    ----------
    model: Model
        :class:`Model` instance to save to specs file.
    file_name: StrOrPath
        File to write the model specs to.
    format_name: str | None
        Format the file should be in, if not provided it will be inferred from the file extension.
    allow_overwrite: bool
        Whether or not to allow overwriting existing files, by default False
    update_source_path: bool
        Whether or not to update the ``source_path`` attribute to ``file_name`` when saving.
        by default True
    **kwargs: Any
        Additional keyword arguments passes to the ``save_model`` implementation
        of the project io plugin.
    """
    protect_from_overwrite(file_name, allow_overwrite=allow_overwrite)
    io = get_project_io(format_name or infer_file_format(file_name, needs_to_exist=False))
    io.save_model(file_name=Path(file_name).as_posix(), model=model, **kwargs)
    if update_source_path is True:
        model.source_path = Path(file_name).as_posix()


@not_implemented_to_value_error
def load_parameters(file_name: StrOrPath, format_name: str | None = None, **kwargs) -> Parameters:
    """Create a :class:`Parameters` instance from the specs defined in a file.

    Parameters
    ----------
    file_name: StrOrPath
        File containing the parameter specs.
    format_name: str | None
        Format the file is in, if not provided it will be inferred from the file extension.
    **kwargs: Any
        Additional keyword arguments passes to the ``load_parameters`` implementation
        of the project io plugin.

    Returns
    -------
    Parameters
        :class:`Parameters` instance created from the file.

    .. # noqa: D414
    """
    io = get_project_io(format_name or infer_file_format(file_name))
    parameters = io.load_parameters(
        Path(file_name).as_posix(),
        **kwargs,
    )
    parameters.source_path = Path(file_name).as_posix()
    return parameters


@not_implemented_to_value_error
def save_parameters(
    parameters: Parameters,
    file_name: StrOrPath,
    format_name: str | None = None,
    *,
    allow_overwrite: bool = False,
    update_source_path: bool = True,
    **kwargs: Any,
) -> None:
    """Save a :class:`Parameters` instance to a spec file.

    Parameters
    ----------
    parameters: Parameters
        :class:`Parameters` instance to save to specs file.
    file_name: StrOrPath
        File to write the parameter specs to.
    format_name: str | None
        Format the file should be in, if not provided it will be inferred from the file extension.
    allow_overwrite: bool
        Whether or not to allow overwriting existing files, by default False
    update_source_path: bool
        Whether or not to update the ``source_path`` attribute to ``file_name`` when saving.
        by default True
    **kwargs: Any
        Additional keyword arguments passes to the ``save_parameters`` implementation
        of the project io plugin.
    """
    protect_from_overwrite(file_name, allow_overwrite=allow_overwrite)
    io = get_project_io(format_name or infer_file_format(file_name, needs_to_exist=False))
    io.save_parameters(file_name=Path(file_name).as_posix(), parameters=parameters, **kwargs)
    if update_source_path is True:
        parameters.source_path = Path(file_name).as_posix()


@not_implemented_to_value_error
def load_scheme(file_name: StrOrPath, format_name: str | None = None, **kwargs: Any) -> Scheme:
    """Create a :class:`Scheme` instance from the specs defined in a file.

    Parameters
    ----------
    file_name: StrOrPath
        File containing the parameter specs.
    format_name: str | None
        Format the file is in, if not provided it will be inferred from the file extension.
    **kwargs: Any
        Additional keyword arguments passes to the ``load_scheme`` implementation
        of the project io plugin.

    Returns
    -------
    Scheme
        :class:`Scheme` instance created from the file.
    """
    io = get_project_io(format_name or infer_file_format(file_name))

    scheme = io.load_scheme(Path(file_name).as_posix(), **kwargs)
    scheme.source_path = Path(file_name).as_posix()
    return scheme


@not_implemented_to_value_error
def save_scheme(
    scheme: Scheme,
    file_name: StrOrPath,
    format_name: str | None = None,
    *,
    allow_overwrite: bool = False,
    update_source_path: bool = True,
    **kwargs: Any,
) -> None:
    """Save a :class:`Scheme` instance to a spec file.

    Parameters
    ----------
    scheme: Scheme
        :class:`Scheme` instance to save to specs file.
    file_name: StrOrPath
        File to write the scheme specs to.
    format_name: str | None
        Format the file should be in, if not provided it will be inferred from the file extension.
    allow_overwrite: bool
        Whether or not to allow overwriting existing files, by default False
    update_source_path: bool
        Whether or not to update the ``source_path`` attribute to ``file_name`` when saving.
        by default True
    **kwargs: Any
        Additional keyword arguments passes to the ``save_scheme`` implementation
        of the project io plugin.
    """
    protect_from_overwrite(file_name, allow_overwrite=allow_overwrite)
    io = get_project_io(format_name or infer_file_format(file_name, needs_to_exist=False))
    io.save_scheme(file_name=Path(file_name).as_posix(), scheme=scheme, **kwargs)
    if update_source_path is True:
        scheme.source_path = Path(file_name).as_posix()


@not_implemented_to_value_error
def load_result(result_path: StrOrPath, format_name: str | None = None, **kwargs: Any) -> Result:
    """Create a :class:`Result` instance from the specs defined in a file.

    Parameters
    ----------
    result_path: StrOrPath
        Path containing the result data.
    format_name: str | None
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
    io = get_project_io(format_name or infer_file_format(result_path, allow_folder=True))

    result = io.load_result(Path(result_path).as_posix(), **kwargs)
    result.source_path = Path(result_path).as_posix()
    return result


@not_implemented_to_value_error
def save_result(
    result: Result,
    result_path: StrOrPath,
    format_name: str | None = None,
    *,
    allow_overwrite: bool = False,
    update_source_path: bool = True,
    saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT,
    **kwargs: Any,
) -> list[str]:
    """Write a :class:`Result` instance to a spec file.

    Parameters
    ----------
    result: Result
        :class:`Result` instance to write.
    result_path: StrOrPath
        Path to write the result data to.
    format_name: str | None
        Format the result should be saved in, if not provided and it is a file
        it will be inferred from the file extension.
    allow_overwrite: bool
        Whether or not to allow overwriting existing files, by default False
    update_source_path: bool
        Whether or not to update the ``source_path`` attribute to ``result_path`` when saving.
        by default True
    saving_options: SavingOptions
        Options for the saved result.
    **kwargs: Any
        Additional keyword arguments passes to the ``save_result`` implementation
        of the project io plugin.

    Returns
    -------
    list[str] | None
        List of file paths which were saved.
    """
    protect_from_overwrite(result_path, allow_overwrite=allow_overwrite)
    io = get_project_io(
        format_name or infer_file_format(result_path, needs_to_exist=False, allow_folder=True)
    )
    paths = io.save_result(
        result_path=Path(result_path).as_posix(),
        result=result,
        saving_options=saving_options,
        **kwargs,
    )
    if update_source_path is True:
        result.source_path = Path(result_path).as_posix()
    return paths


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


def project_io_plugin_table(
    *, plugin_names: bool = False, full_names: bool = False
) -> MarkdownStr:
    """Return registered project io plugins and which functions they support as markdown table.

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
        Markdown table of project io plugins.
    """
    table_data = methods_differ_from_baseclass_table(
        PROJECT_IO_METHODS,
        known_project_formats(full_names=full_names),
        get_project_io,
        ProjectIoInterface,
        plugin_names=plugin_names,
    )
    header_values = ["Format name", *PROJECT_IO_METHODS]
    if plugin_names:
        header_values.append("Plugin name")
    headers = tuple(f"__{x}__" for x in header_values)
    return MarkdownStr(
        tabulate(
            bool_table_repr(table_data), tablefmt="github", headers=headers, stralign="center"
        )
    )


def supported_file_extensions_project_io(
    method_names: str | Sequence[str],
) -> Generator[str, None, None]:
    """Get project io formats that support all methods in ``method_names``.

    Parameters
    ----------
    method_names: str | Sequence[str]
        Names of Methods that need to support the file extension.

    Yields
    ------
    Generator[str, None, None]
        File extension supported by all methods in ``method_names``.

    See Also
    --------
    supported_file_extensions
    PROJECT_IO_METHODS
    """
    yield from supported_file_extensions(
        method_names,
        known_project_formats(),
        get_project_io,
        ProjectIoInterface,
    )
