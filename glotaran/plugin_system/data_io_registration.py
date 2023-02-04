"""Data Io registration convenience functions.

Note
----
The [call-arg] type error would be raised since the base methods doesn't have a ``**kwargs``
argument, but we rather ignore this error here, than adding ``**kwargs`` to the base method
and causing an [override] type error in the plugins implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from tabulate import tabulate

from glotaran.io.interface import DataIoInterface
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

    from glotaran.io.interface import DataLoader
    from glotaran.io.interface import DataSaver
    from glotaran.typing import StrOrPath

DATA_IO_METHODS = ("load_dataset", "save_dataset")
"""Methods used by DataIoInterface plugins."""


def register_data_io(
    format_names: str | list[str],
) -> Callable[[type[DataIoInterface]], type[DataIoInterface]]:
    """Register data io plugins to one or more formats.

    Decorate a data io plugin class with ``@register_data_io(format_name|[*format_names])``
    to add it to the registry.

    Parameters
    ----------
    format_names : str | list[str]
        Name of the data io plugin under which it is registered.

    Returns
    -------
    Callable[[type[DataIoInterface]], type[DataIoInterface]]
        Inner decorator function.

    Examples
    --------
    >>> @register_data_io("my_format_1")
    ... class MyDataIo1(DataIoInterface):
    ...     pass

    >>> @register_data_io(["my_format_1", "my_format_1_alias"])
    ... class MyDataIo2(DataIoInterface):
    ...     pass
    """

    def wrapper(cls: type[DataIoInterface]) -> type[DataIoInterface]:
        add_instantiated_plugin_to_registry(
            plugin_register_keys=format_names,
            plugin_class=cls,
            plugin_registry=__PluginRegistry.data_io,
            plugin_set_func_name="set_data_plugin",
        )
        return cls

    return wrapper


def is_known_data_format(format_name: str) -> bool:
    """Check if a data format is in the data_io registry.

    Parameters
    ----------
    format_name : str
        Name of the data io plugin under which it is registered.

    Returns
    -------
    bool
        Whether or not the data format is a registered data io plugins.
    """
    return is_registered_plugin(
        plugin_register_key=format_name, plugin_registry=__PluginRegistry.data_io
    )


def known_data_formats(full_names: bool = False) -> list[str]:
    """Names of the registered data io plugins.

    Parameters
    ----------
    full_names : bool
        Whether to display the full names the plugins are
        registered under as well.

    Returns
    -------
    list[str]
        List of registered data io plugins.
    """
    return registered_plugins(plugin_registry=__PluginRegistry.data_io, full_names=full_names)


def set_data_plugin(
    format_name: str,
    full_plugin_name: str,
) -> None:
    """Set the plugin used for a specific data format.

    This function is useful when you want to resolve conflicts of installed plugins
    or overwrite the plugin used for a specific format.

    Effected functions:

    - :func:`load_dataset`
    - :func:`save_dataset`

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
        plugin_registry=__PluginRegistry.data_io,
    )


def get_data_io(format_name: str) -> DataIoInterface:
    """Retrieve a data io plugin from the data_io registry.

    Parameters
    ----------
    format_name : str
        Name of the data io plugin under which it is registered.

    Returns
    -------
    DataIoInterface
        Data io plugin instance.
    """
    return get_plugin_from_registry(
        plugin_register_key=format_name,
        plugin_registry=__PluginRegistry.data_io,
        not_found_error_message=(
            f"Unknown  Data Io format {format_name!r}. Known formats are: {known_data_formats()}"
        ),
    )


@not_implemented_to_value_error
def load_dataset(
    file_name: StrOrPath, format_name: str | None = None, **kwargs: Any
) -> xr.Dataset:
    """Read data from a file to :xarraydoc:`Dataset` or :xarraydoc:`DataArray`.

    Parameters
    ----------
    file_name : StrOrPath
        File containing the data.
    format_name : str
        Format the file is in, if not provided it will be inferred from the file extension.
    **kwargs : Any
        Additional keyword arguments passes to the ``read_dataset`` implementation
        of the data io plugin. If you aren't sure about those use ``get_dataloader``
        to get the implementation with the proper help and autocomplete.

    Returns
    -------
    xr.Dataset
        Data loaded from the file.
    """
    io = get_data_io(format_name or infer_file_format(file_name))
    dataset = io.load_dataset(Path(file_name).as_posix(), **kwargs)

    if isinstance(dataset, xr.DataArray):
        dataset = dataset.to_dataset(name="data")
    dataset.attrs["loader"] = load_dataset
    dataset.attrs["source_path"] = Path(file_name).as_posix()
    return dataset


@not_implemented_to_value_error
def save_dataset(
    dataset: xr.Dataset | xr.DataArray,
    file_name: StrOrPath,
    format_name: str | None = None,
    *,
    data_filters: list[str] | None = None,
    allow_overwrite: bool = False,
    update_source_path: bool = True,
    **kwargs: Any,
) -> None:
    """Save data from :xarraydoc:`Dataset` or :xarraydoc:`DataArray` to a file.

    Parameters
    ----------
    dataset : xr.Dataset | xr.DataArray
        Data to be written to file.
    file_name : StrOrPath
        File to write the data to.
    format_name : str
        Format the file should be in, if not provided it will be inferred from the file extension.
    data_filters : list[str] | None
        Optional list of items in the dataset to be saved.
    allow_overwrite : bool
        Whether or not to allow overwriting existing files, by default False
    update_source_path: bool
        Whether or not to update the ``source_path`` attribute to ``file_name`` when saving.
        by default True
    **kwargs : Any
        Additional keyword arguments passes to the ``write_dataset`` implementation
        of the data io plugin. If you aren't sure about those use ``get_datasaver``
        to get the implementation with the proper help and autocomplete.
    """
    protect_from_overwrite(file_name, allow_overwrite=allow_overwrite)
    io = get_data_io(format_name or infer_file_format(file_name, needs_to_exist=False))
    if "loader" in dataset.attrs:
        del dataset.attrs["loader"]
    io.save_dataset(file_name=Path(file_name).as_posix(), dataset=dataset, **kwargs)
    dataset.attrs["loader"] = load_dataset
    if update_source_path is True or "source_path" not in dataset.attrs:
        dataset.attrs["source_path"] = Path(file_name).as_posix()


def get_dataloader(format_name: str) -> DataLoader:
    """Retrieve implementation of the ``read_dataset`` functionality for the format 'format_name'.

    This allows to get the proper help and autocomplete for the function,
    which is especially valuable if the function provides additional options.

    Parameters
    ----------
    format_name : str
        Format the dataloader should be able to read.

    Returns
    -------
    DataLoader
        Function to load data of format ``format_name`` as
        :xarraydoc:`Dataset` or :xarraydoc:`DataArray`.
    """
    io = get_data_io(format_name)
    return get_method_from_plugin(io, "load_dataset")


def get_datasaver(format_name: str) -> DataSaver:
    """Retrieve implementation of the ``save_dataset`` functionality for the format 'format_name'.

    This allows to get the proper help and autocomplete for the function,
    which is especially valuable if the function provides additional options.

    Parameters
    ----------
    format_name : str
        Format the datawriter should be able to write.

    Returns
    -------
    DataSaver
        Function to write :xarraydoc:`Dataset` to the format ``format_name`` .
    """
    io = get_data_io(format_name)
    return get_method_from_plugin(io, "save_dataset")


def show_data_io_method_help(
    format_name: str, method_name: Literal["load_dataset", "save_dataset"]
) -> None:
    """Show help for the implementation of data io plugin methods.

    Parameters
    ----------
    format_name : str
        Format the method should support.
    method_name : {'load_dataset', 'save_dataset'}
        Method name
    """
    io = get_data_io(format_name)
    show_method_help(io, method_name)


def data_io_plugin_table(*, plugin_names: bool = False, full_names: bool = False) -> MarkdownStr:
    """Return registered data io plugins and which functions they support as markdown table.

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
        Markdown table of data io plugins.
    """
    table_data = methods_differ_from_baseclass_table(
        DATA_IO_METHODS,
        known_data_formats(full_names=full_names),
        get_data_io,
        DataIoInterface,
        plugin_names=plugin_names,
    )
    header_values = ["Format name", *DATA_IO_METHODS]
    if plugin_names:
        header_values.append("Plugin name")
    headers = tuple(f"__{x}__" for x in header_values)
    return MarkdownStr(
        tabulate(
            bool_table_repr(table_data), tablefmt="github", headers=headers, stralign="center"
        )
    )


def supported_file_extensions_data_io(
    method_names: str | Sequence[str],
) -> Generator[str, None, None]:
    """Get data io formats that support all methods in ``method_names``.

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
    DATA_IO_METHODS
    """
    yield from supported_file_extensions(
        method_names,
        known_data_formats(),
        get_data_io,
        DataIoInterface,
    )
