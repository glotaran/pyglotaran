"""Data Io registration convenience functions.

Note
----
The [call-arg] type error would be raised since the base methods doesn't have a ``**kwargs``
argument, but we rather ignore this error here, than adding ``**kwargs`` to the base method
and causing an [override] type error in the plugins implementation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.plugin_system.base_registry import __PluginRegistry
from glotaran.plugin_system.base_registry import add_instantiated_plugin_to_registry
from glotaran.plugin_system.base_registry import get_method_from_plugin
from glotaran.plugin_system.base_registry import get_plugin_from_registry
from glotaran.plugin_system.base_registry import is_registered_plugin
from glotaran.plugin_system.base_registry import registered_plugins
from glotaran.plugin_system.base_registry import show_method_help
from glotaran.plugin_system.io_plugin_utils import inferr_file_format
from glotaran.plugin_system.io_plugin_utils import not_implemented_to_value_error
from glotaran.plugin_system.io_plugin_utils import protect_from_overwrite
from glotaran.project import SavingOptions

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Literal

    import xarray as xr

    from glotaran.io.interface import DataIoInterface
    from glotaran.io.interface import DataLoader
    from glotaran.io.interface import DataWriter


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

    def decorator(cls: type[DataIoInterface]) -> type[DataIoInterface]:
        add_instantiated_plugin_to_registry(
            plugin_register_keys=format_names,
            plugin_class=cls,
            plugin_registry=__PluginRegistry.data_io,
        )
        return cls

    return decorator


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


def known_data_formats() -> list[str]:
    """Names of the registered data io plugins.

    Returns
    -------
    list[str]
        List of registered data io plugins.
    """
    return registered_plugins(plugin_registry=__PluginRegistry.data_io)


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
    file_name: str, format_name: str = None, **kwargs: Any
) -> xr.Dataset | xr.DataArray:
    """Read data from a file to :xarraydoc:`Dataset` or :xarraydoc:`DataArray`.

    Parameters
    ----------
    file_name : str
        File containing the data.
    format_name : str
        Format the file is in, if not provided it will be inferred from the file extension.
    **kwargs: Any
        Additional keyword arguments passes to the ``read_dataset`` implementation
        of the data io plugin. If you aren't sure about those use ``get_dataloader``
        to get the implementation with the proper help and autocomplete.

    Returns
    -------
    xr.Dataset|xr.DataArray
        Data loaded from the file.
    """
    io = get_data_io(format_name or inferr_file_format(file_name))
    return io.load_dataset(file_name, **kwargs)  # type: ignore[call-arg]


@not_implemented_to_value_error
def write_dataset(
    file_name: str,
    format_name: str,
    dataset: xr.Dataset | xr.DataArray,
    saving_options: SavingOptions | None = SavingOptions(),
    *,
    allow_overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Write data from :xarraydoc:`Dataset` or :xarraydoc:`DataArray` to a file.

    Parameters
    ----------
    file_name : str
        File to write the data to.
    format_name : str
        Format the file should be in.
    dataset: xr.Dataset|xr.DataArray
        Data to be written to file.
    saving_options: SavingOptions | None
        Options on how to save the data.
    allow_overwrite : bool
        Whether or not to allow overwriting existing files, by default False
    **kwargs: Any
        Additional keyword arguments passes to the ``write_dataset`` implementation
        of the data io plugin. If you aren't sure about those use ``get_datawriter``
        to get the implementation with the proper help and autocomplete.
    """
    protect_from_overwrite(file_name, allow_overwrite=allow_overwrite)
    io = get_data_io(format_name)
    io.write_dataset(  # type: ignore[call-arg]
        file_name=file_name,
        dataset=dataset,
        saving_options=saving_options,
        **kwargs,
    )


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


def get_datawriter(format_name: str) -> DataWriter:
    """Retrieve implementation of the ``write_dataset`` functionality for the format 'format_name'.

    This allows to get the proper help and autocomplete for the function,
    which is especially valuable if the function provides additional options.

    Parameters
    ----------
    format_name : str
        Format the datawriter should be able to write.

    Returns
    -------
    DataWriter
        Function to write :xarraydoc:`Dataset` to the format ``format_name`` .
    """
    io = get_data_io(format_name)
    return get_method_from_plugin(io, "write_dataset")


def show_data_io_method_help(
    format_name: str, method_name: Literal["load_dataset", "write_dataset"]
) -> None:
    """Show help for the implementation of data io plugin methods.

    Parameters
    ----------
    format_name : str
        Format the method should support.
    method_name : {'load_dataset', 'write_dataset'}
        Method name
    """
    io = get_data_io(format_name)
    show_method_help(io, method_name)
