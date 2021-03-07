from __future__ import annotations

import os
from typing import TYPE_CHECKING

import xarray as xr

from glotaran.project import SavingOptions

if TYPE_CHECKING:
    from glotaran.io.interface import DataIoInterface
    from glotaran.io.interface import DataLoader
    from glotaran.io.interface import DataWriter
    from glotaran.io.interface import ProjectIoInterface
    from glotaran.model import Model
    from glotaran.parameter import ParameterGroup
    from glotaran.project import Result
    from glotaran.project import Scheme

__data_io_register: dict[str, DataIoInterface] = {}

__project_io_register: dict[str, ProjectIoInterface] = {}


def _register_data_io(fmt: str | list[str], io: type[DataIoInterface]):
    fmt = fmt if isinstance(fmt, list) else [fmt]
    for fmt_name in fmt:
        __data_io_register[fmt_name] = io(fmt_name)


def known_data_fmt(fmt: str) -> bool:
    return fmt in __data_io_register


def get_data_io(fmt: str) -> DataIoInterface:
    if not known_data_fmt(fmt):
        raise ValueError(f"Unknown format '{fmt}'. Known formats are: {known_data_fmts()}")
    return __data_io_register[fmt]


def known_data_fmts() -> list[str]:
    return [fmt for fmt in __data_io_register]


def _register_project_io(fmt: str | list[str], io: type[ProjectIoInterface]):
    fmt = fmt if isinstance(fmt, list) else [fmt]
    for fmt_name in fmt:
        __project_io_register[fmt_name] = io(fmt_name)


def known_project_fmt(fmt: str) -> bool:
    return fmt in __project_io_register


def get_project_io(fmt: str) -> ProjectIoInterface:
    if not known_project_fmt(fmt):
        raise ValueError(f"Unknown format '{fmt}'. Known formats are: {known_data_fmts()}")
    return __project_io_register[fmt]


def known_project_fmts() -> list[str]:
    return [fmt for fmt in __project_io_register]


def load_model(file_name: str, fmt: str = None) -> Model:
    fmt = _get_fmt_from_file_name(file_name) if fmt is None else fmt
    io = get_project_io(fmt)
    try:
        return io.read_model(file_name)
    except NotImplementedError:
        raise ValueError(f"Cannot read models with format '{fmt}'")


def load_scheme(file_name: str, fmt: str = None) -> Scheme:
    fmt = _get_fmt_from_file_name(file_name) if fmt is None else fmt
    io = get_project_io(fmt)
    try:
        return io.read_scheme(file_name)
    except NotImplementedError:
        raise ValueError(f"Cannot read scheme with format '{fmt}'")


def write_scheme(file_name: str, fmt: str, scheme: Scheme):
    io = get_project_io(fmt)
    try:
        return io.write_scheme(file_name, scheme)
    except NotImplementedError:
        raise ValueError(f"Cannot write scheme with format '{fmt}'")


def load_parameters(file_name: str, fmt: str = None) -> ParameterGroup:
    fmt = _get_fmt_from_file_name(file_name) if fmt is None else fmt
    io = get_project_io(fmt)
    try:
        return io.read_parameters(file_name)
    except NotImplementedError:
        raise ValueError(f"Cannot read parameters with format '{fmt}'")


def write_parameters(file_name: str, fmt: str, parameters: ParameterGroup):
    io = get_project_io(fmt)
    try:
        return io.write_parameters(file_name, parameters)
    except NotImplementedError:
        raise ValueError(f"Cannot write parameters with format '{fmt}'")


def load_dataset(file_name: str, fmt: str = None) -> xr.Dataset | xr.DataArray:
    fmt = _get_fmt_from_file_name(file_name) if fmt is None else fmt
    io = get_data_io(fmt)
    try:
        return io.read_dataset(file_name)
    except NotImplementedError:
        raise ValueError(f"Cannot read dataset with format '{fmt}'")


def write_dataset(
    file_name: str, fmt: str, dataset: xr.Dataset, saving_options: SavingOptions = SavingOptions()
):
    io = get_data_io(fmt)
    try:
        return io.write_dataset(file_name, saving_options, dataset)
    except NotImplementedError:
        raise ValueError(f"Cannot write dataset with format '{fmt}'")


def get_dataloader(fmt: str) -> DataLoader:
    """Retrieve implementation of the read_dataset functionalityfor the format 'fmt'.

    This allows to get the proper help and autocomplete for the function,
    which is especially valuable if the function provides additional options.

    Returns
    -------
    DataLoader
        Function to load data of format 'fmt' as :xarraydoc:`Dataset` or :xarraydoc:`DataArray`.
    """
    io = get_data_io(fmt)
    return io.get_dataloader()


def get_datawriter(fmt: str) -> DataWriter:
    """Retrieve implementation of the write_dataset functionality for the format 'fmt'.

    This allows to get the proper help and autocomplete for the function,
    which is especially valuable if the function provides additional options.

    Returns
    -------
    DataWriter
        Function to write :xarraydoc:`Dataset` to the format 'fmt' .
    """
    io = get_data_io(fmt)
    return io.get_datawriter()


def write_result(
    file_name: str, fmt: str, result: Result, saving_options: SavingOptions = SavingOptions()
):
    io = get_project_io(fmt)
    try:
        return io.write_result(file_name, saving_options, result)
    except NotImplementedError:
        raise ValueError(f"Cannot write dataset with format '{fmt}'")


def _get_fmt_from_file_name(file_name: str) -> str:
    _, fmt = os.path.splitext(file_name)
    if fmt == "":
        raise ValueError(f"Cannot determine format of modelfile '{file_name}'")
    return fmt[1:]
