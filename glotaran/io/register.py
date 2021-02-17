from __future__ import annotations

import os

import xarray as xr

from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Result
from glotaran.project import SavingOptions
from glotaran.project import Scheme

from .io import Io

_io_register = {}


def register_io(fmt: str | list[str], io: Io):
    fmt = fmt if isinstance(fmt, list) else [fmt]
    for fmt_name in fmt:
        _io_register[fmt_name] = io()


def known_fmt(fmt: str) -> bool:
    return fmt in _io_register


def get_io(fmt: str):
    if not known_fmt(fmt):
        raise ValueError(f"Unknown format '{fmt}'. Known formats are: {known_fmts()}")
    return _io_register[fmt]


def known_fmts() -> list[str]:
    return [fmt for fmt in _io_register]


def load_model(file_name: str, fmt: str = None) -> Model:
    fmt = _get_fmt_from_file_name(file_name) if fmt is None else fmt
    io = get_io(fmt)
    try:
        return io.read_model(fmt, file_name)
    except NotImplementedError:
        raise ValueError(f"Cannot read models with format '{fmt}'")


def load_scheme(file_name: str, fmt: str = None) -> Scheme:
    fmt = _get_fmt_from_file_name(file_name) if fmt is None else fmt
    io = get_io(fmt)
    try:
        return io.read_scheme(fmt, file_name)
    except NotImplementedError:
        raise ValueError(f"Cannot read scheme with format '{fmt}'")


def write_scheme(file_name: str, fmt: str, scheme: Scheme):
    io = get_io(fmt)
    try:
        return io.write_scheme(fmt, file_name, scheme)
    except NotImplementedError:
        raise ValueError(f"Cannot write dataset with format '{fmt}'")


def load_parameters(file_name: str, fmt: str = None) -> ParameterGroup:
    fmt = _get_fmt_from_file_name(file_name) if fmt is None else fmt
    io = get_io(fmt)
    try:
        return io.read_parameters(fmt, file_name)
    except NotImplementedError:
        raise ValueError(f"Cannot read parameters with format '{fmt}'")


def write_parameters(file_name: str, fmt: str, parameters: ParameterGroup):
    io = get_io(fmt)
    try:
        return io.write_parameters(fmt, file_name, parameters)
    except NotImplementedError:
        raise ValueError(f"Cannot write parameters with format '{fmt}'")


def load_dataset(file_name: str, fmt: str = None) -> xr.Dataset | xr.DataArray:
    fmt = _get_fmt_from_file_name(file_name) if fmt is None else fmt
    io = get_io(fmt)
    try:
        return io.read_dataset(fmt, file_name)
    except NotImplementedError:
        raise ValueError(f"Cannot read dataset with format '{fmt}'")


def write_dataset(
    file_name: str, fmt: str, dataset: xr.Dataset, saving_options: SavingOptions = SavingOptions()
):
    io = get_io(fmt)
    try:
        return io.write_dataset(fmt, file_name, saving_options, dataset)
    except NotImplementedError:
        raise ValueError(f"Cannot write dataset with format '{fmt}'")


def write_result(
    file_name: str, fmt: str, result: Result, saving_options: SavingOptions = SavingOptions()
):
    io = get_io(fmt)
    try:
        return io.write_result(fmt, file_name, saving_options, result)
    except NotImplementedError:
        raise ValueError(f"Cannot write dataset with format '{fmt}'")


def _get_fmt_from_file_name(file_name: str) -> str:
    _, fmt = os.path.splitext(file_name)
    if fmt == "":
        raise ValueError(f"Cannot determine format of modelfile '{file_name}'")
    return fmt[1:]
