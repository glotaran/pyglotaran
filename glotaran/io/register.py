"""A register for models"""
from __future__ import annotations

import os

from glotaran.parameter import ParameterGroup

from .io import Io

_io_register = {}


def register_io(fmt: str | list(str), io: Io):
    fmt = fmt if isinstance(fmt, list) else [fmt]
    for fmt_name in fmt:
        _io_register[fmt_name] = io


def known_fmt(fmt: str) -> bool:
    return fmt in _io_register


def get_io(fmt: str):
    if not known_fmt(fmt):
        raise ValueError(f"Unknown format '{fmt}'. Known formats are: {known_fmts()}")
    return _io_register[fmt]


def known_fmts() -> list[str]:
    return [fmt for fmt in _io_register]


def load_model(file_name: str, fmt: str = None):
    fmt = _get_fmt_from_file_name(file_name) if fmt is None else fmt
    io = get_io(fmt)
    try:
        return io.read_model(fmt, file_name)
    except NotImplementedError:
        raise ValueError(f"Cannot read models with format '{fmt}'")


def load_parameters(file_name: str, fmt: str = None):
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
        raise ValueError(f"Cannot read parameters with format '{fmt}'")


def _get_fmt_from_file_name(file_name: str) -> str:
    _, fmt = os.path.splitext(file_name)
    if fmt == "":
        raise ValueError(f"Cannot determine format of modelfile '{file_name}'")
    return fmt[1:]
