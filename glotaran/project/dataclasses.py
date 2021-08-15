"""Contains helper methods for dataclasses."""
from __future__ import annotations

import dataclasses
from typing import Any


def serialize_to_file_name_field(
    file_name: str, default: Any = dataclasses.MISSING
) -> dataclasses.Field:
    """Create a dataclass field with file_name as metadata.

    The field will be replace with the file_name as value. within
    :function:``glotaran.project.dataclasses.asdict``.

    Parameters
    ----------
    file_name : str
        The file_name with which the field gets replaced in asdict.
    default : Any
        The default value of the field.

    Returns
    -------
    dataclasses.Field
        The created field.
    """
    return dataclasses.field(default=default, metadata={"file_name": file_name})


def serialize_to_file_name_dict_field(
    file_suffix: str, default: Any = dataclasses.MISSING
) -> dataclasses.Field:
    """Create a dataclass field with file_name as metadata.

    The field will be replace with the file_name as value. within
    :function:``glotaran.project.dataclasses.asdict``.

    Parameters
    ----------
    file_name : str
        The file_name with which the field gets replaced in asdict.
    default : Any
        The default value of the field.

    Returns
    -------
    dataclasses.Field
        The created field.
    """
    return dataclasses.field(default=default, metadata={"file_suffix": file_suffix})


def asdict(dataclass: object) -> dict[str, Any]:
    """Create a dictinory from a dataclass.

    A wrappper for ``dataclasses.asdict`` which recognizes fields created
    with :function:``glotaran.project.dataclasses.serialize_to_file_name_field``.

    Parameters
    ----------
    dataclass : object
        A dataclass instance.

    Returns
    -------
    dict[str, Any]
        The dataclass represented as a dictionary.
    """
    fields = dataclasses.fields(dataclass)

    def dict_factory(values: list):
        for i, field in enumerate(fields):
            if "file_name" in field.metadata:
                values[i] = (field.name, field.metadata["file_name"])
            elif "file_suffix" in field.metadata:
                file_suffix = field.metadata["file_name"]
                values[i] = (field.name, {key: f"{key}.{file_suffix}" for key in values[i][1]})
        return dict(values)

    return dataclasses.asdict(dataclass, dict_factory=dict_factory)
