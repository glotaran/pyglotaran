"""Contains helper methods for dataclasses."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import TypeVar

    DefaultType = TypeVar("DefaultType")


def exclude_from_dict_field(
    default: DefaultType = dataclasses.MISSING,  # type:ignore[assignment]
) -> DefaultType:
    """Create a dataclass field with which will be excluded from ``asdict``.

    Parameters
    ----------
    default : DefaultType
        The default value of the field.

    Returns
    -------
    DefaultType
        The created field.
    """
    return dataclasses.field(default=default, metadata={"exclude_from_dict": True})


def file_representation_field(
    target: str,
    loader: Callable[[str], Any],
    default: DefaultType = dataclasses.MISSING,  # type:ignore[assignment]
) -> DefaultType:
    """Create a dataclass field with target and loader as metadata.

    Parameters
    ----------
    target : str
        The name of the represented field.
    loader : Callable[[str], Any]
        A function to load the target field from a file.
    default : DefaultType
        The default value of the field.

    Returns
    -------
    DefaultType
        The created field.
    """
    return dataclasses.field(default=default, metadata={"target": target, "loader": loader})


def asdict(dataclass: object) -> dict[str, Any]:
    """Create a dictionary containing all fields of the dataclass.

    Parameters
    ----------
    dataclass : object
        A dataclass instance.

    Returns
    -------
    dict[str, Any] :
        The dataclass represented as a dictionary.
    """
    fields = dataclasses.fields(dataclass)

    dataclass_dict = {}
    for field in fields:
        if "exclude_from_dict" not in field.metadata:
            value = getattr(dataclass, field.name)
            dataclass_dict[field.name] = (
                asdict(value) if dataclasses.is_dataclass(value) else value
            )

    return dataclass_dict


def fromdict(dataclass_type: type, dataclass_dict: dict[str, Any], folder: Path = None) -> object:
    """Create a dataclass instance from a dict and loads all file represented fields.

    Parameters
    ----------
    dataclass_type : type
        A dataclass type.
    dataclass_dict : dict[str, Any]
        A dict for instancing the the dataclass.
    folder : Path
        The root folder for file paths. If ``None`` file paths are consider absolute.

    Returns
    -------
    object
        Created instance of dataclass_type.
    """
    fields = dataclasses.fields(dataclass_type)

    for field in fields:
        if "target" in field.metadata and "loader" in field.metadata:
            file_path = dataclass_dict.get(field.name)
            if file_path is None:
                continue
            elif isinstance(file_path, list):
                dataclass_dict[field.metadata["target"]] = [
                    field.metadata["loader"](f if folder is None else folder / f)
                    for f in file_path
                ]
            elif isinstance(file_path, dict):
                dataclass_dict[field.metadata["target"]] = {
                    k: field.metadata["loader"](f if folder is None else folder / f)
                    for k, f in file_path.items()
                }
            else:
                dataclass_dict[field.metadata["target"]] = field.metadata["loader"](
                    file_path if folder is None else folder / file_path
                )
        elif dataclasses.is_dataclass(field.default) and field.name in dataclass_dict:
            dataclass_dict[field.name] = type(field.default)(**dataclass_dict[field.name])

    return dataclass_type(**dataclass_dict)
