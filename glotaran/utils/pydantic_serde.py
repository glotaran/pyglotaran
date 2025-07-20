"""Module containing functions to work with Pydantic serialization and validation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from glotaran.utils.io import relative_posix_path

if TYPE_CHECKING:
    from pydantic import SerializationInfo
    from pydantic import ValidationInfo

    from glotaran.typing.protocols import ToFromCsvSerializable


def save_folder_from_info(info: SerializationInfo | ValidationInfo) -> Path | None:
    """Get the save folder from the serialization or validations info context.

    Parameters
    ----------
    info : SerializationInfo | ValidationInfo
        The serialization or validation info context.

    Returns
    -------
    Path | None
        The save folder path if available, otherwise None.
    """
    if info is not None and isinstance(info.context, dict) and "save_folder" in info.context:
        return Path(info.context["save_folder"])
    return None


def serialize_to_csv(value: ToFromCsvSerializable, info: SerializationInfo) -> str:
    """Serialize a value to a CSV file.

    To be used with a ``pydantic.field_serializer``.

    Parameters
    ----------
    value : ToCsvSerializable
        The value to serialize.
    info : SerializationInfo
        The serialization info context.

    Returns
    -------
    str
        The relative path to the CSV file.
    """
    if (save_folder := save_folder_from_info(info)) is not None:
        path = Path(save_folder) / f"{info.field_name}.csv"  # type: ignore[attr-defined]
        path.parent.mkdir(parents=True, exist_ok=True)
        value.to_csv(path)
        return relative_posix_path(path, base_path=Path(save_folder))
    msg = f"SerializationInfo context is missing 'save_folder':\n{info}"
    raise ValueError(msg)


def deserialize_from_csv(
    cls: type[ToFromCsvSerializable], value: Any, info: ValidationInfo
) -> ToFromCsvSerializable:
    """Deserialize a value from a CSV file.

    To be used with a ``pydantic.field_serializer``.

    Parameters
    ----------
    cls : type[ToFromCsvSerializable]
        The class to deserialize to.
    value : Any
        The value to deserialize.
    info : SerializationInfo
        The serialization info context.

    Returns
    -------
    ToFromCsvSerializable
        The deserialized value.
    """
    if isinstance(value, str | Path):
        if (save_folder := save_folder_from_info(info)) is not None:
            return cls.from_csv(Path(save_folder) / value)
        msg = f"ValidationInfo context is missing 'save_folder':\n{info}"
        raise ValueError(msg)
    return value
