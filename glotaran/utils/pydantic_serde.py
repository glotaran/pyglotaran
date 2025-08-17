"""Module containing functions to work with Pydantic serialization and validation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeGuard
from typing import overload

from pydantic import SerializationInfo
from pydantic import ValidationInfo

from glotaran.io import load_parameters
from glotaran.io import save_parameters
from glotaran.utils.io import relative_posix_path

if TYPE_CHECKING:
    from glotaran.parameter.parameters import Parameters
    from glotaran.typing.protocols import ToFromCsvSerializable


class SerializationInfoWithContext(SerializationInfo):
    """Stub for type guarding the context of ``SerializationInfo``."""

    context: dict[str, Any]


class ValidationInfoWithContext(ValidationInfo):
    """Stub for type guarding the context of ``ValidationInfo``."""

    context: dict[str, Any]


@overload
def context_is_dict(info: SerializationInfo) -> TypeGuard[SerializationInfoWithContext]: ...
@overload
def context_is_dict(info: ValidationInfo) -> TypeGuard[ValidationInfoWithContext]: ...


def context_is_dict(
    info: SerializationInfo | ValidationInfo,
) -> TypeGuard[SerializationInfoWithContext | ValidationInfoWithContext]:
    """Check if the context of the serialization or validation info is a dictionary."""
    return info is not None and isinstance(info.context, dict)


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
    if context_is_dict(info) and "save_folder" in info.context:
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


def serialize_parameters(value: Parameters, info: SerializationInfo) -> str:
    """Serialize parameters to a file format provided via ``saving_options`` in context.

    To be used with a ``pydantic.field_serializer``.

    Parameters
    ----------
    value : Parameters
        The parameters to serialize.
    info : SerializationInfo
        The serialization info context.

    Returns
    -------
    str
        The relative path to the file.
    """
    if context_is_dict(info) and (save_folder := save_folder_from_info(info)) is not None:
        saving_options = info.context.get("saving_options", {})
        parameters_format = saving_options.get("parameter_format", "csv")
        parameters_plugin = saving_options.get("parameters_plugin", None)
        path = Path(save_folder) / f"{info.field_name}.{parameters_format}"  # type: ignore[attr-defined]
        path.parent.mkdir(parents=True, exist_ok=True)
        save_parameters(value, path, format_name=parameters_plugin)
        return relative_posix_path(path, base_path=Path(save_folder))
    msg = f"SerializationInfo context is missing 'save_folder':\n{info}"
    raise ValueError(msg)


def deserialize_parameters(value: Any, info: ValidationInfo) -> Parameters | Any:
    """Deserialize parameters from a file format provided via ``saving_options`` in context.

    To be used with a ``pydantic.field_validator``.

    Parameters
    ----------
    value : Parameters
        The parameters to deserialize.
    info : SerializationInfo
        The serialization info context.

    Returns
    -------
    "Parameters"
        The deserialized parameters.
    """
    if isinstance(value, str | Path) and context_is_dict(info):
        if (save_folder := save_folder_from_info(info)) is not None:
            saving_options = info.context.get("saving_options", {})
            return load_parameters(
                Path(save_folder) / value,
                format_name=saving_options.get("parameters_plugin", None),
            )
        msg = f"ValidationInfo context is missing 'save_folder':\n{info}"
        raise ValueError(msg)
    return value
