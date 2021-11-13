"""Contains helper methods for dataclasses."""
from __future__ import annotations

from dataclasses import MISSING
from dataclasses import field
from dataclasses import fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import TypeVar

    from glotaran.typing.protocols import FileLoadable

    DefaultType = TypeVar("DefaultType")


def exclude_from_dict_field(
    default: DefaultType = MISSING,  # type:ignore[assignment]
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
    return field(default=default, metadata={"exclude_from_dict": True})


def file_representation_field(
    target: str,
    loader: Callable[[str], Any],
    default: DefaultType = MISSING,  # type:ignore[assignment]
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
    return field(default=default, metadata={"target": target, "loader": loader})


def file_loader_factory(
    targetClass: type[FileLoadable],
) -> Callable[[FileLoadable | str | Path], FileLoadable]:
    """Create ``file_loader`` functions to load ``targetClass`` from file.

    Parameters
    ----------
    targetClass: type[FileLoadable]
        Class the loader function should return an instance of.

    Returns
    -------
    file_loader: Callable[[FileLoadable | str | Path], FileLoadable]
        Function to load ``FileLoadable`` from source file or return instance if already loaded.
    """

    def file_loader(
        source_path: FileLoadable | str | Path, folder: str | Path | None = None
    ) -> FileLoadable:
        """Functions to load ``targetClass`` from file.

        Parameters
        ----------
        source_path : FileLoadable | str | Path
            Instance to ``targetClass`` or a file path to load it from.
        folder : str | Path | None
            Path to the base folder ``source_path`` is a relative path to., by default None

        Returns
        -------
        FileLoadable
            Instance of ``targetClass``.

        Raises
        ------
        ValueError
            If not an instance of ``targetClass`` or a source path to load from.
        """
        if isinstance(source_path, (str, Path)):
            if folder is not None:
                target_obj = targetClass.loader(Path(folder) / source_path)
            else:
                target_obj = targetClass.loader(source_path)
            target_obj.source_path = str(source_path)
            return target_obj  # type:ignore[return-value]
        if isinstance(source_path, targetClass):
            return source_path
        raise ValueError(
            f"The value of 'target' needs to be of class {targetClass.__name__} or a file path."
        )

    return file_loader


def file_loadable_field(targetClass: type[FileLoadable]) -> FileLoadable:
    """Create a dataclass field which can be and object of type ``targetClass`` or file path.

    Parameters
    ----------
    targetClass : type[FileLoadable]
        Class the resulting value should be an instance of.

    Returns
    -------
    FileLoadable
        Instance of ``targetClass``.
    """
    return field(metadata={"file_loader": file_loader_factory(targetClass)})


def init_file_loadable_fields(dataclass_instance: object):
    """Load objects into class when dataclass is initialized with paths.

    If the class has file_loadable fields, this should be called in the
    ``__post_init__`` method of that class.

    Parameters
    ----------
    dataclass_instance : object
        Instance of the dataclass being initialized.
        When used inside of ``__post_init__`` for the class itself use ``self``.
    """
    for field_item in fields(dataclass_instance):
        if "file_loader" in field_item.metadata:
            file_loader = field_item.metadata["file_loader"]
            value = getattr(dataclass_instance, field_item.name)
            setattr(dataclass_instance, field_item.name, file_loader(value))


def asdict(dataclass: object, folder: Path = None) -> dict[str, Any]:
    """Create a dictionary containing all fields of the dataclass.

    Parameters
    ----------
    dataclass : object
        A dataclass instance.
    folder: Path
        Parent folder of :class:`FileLoadable` fields. by default None

    Returns
    -------
    dict[str, Any] :
        The dataclass represented as a dictionary.
    """
    dataclass_dict = {}
    for field_item in fields(dataclass):
        if "exclude_from_dict" not in field_item.metadata:
            value = getattr(dataclass, field_item.name)
            dataclass_dict[field_item.name] = asdict(value) if is_dataclass(value) else value
        if "file_loader" in field_item.metadata:
            value = getattr(dataclass, field_item.name)
            if value.source_path is not None:
                source_path = Path(value.source_path)
                if folder is not None and source_path.is_absolute():
                    source_path = source_path.relative_to(folder)
                dataclass_dict[field_item.name] = source_path.as_posix()

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
    for field_item in fields(dataclass_type):
        if "target" in field_item.metadata and "loader" in field_item.metadata:
            file_path = dataclass_dict.get(field_item.name)
            if file_path is None:
                continue
            elif isinstance(file_path, list):
                dataclass_dict[field_item.metadata["target"]] = [
                    field_item.metadata["loader"](f if folder is None else folder / f)
                    for f in file_path
                ]
            elif isinstance(file_path, dict):
                dataclass_dict[field_item.metadata["target"]] = {
                    k: field_item.metadata["loader"](f if folder is None else folder / f)
                    for k, f in file_path.items()
                }
            else:
                dataclass_dict[field_item.metadata["target"]] = field_item.metadata["loader"](
                    file_path if folder is None else folder / file_path
                )
        if "file_loader" in field_item.metadata:
            file_path = dataclass_dict.get(field_item.name)
            dataclass_dict[field_item.name] = field_item.metadata["file_loader"](file_path, folder)
        elif is_dataclass(field_item.default) and field_item.name in dataclass_dict:
            dataclass_dict[field_item.name] = type(field_item.default)(
                **dataclass_dict[field_item.name]
            )

    return dataclass_type(**dataclass_dict)
