"""Contains helper methods for dataclasses."""
from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import MISSING
from dataclasses import field
from dataclasses import fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from glotaran.utils.io import relative_posix_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any
    from typing import TypeVar

    from _typeshed import DataclassInstance

    from glotaran.typing.protocols import FileLoadable

    DefaultType = TypeVar("DefaultType")
    DataclassInstanceType = TypeVar("DataclassInstanceType", bound=DataclassInstance)


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


def file_loader_factory(
    targetClass: type[FileLoadable], *, is_wrapper_class: bool = False
) -> Callable[[FileLoadable | str | Path], FileLoadable]:
    """Create ``file_loader`` functions to load ``targetClass`` from file.

    Parameters
    ----------
    targetClass: type[FileLoadable]
        Class the loader function should return an instance of.
    is_wrapper_class: bool
        Whether or not ``targetClass`` is a wrapper class, so the isinstance check will be ignored
        and instead the responsibility for supported types lies at the implementation of
        the loader.

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
        if isinstance(source_path, targetClass):
            return source_path
        if isinstance(source_path, (str, Path)):
            if folder is not None:
                target_obj = targetClass.loader(Path(folder) / source_path)
            else:
                target_obj = targetClass.loader(source_path)
            target_obj.source_path = str(source_path)
            return target_obj  # type:ignore[return-value]
        if is_wrapper_class is True:
            if isinstance(source_path, Sequence) and folder is not None:
                source_path = [Path(folder) / val for val in source_path]
            if isinstance(source_path, Mapping) and folder is not None:
                source_path = {key: Path(folder) / val for key, val in source_path.items()}
            return targetClass.loader(source_path)  # type:ignore[return-value, arg-type]
        raise ValueError(
            f"The value of 'source_path' needs to be of class {targetClass.__name__} "
            "or a file path. If the class is a wrapper class, you can use the argument:\n"
            "'is_wrapper_class=True'"
        )

    return file_loader


def file_loadable_field(
    targetClass: type[FileLoadable], *, is_wrapper_class=False
) -> FileLoadable:
    """Create a dataclass field which can be and object of type ``targetClass`` or file path.

    Parameters
    ----------
    targetClass : type[FileLoadable]
        Class the resulting value should be an instance of.
    is_wrapper_class: bool
        Whether or not ``targetClass`` is a wrapper class, so the isinstance check will be ignored
        and instead the responsibility for supported types lies at the implementation of
        the loader.

    Notes
    -----
    This also requires to add ``init_file_loadable_fields`` in the ``__post_init__`` method.

    Returns
    -------
    FileLoadable
        Instance of ``targetClass``.

    See Also
    --------
    init_file_loadable_fields
    """
    return field(
        metadata={
            "file_loader": file_loader_factory(targetClass, is_wrapper_class=is_wrapper_class)
        }
    )


def init_file_loadable_fields(dataclass_instance: DataclassInstance):
    """Load objects into class when dataclass is initialized with paths.

    If the class has file_loadable fields, this needs be called in the
    ``__post_init__`` method of that class.

    Parameters
    ----------
    dataclass_instance : DataclassInstance
        Instance of the dataclass being initialized.
        When used inside of ``__post_init__`` for the class itself use ``self``.

    See Also
    --------
    file_loadable_field
    """
    for field_item in fields(dataclass_instance):
        if "file_loader" in field_item.metadata:
            file_loader = field_item.metadata["file_loader"]
            value = getattr(dataclass_instance, field_item.name)
            setattr(dataclass_instance, field_item.name, file_loader(value))


def asdict(dataclass: DataclassInstance, folder: Path | None = None) -> dict[str, Any]:
    """Create a dictionary containing all fields of the dataclass.

    Parameters
    ----------
    dataclass : DataclassInstance
        A dataclass instance.
    folder: Path | None
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
                if isinstance(value.source_path, (str, Path)):
                    dataclass_dict[field_item.name] = relative_posix_path(
                        value.source_path, folder
                    )
                elif isinstance(value.source_path, Sequence):
                    dataclass_dict[field_item.name] = [
                        relative_posix_path(val, folder) for val in value.source_path
                    ]
                elif isinstance(value.source_path, Mapping):
                    dataclass_dict[field_item.name] = {
                        key: relative_posix_path(val, folder)
                        for key, val in value.source_path.items()
                    }

    return dataclass_dict


def fromdict(
    dataclass_type: type[DataclassInstanceType],
    dataclass_dict: dict[str, Any],
    folder: Path | None = None,
) -> DataclassInstanceType:
    """Create a dataclass instance from a dict and loads all file represented fields.

    Parameters
    ----------
    dataclass_type : type[DataclassInstanceType]
        A dataclass type.
    dataclass_dict : dict[str, Any]
        A dict for instancing the the dataclass.
    folder : Path
        The root folder for file paths. If ``None`` file paths are consider absolute.

    Returns
    -------
    DataclassInstanceType
        Created instance of dataclass_type.
    """
    for field_item in fields(dataclass_type):
        if "file_loader" in field_item.metadata:
            file_path = dataclass_dict.get(field_item.name)
            dataclass_dict[field_item.name] = field_item.metadata["file_loader"](file_path, folder)
        elif is_dataclass(field_item.default) and field_item.name in dataclass_dict:
            dataclass_dict[field_item.name] = type(field_item.default)(  # type:ignore[misc]
                **dataclass_dict[field_item.name]
            )

    return dataclass_type(**dataclass_dict)
