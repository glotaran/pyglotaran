"""Glotaran IO utility module."""
from __future__ import annotations

import os
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from glotaran.plugin_system.data_io_registration import load_dataset
from glotaran.typing.types import DatasetMappable

if TYPE_CHECKING:
    from typing import Iterator

    from glotaran.typing.types import StrOrPath


def _load_datasets(dataset_mappable: DatasetMappable, index: int = 1) -> dict[str, xr.Dataset]:
    """Implement functionality for ``load_datasets`` and  internal use.

    Parameters
    ----------
    dataset_mappable : DatasetMappable
        Instance of ``DatasetMappable`` that can be used to create a dataset mapping.
    index : int
        Index used to create key and ``source_path`` if not present.
        , by default 1

    Returns
    -------
    dict[str, xr.Dataset]
        Mapping of datasets to initialize :class:`DatasetMapping`.

    Raises
    ------
    TypeError
        If the type of ``dataset_mappable`` is not explicitly supported.
    """
    dataset_mapping = {}
    if isinstance(dataset_mappable, (str, Path)):
        dataset_mapping[Path(dataset_mappable).stem] = load_dataset(dataset_mappable)
    elif isinstance(dataset_mappable, (xr.Dataset, xr.DataArray)):
        if isinstance(dataset_mappable, xr.DataArray):
            dataset_mappable: xr.Dataset = dataset_mappable.to_dataset(  # type:ignore[no-redef]
                name="data"
            )
        if "source_path" not in dataset_mappable.attrs:
            dataset_mappable.attrs["source_path"] = f"dataset_{index}.nc"
        dataset_mapping[Path(dataset_mappable.source_path).stem] = dataset_mappable
    elif isinstance(dataset_mappable, Sequence):
        for index, dataset in enumerate(dataset_mappable, start=1):
            key, value = next(iter(_load_datasets(dataset, index=index).items()))
            dataset_mapping[key] = value
    elif isinstance(dataset_mappable, Mapping):
        for key, dataset in dataset_mappable.items():
            _, value = next(iter(_load_datasets(dataset).items()))
            dataset_mapping[key] = value
    else:
        raise TypeError(
            f"Type '{type(dataset_mappable).__name__}' for 'dataset_mappable' of value "
            f"'{dataset_mappable}' is not supported."
            f"\nSupported types are:\n {DatasetMappable}."
        )
    return dataset_mapping


class DatasetMapping(MutableMapping):
    """Wrapper class for a mapping of datasets which can be used for a ``file_loadable_field``."""

    def __init__(self, init_map: Mapping[str, xr.Dataset] = None) -> None:
        """Initialize an instance of :class:`DatasetMapping`.

        Parameters
        ----------
        init_dict : dict[str, xr.Dataset], optional
            Mapping to initially populate the instance., by default None
        """
        super().__init__()
        self.__data_dict: dict[str, xr.Dataset] = {}
        if init_map is not None:
            for key, dataset in init_map.items():
                self[key] = dataset

    @classmethod
    def loader(cls: type[DatasetMapping], dataset_mappable: DatasetMappable) -> DatasetMapping:
        """Loader function utilized by ``file_loadable_field``.

        Parameters
        ----------
        dataset_mappable : DatasetMappable
            Mapping of datasets to initialize :class:`DatasetMapping`.

        Returns
        -------
        DatasetMapping
            Populated instance of :class:`DatasetMapping`.
        """
        return cls(_load_datasets(dataset_mappable))

    @property
    def source_path(self):
        """Map the ``source_path`` attribute of each dataset to a standalone mapping.

        Note
        ----
        When the ``source_path`` attribute of the dataset gets updated
        (e.g. by calling ``save_dataset`` with the default ``update_source_path=True``)
        this value will be updated as well.

        Returns
        -------
        Mapping[str, str]
            Mapping of the dataset source paths.
        """
        return {key: val.source_path for key, val in self.__data_dict.items()}

    def __getitem__(self, key: str) -> xr.Dataset:
        """Implement retrieving an element by its key."""
        return self.__data_dict[key]

    def __setitem__(self, key: str, value: xr.Dataset) -> None:
        """Implement setting an elements value."""
        if "source_path" not in value.attrs:
            value.attrs["source_path"] = f"{key}.nc"
        self.__data_dict[key] = value

    def __iter__(self) -> Iterator[str]:
        """Implement looping over an instance."""
        yield from self.__data_dict.keys()

    def __delitem__(self, key: str) -> None:
        """Implement deleting an item."""
        del self.__data_dict[key]

    def __len__(self) -> int:
        """Implement calling ``len`` on an instance."""
        return len(self.__data_dict)


def load_datasets(dataset_mappable: DatasetMappable) -> DatasetMapping:
    """Load multiple datasets into a mapping (convenience function).

    This is used for ``file_loadable_field`` of a dataset mapping e.g.
    in :class:`Scheme`

    Parameters
    ----------
    dataset_mappable : DatasetMappable
        Single dataset/file path to a dataset or sequence or mapping of it.

    Returns
    -------
    DatasetMapping
        Mapping of dataset with string keys, where datasets hare ensured to have
        the ``source_path`` attr.
    """
    return DatasetMapping.loader(dataset_mappable)


def relative_posix_path(source_path: StrOrPath, base_path: StrOrPath | None = None) -> str:
    """Ensure that ``source_path`` is a posix path, relative to ``base_path`` if defined.

    Parameters
    ----------
    source_path : StrOrPath
        Path which should be converted to a relative posix path.
    base_path : StrOrPath, optional
        Base path the resulting path string should be relative to., by default None

    Returns
    -------
    str
        ``source_path`` as posix path relative to ``base_path`` if defined.
    """
    source_path = Path(source_path).as_posix()
    if base_path is not None and os.path.isabs(source_path):
        source_path = os.path.relpath(source_path, Path(base_path).as_posix())
    return Path(source_path).as_posix()
