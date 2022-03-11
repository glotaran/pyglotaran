"""Glotaran IO utility module."""
from __future__ import annotations

import html
import os
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import xarray as xr

from glotaran.plugin_system.data_io_registration import load_dataset
from glotaran.typing.types import DatasetMappable

if TYPE_CHECKING:
    from typing import Iterator

    import pandas as pd

    from glotaran.project.result import Result
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

    def __repr__(self) -> str:
        """Implement calling ``repr`` on an instance."""
        items = [f"{dataset_name!r}: <xarray.Dataset>" for dataset_name in self]
        return f"{{{', '.join(items)}}}"

    def _repr_html_(self) -> str:
        """Return a html representation str.

        Special method used by ``ipython`` to render html.

        Returns
        -------
        str
            DatasetMapping as html string.
        """
        items = [
            f"<details><summary>{dataset_name}</summary>{dataset._repr_html_()}</details>\n"
            for dataset_name, dataset in self.items()
        ]
        return f"<pre>{html.escape(repr(self))}</pre>\n{''.join(items)}"


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

    On Windows if ``source_path`` and ``base_path`` are on different drives, it will return
    the absolute posix path to the file.

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
        try:
            source_path = os.path.relpath(source_path, Path(base_path).as_posix())
        except ValueError:
            pass
    return Path(source_path).as_posix()


def safe_dataframe_fillna(df: pd.DataFrame, column_name: str, fill_value: Any) -> None:
    """Fill NaN values with ``fill_value``  if the column exists or do nothing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from which specific column values will be replaced
    column_name : str
        Name of column of ``df`` to fill NaNs
    fill_value : Any
        Value to fill NaNs with
    """
    if column_name in df.columns:
        df[column_name].fillna(fill_value, inplace=True)


def safe_dataframe_replace(
    df: pd.DataFrame, column_name: str, to_be_replaced_values: Any, replace_value: Any
) -> None:
    """Replace column values with ``replace_value`` if the column exists or do nothing.

    If ``to_be_replaced_values`` is not list or tuple format,
    convert into list with same ``to_be_replaced_values`` as element.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from which specific column values will be replaced
    column_name : str
        Name of column of ``df`` to replace values for
    to_be_replaced_values : Any
        Values to be replaced
    replace_value : Any
        Value to replace ``to_be_replaced_values`` with
    """
    if not isinstance(to_be_replaced_values, (list, tuple)):
        to_be_replaced_values = [to_be_replaced_values]
    if column_name in df.columns:
        df[column_name].replace(to_be_replaced_values, replace_value, inplace=True)


def extract_sas(
    result: Result | xr.Dataset, dataset: str | None = None, species: str | None = None
) -> xr.DataArray:
    """Extract SAS data from a result.

    This is a helper function to easily extract clp guidance spectra.

    Parameters
    ----------
    result: Result | xr.Dataset
        Optimization ``Result`` instance or result dataset.
    dataset: str | None
        Name of the dataset to look up in a Result instance. Defaults to None
    species: str | None
        Name op the species to extract the SAS for. Defaults to None

    Returns
    -------
    xr.DataArray
        SAS data with dummy time dimension.

    Raises
    ------
    ValueError
        If result is of type ``Result`` and ``dataset`` is None or not in the result data.
    ValueError
        If result is not of type ``Result`` or ``xr.Dataset``.
    ValueError
        If ``species`` is None or not in the species coordinates.
    ValueError
        If the result dataset does not contain a ``species_associated_spectra`` data variable.

    Examples
    --------
    Extracting the SAS from an optimization result object.

    .. code-block:: python

        from glotaran.utils.io import extract_sas

        sas = extract_sas(result, "dataset_1", "species_1")


    Extracting the SAS from a result dataset loaded from file.

    .. code-block:: python

        from glotaran.io import load_result
        from glotaran.utils.io import extract_sas

        result_dataset = load_result("result_dataset_1.nc")
        sas = extract_sas(result_dataset, "species_1")

    """
    # workaround to prevent circular imports
    from glotaran.project.result import Result

    if isinstance(result, xr.Dataset):
        result_dataset = result
    elif isinstance(result, Result):
        if dataset is None or dataset not in result.data:
            raise ValueError(
                f"The result doesn't contain a dataset with name {dataset!r}.\n"
                f"Valid values are: {list(result.data.keys())}"
            )
        result_dataset = result.data[dataset]
    else:
        raise ValueError(
            f"Unsupported result type: {type(result).__name__!r}\n"
            "Supported types are: ['Result', 'xr.Dataset']"
        )
    if species is None or species not in result_dataset.species:
        raise ValueError(
            f"The result doesn't contain a species with name {species!r}.\n"
            f"Valid values are: {list(result_dataset.species.values)}"
        )
    if "species_associated_spectra" not in result_dataset:
        raise ValueError(
            "The result does not have a 'species_associated_spectra' data variable.\n"
            f"Contained data variables are: {list(result_dataset.data_vars.keys())}"
        )
    sas_values = result_dataset.species_associated_spectra.sel(species=species)
    # For backwards compatibility we need to insert a fake time dimension
    return xr.DataArray(
        [sas_values.values], coords={"time": [0.0], "spectral": sas_values.coords["spectral"]}
    )
