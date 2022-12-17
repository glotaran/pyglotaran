"""Glotaran IO utility module."""

from __future__ import annotations

import contextlib
import html
import inspect
import os
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import xarray as xr

from glotaran.plugin_system.data_io_registration import load_dataset
from glotaran.typing.types import DatasetMappable

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Iterator

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

    def __init__(self, init_map: Mapping[str, xr.Dataset] | None = None) -> None:
        """Initialize an instance of :class:`DatasetMapping`.

        Parameters
        ----------
        init_dict : dict[str, xr.Dataset] | None
            Mapping to initially populate the instance. Defaults to ``None``.
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


@contextmanager
def chdir_context(folder_path: StrOrPath) -> Generator[Path, None, None]:
    """Context manager to change directory to ``folder_path``.

    Parameters
    ----------
    folder_path: StrOrPath
        Path to change to.

    Yields
    ------
    Generator[Path, None, None]
        Resolved path of ``folder_path``.

    Raises
    ------
    ValueError
        If ``folder_path`` is an existing file.
    """
    original_dir = Path(os.curdir).resolve()
    folder_path = Path(folder_path)
    if folder_path.is_file() is True:
        raise ValueError("Value of 'folder_path' needs to be a folder but was an existing file.")
    folder_path.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(folder_path)
        yield folder_path.resolve()
    finally:
        os.chdir(original_dir)


def relative_posix_path(source_path: StrOrPath, base_path: StrOrPath | None = None) -> str:
    """Ensure that ``source_path`` is a posix path, relative to ``base_path`` if defined.

    For ``source_path`` to be converted to a relative path it either needs to a an absolute path or
    ``base_path`` needs to be a parent directory of ``source_path``.
    On Windows if ``source_path`` and ``base_path`` are on different drives, it will return
    the absolute posix path to the file.

    Parameters
    ----------
    source_path : StrOrPath
        Path which should be converted to a relative posix path.
    base_path : StrOrPath, optional
        Base path the resulting path string should be relative to. Defaults to ``None``.

    Returns
    -------
    str
        ``source_path`` as posix path relative to ``base_path`` if defined.
    """
    source_path = Path(source_path)
    if base_path is not None and (
        source_path.is_absolute() or Path(base_path).resolve() in source_path.resolve().parents
    ):
        with contextlib.suppress(ValueError):
            source_path = os.path.relpath(source_path.as_posix(), Path(base_path).as_posix())

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


def get_script_dir(*, nesting: int = 0) -> Path:
    """Get the parent folder a script is executed in.

    This is a helper function for cross compatibility with jupyter notebooks.
    In notebooks the global ``__file__`` variable isn't set, thus we need different
    means to get the folder a script is defined in, which doesn't change with the
    current working director the ``python interpreter`` was called from.
    Parameters
    ----------
    nesting : int
        Number to go up in the call stack to get to the initially calling function.
        This is only needed for library code and not for user code.
        , by default 0 (direct call)
    Returns
    -------
    Path
        Path to the folder the script was resides in.
    """
    calling_frame = inspect.stack()[nesting + 1].frame
    file_var = calling_frame.f_globals.get("__file__", ".")
    file_path = Path(file_var).resolve()
    return file_path if file_var == "." else file_path.parent


def make_path_absolute_if_relative(path: Path) -> Path:
    """Get a path as absolute if relative.

    Parameters
    ----------
    path : Path
        The path to make absolute.
    Returns
    -------
    Path
        Either the original path or the path as absolute relative to the script directory.
    """
    if not path.is_absolute():
        path = get_script_dir(nesting=2) / path
    return path


def create_clp_guide_dataset(
    result: Result | xr.Dataset, clp_label: str, dataset_name: str | None = None
) -> xr.Dataset:
    """Create dataset for clp guidance.

    Parameters
    ----------
    result: Result | xr.Dataset
        Optimization result object or dataset, created with pyglotaran>=0.6.0.
    clp_label : str
        Label of the clp to guide.
    dataset_name : str | None
        Name of dataset to extract the guide from. Defaults to None.

    Returns
    -------
    xr.Dataset
        DataArray containing the clp guide, with ``clp_label`` dimension replaced by the
        model dimensions first value.

    Raises
    ------
    ValueError
        If result is an instance of ``Result`` and ``dataset_name`` is ``None`` or not in result.
    ValueError
        If ``clp_labels`` is not in result.
    ValueError
        The result dataset was created with pyglotaran<0.6.0.

    Examples
    --------
    Extracting the clp guide from an optimization result object.

    .. code-block:: python

        from glotaran.io import save_dataset
        from glotaran.utils.io import create_clp_guide_dataset

        clp_guide = create_clp_guide_dataset(result, "species_1", "dataset_1")
        save_dataset(clp_guide, "clp_guide__result_dataset_1__species_1.nc")

    Extracting the clp guide from a result dataset loaded from file.

    .. code-block:: python

        from glotaran.io import load_dataset
        from glotaran.io import save_dataset
        from glotaran.utils.io import create_clp_guide_dataset

        result_dataset = load_dataset("result_dataset_1.nc")
        clp_guide = create_clp_guide_dataset(result_dataset, "species_1")
        save_dataset(clp_guide, "clp_guide__result_dataset_1__species_1.nc")

    """
    if isinstance(result, xr.Dataset):
        dataset = result
    elif dataset_name is None or dataset_name not in result.data:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Known datasets are:\n {list(result.data.keys())}"
        )
    else:
        dataset = result.data[dataset_name]
    if clp_label not in dataset.clp_label:
        raise ValueError(
            f"Unknown clp_label {clp_label!r}. "
            f"Known clp_labels are:\n {list(dataset.clp_label.values)}"
        )
    if "model_dimension" not in dataset.attrs:
        raise ValueError(
            "Result dataset is missing attribute 'model_dimension', "
            "which means that it was created with pyglotaran<0.6.0."
            "Please recreate the result with the latest version of pyglotaran."
        )

    clp_values = dataset.clp.sel(clp_label=[clp_label])
    value_dimension = next(filter(lambda x: x != dataset.model_dimension, clp_values.dims))

    return xr.DataArray(
        clp_values.values.T,
        coords={
            dataset.model_dimension: [dataset.coords[dataset.model_dimension][0].item()],
            value_dimension: clp_values.coords[value_dimension].values,
        },
    ).to_dataset(name="data")
