"""Glotaran types module containing commonly used types."""
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar
from typing import Union

import numpy as np

try:
    from numpy._typing._array_like import _SupportsArray
except ImportError:
    # numpy < 1.23
    from numpy.typing._array_like import _SupportsArray  # type:ignore[no-redef]

import xarray as xr

T = TypeVar("T")
StrOrPath = Union[str, Path]
LoadableDataset = Union[StrOrPath, xr.Dataset, xr.DataArray]
DatasetMappable = Union[LoadableDataset, Sequence[LoadableDataset], Mapping[str, LoadableDataset]]


ArrayLike = np.ndarray
