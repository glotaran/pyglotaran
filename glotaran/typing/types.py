"""Glotaran types module containing commonly used types."""
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar
from typing import Union

import xarray as xr

T = TypeVar("T")
StrOrPath = Union[str, Path]
LoadableDataset = Union[StrOrPath, xr.Dataset, xr.DataArray]
DatasetMappable = Union[LoadableDataset, Sequence[LoadableDataset], Mapping[str, LoadableDataset]]
