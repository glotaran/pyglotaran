"""Glotaran types module containing commonly used types."""
from pathlib import Path
from typing import Mapping
from typing import Sequence
from typing import Union

import xarray as xr

StrOrPath = Union[str, Path]
LoadableDataset = Union[StrOrPath, xr.Dataset, xr.DataArray]
DatasetMappable = Union[LoadableDataset, Sequence[LoadableDataset], Mapping[str, LoadableDataset]]
