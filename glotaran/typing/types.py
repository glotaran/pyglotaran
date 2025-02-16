"""Glotaran types module containing commonly used types."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias
from typing import TypeVar

import numpy as np

try:
    from numpy._typing._array_like import _SupportsArray
except ImportError:
    # numpy < 1.23
    from numpy.typing._array_like import _SupportsArray  # type:ignore[no-redef]  # noqa: F401

import xarray as xr

T = TypeVar("T")
StrOrPath: TypeAlias = str | Path
LoadableDataset: TypeAlias = StrOrPath | xr.Dataset | xr.DataArray
DatasetMappable: TypeAlias = (
    LoadableDataset | Sequence[LoadableDataset] | Mapping[str, LoadableDataset]
)


ArrayLike = np.ndarray
