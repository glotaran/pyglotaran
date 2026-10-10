"""Glotaran types module containing commonly used types."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias
from typing import TypeVar

import numpy as np
import xarray as xr
from numpy._typing._array_like import _SupportsArray  # noqa: F401

T = TypeVar("T")
StrOrPath: TypeAlias = str | Path
LoadableDataset: TypeAlias = StrOrPath | xr.Dataset | xr.DataArray
DatasetMappable: TypeAlias = (
    LoadableDataset | Sequence[LoadableDataset] | Mapping[str, LoadableDataset]
)


ArrayLike = np.ndarray
