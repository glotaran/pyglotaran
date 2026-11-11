"""Glotaran types module containing commonly used types."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np
import xarray as xr
from numpy._typing._array_like import _SupportsArray  # noqa: F401

T = TypeVar("T")
type StrOrPath = str | Path
type LoadableDataset = StrOrPath | xr.Dataset | xr.DataArray
type DatasetMappable = LoadableDataset | Sequence[LoadableDataset] | Mapping[str, LoadableDataset]


ArrayLike = np.ndarray
