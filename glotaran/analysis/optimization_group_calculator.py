from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from glotaran.analysis.optimization_group import OptimizationGroup


class OptimizationGroupCalculator:
    """A Problem class"""

    def __init__(self, group: OptimizationGroup):
        self._group = group

    def calculate_matrices(self):
        raise NotImplementedError

    def calculate_residual(self):
        raise NotImplementedError

    def calculate_full_penalty(self) -> np.ndarray:
        raise NotImplementedError

    def prepare_result_creation(self):
        pass

    def create_index_dependent_result_dataset(self, label: str, dataset: xr.Dataset) -> xr.Dataset:
        """Creates a result datasets for index dependent matrices."""
        raise NotImplementedError

    def create_index_independent_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:
        """Creates a result datasets for index independent matrices."""
        raise NotImplementedError
