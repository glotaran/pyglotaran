from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from glotaran.model.item import model_item
from glotaran.model.model import Model
from glotaran.parameter import Parameter
from glotaran.parameter import ParameterGroup

class EqualAreaPenalty:
    source: str
    source_intervals: list[tuple[float, float]]
    target: str
    target_intervals: list[tuple[float, float]]
    parameter: Parameter
    weight: str
    def applies(self, index: Any) -> bool: ...
    def fill(self, model: Model, parameters: ParameterGroup | None) -> EqualAreaPenalty: ...

def has_spectral_penalties(model: Model) -> bool: ...
def apply_spectral_penalties(
    model: Model,
    parameters: ParameterGroup,
    clp_labels: dict[str, list[str] | list[list[str]]],
    clps: dict[str, list[np.ndarray]],
    matrices: dict[str, np.ndarray | list[np.ndarray]],
    data: dict[str, xr.Dataset],
    group_tolerance: float,
) -> np.ndarray: ...
def _get_area(
    index_dependent: bool,
    global_dimension: str,
    clp_labels: dict[str, list[list[str]]],
    clps: dict[str, list[np.ndarray]],
    data: dict[str, xr.Dataset],
    group_tolerance: float,
    intervals: list[tuple[float, float]],
    compartment: str,
) -> np.ndarray: ...
def _get_idx_from_interval(
    interval: tuple[float, float], axis: Sequence[float] | np.ndarray
) -> tuple[int, int]: ...
