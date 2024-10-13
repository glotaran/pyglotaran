from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

import numpy as np
import xarray as xr

from glotaran.model.element import Element

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike


class ClpGuideElement(Element):
    type: Literal["clp-guide"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "clp-guide"
    _exclusive: bool = True
    target: str

    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ) -> tuple[list[str], ArrayLike]:
        return [self.target], np.ones((1, 1), dtype=np.float64)

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        return xr.Dataset()  # TODO: return correct data
