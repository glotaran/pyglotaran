from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

import numpy as np
import xarray as xr

from glotaran.model.element import Element

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike


class BaselineElement(Element):
    """A constant along the model dimension, with one amplitude for each global value.

    Set ``dimension`` to the model dimension of the dataset, for example ``time``.
    """

    type: Literal["baseline"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "baseline"
    _unique: ClassVar[bool] = True

    def clp_label(self) -> str:
        return f"baseline_{self.label}"

    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[list[str], ArrayLike]:
        clp_label = [self.clp_label()]
        matrix = np.ones((model_axis.size, 1), dtype=np.float64)
        return clp_label, matrix

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        return xr.Dataset({"amplitudes": amplitudes.sel(amplitude_label=self.clp_label())})
