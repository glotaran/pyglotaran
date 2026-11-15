from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

import numpy as np
import xarray as xr

from glotaran.model.element import Element
from glotaran.model.item import Attribute

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike


class ClpGuideElement(Element):
    """Guides one amplitude with a dataset that holds its known values."""

    type: Literal["clp-guide"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "clp-guide"
    _exclusive: bool = True
    target: str = Attribute(
        description=(
            "Label of the guided amplitude, for example a compartment of a kinetic element. "
            "The dataset with this element holds the known values of that amplitude."
        )
    )

    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs: Any,  # noqa: ANN401
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
        # The guided clp estimate is part of the dataset's ``fit_decomposition.clp``.
        return xr.Dataset()
