from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

import numpy as np

from glotaran.model.element import Element

if TYPE_CHECKING:
    import xarray as xr

    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike


class BaselineElement(Element):
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
        **kwargs,
    ) -> tuple[list[str], ArrayLike]:
        clp_label = [self.clp_label()]
        matrix = np.ones((model_axis.size, 1), dtype=np.float64)
        return clp_label, matrix

    def add_to_result_data(
        self,
        model: DataModel,
        data: xr.Dataset,
        as_global: bool = False,
    ):
        if not as_global:
            data["baseline"] = data.clp.sel(clp_label=self.clp_label())
