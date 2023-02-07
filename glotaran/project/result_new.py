from typing import TYPE_CHECKING

import xarray as xr
from pydantic import BaseModel

from glotaran.optimization import OptimizationResult
from glotaran.parameter import Parameters

if TYPE_CHECKING:
    from glotaran.project.scheme_new import Scheme


class Result(BaseModel):
    data: dict[str, xr.Dataset]
    optimization: OptimizationResult
    parameters_intitial: Parameters
    parameters_optimized: Parameters
    scheme: Scheme
