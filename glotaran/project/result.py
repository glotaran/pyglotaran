import xarray as xr
from pydantic import BaseModel
from pydantic import Extra

from glotaran.model import ExperimentModel
from glotaran.optimization import OptimizationResult
from glotaran.parameter import Parameters


class Result(BaseModel):
    class Config:
        """Config for pydantic.BaseModel."""

        arbitrary_types_allowed = True
        extra = Extra.forbid

    data: dict[str, xr.Dataset]
    experiments: dict[str, ExperimentModel]
    optimization: OptimizationResult
    parameters_intitial: Parameters
    parameters_optimized: Parameters
