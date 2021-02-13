from __future__ import annotations

import xarray as xr

from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Result
from glotaran.project import SavingOptions
from glotaran.project import Scheme


class Io:
    def read_model(fmt: str, file_name: str) -> Model:
        raise NotImplementedError

    def write_model(fmt: str, file_name: str, model: Model):
        raise NotImplementedError

    def read_parameters(fmt: str, file_name: str) -> ParameterGroup:
        raise NotImplementedError

    def write_parameters(fmt: str, file_name: str, parameters: ParameterGroup):
        raise NotImplementedError

    def read_scheme(fmt: str, file_name: str) -> Scheme:
        raise NotImplementedError

    def write_scheme(fmt: str, file_name: str, scheme: Scheme):
        raise NotImplementedError

    def read_dataset(fmt: str, file_name: str) -> xr.DataSet | xr.DataArray:
        raise NotImplementedError

    def write_dataset(
        fmt: str, file_name: str, saving_options: SavingOptions, dataset: xr.DataSet
    ):
        raise NotImplementedError

    def read_result(fmt: str, file_name: str) -> Result:
        raise NotImplementedError

    def write_result(fmt: str, file_name: str, saving_options: SavingOptions, result: Result):
        raise NotImplementedError
