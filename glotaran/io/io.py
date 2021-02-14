from __future__ import annotations

import xarray as xr

from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Result
from glotaran.project import SavingOptions
from glotaran.project import Scheme


class Io:
    @staticmethod
    def read_model(fmt: str, file_name: str) -> Model:
        raise NotImplementedError

    @staticmethod
    def write_model(fmt: str, file_name: str, model: Model):
        raise NotImplementedError

    @staticmethod
    def read_parameters(fmt: str, file_name: str) -> ParameterGroup:
        raise NotImplementedError

    @staticmethod
    def write_parameters(fmt: str, file_name: str, parameters: ParameterGroup):
        raise NotImplementedError

    @staticmethod
    def read_scheme(fmt: str, file_name: str) -> Scheme:
        raise NotImplementedError

    @staticmethod
    def write_scheme(fmt: str, file_name: str, scheme: Scheme):
        raise NotImplementedError

    @staticmethod
    def read_dataset(fmt: str, file_name: str) -> xr.DataSet | xr.DataArray:
        raise NotImplementedError

    @staticmethod
    def write_dataset(
        fmt: str, file_name: str, saving_options: SavingOptions, dataset: xr.DataSet
    ):
        raise NotImplementedError

    @staticmethod
    def read_result(fmt: str, file_name: str) -> Result:
        raise NotImplementedError

    @staticmethod
    def write_result(fmt: str, file_name: str, saving_options: SavingOptions, result: Result):
        raise NotImplementedError
