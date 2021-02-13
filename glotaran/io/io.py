from __future__ import annotations

from typing import TYPE_CHECKING

from glotaran.model import Model
from glotaran.parameter import ParameterGroup

if TYPE_CHECKING:
    from glotaran.analysis import Result


class Io:
    def read_model(fmt: str, file_name: str) -> Model:
        raise NotImplementedError

    def write_model(fmt: str, file_name: str, model: Model):
        raise NotImplementedError

    def read_parameters(fmt: str, file_name: str) -> ParameterGroup:
        raise NotImplementedError

    def write_parameters(fmt: str, file_name: str, parameters: ParameterGroup):
        raise NotImplementedError

    def read_result(fmt: str, result_name: str) -> Result:
        raise NotImplementedError

    def write_result(fmt: str, result_name: str, result: Result):
        raise NotImplementedError
