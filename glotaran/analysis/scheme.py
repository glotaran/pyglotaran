import functools
import typing
import xarray as xr
import numpy as np

import glotaran
from glotaran.parameter import ParameterGroup


def _not_none(f):

    @functools.wraps(f)
    def decorator(self, value):
        if value is None:
            raise ValueError(f"{f.__name__} cannot be None")
        f(self, value)


class Scheme:

    def __init__(self,
                 model: 'glotaran.model.Model' = None,
                 parameter: ParameterGroup = None,
                 data: typing.Dict[str, typing.Union[xr.DataArray, xr.Dataset]] = None,
                 group_tolerance: float = 0.0,
                 nnls: bool = False,
                 nfev: int = None,
                 ):

        self.model = model
        self.parameter = parameter
        self.data = data
        self.group_tolerance = group_tolerance
        self.nnls = nnls
        self.nfev = nfev

    @property
    def model(self) -> 'glotaran.model.Model':
        return self._model

    @_not_none
    @model.setter
    def model(self, model: 'glotaran.model.Model'):
        self._model = model

    @property
    def parameter(self) -> ParameterGroup:
        return self._parameter

    @_not_none
    @parameter.setter
    def parameter(self, parameter: ParameterGroup):
        self._parameter = parameter

    @property
    def data(self) -> typing.Dict[str, typing.Union[xr.DataArray, xr.Dataset]]:
        return self._data

    @_not_none
    @data.setter
    def data(self, data: typing.Dict[str, typing.Union[xr.DataArray, xr.Dataset]]):
        self._data = data

    @property
    def nnls(self) -> bool:
        return self._nnls

    @_not_none
    @nnls.setter
    def nnls(self, nnls: bool):
        self._nnls = nnls

    @property
    def nfev(self) -> int:
        return self._nfev

    @nfev.setter
    def nfev(self, nfev: int):
        self._nfev = nfev

    @property
    def group_tolerance(self) -> float:
        return self._group_tolerance

    @group_tolerance.setter
    def group_tolerance(self, group_tolerance: float):
        self._group_tolerance = group_tolerance

    def problem_list(self) -> typing.List[str]:
        """Returns a list with all problems in the model and missing parameters."""
        return self.model.problem_list(self.parameter)

    def validate(self) -> str:
        """Returns a string listing all problems in the model and missing parameters."""
        return self.model.validate(self.parameter)

    def valid(self, parameter: ParameterGroup = None) -> bool:
        """Returns `True` if there are no problems with the model or the parameters,
        else `False`."""
        return self.model.valid(parameter)

    def prepared_data(self) -> typing.Dict[str, xr.Dataset]:
        data = {}
        for label, dataset in self.data.items():
            if self.model.matrix_dimension not in dataset.dims:
                raise Exception("Missing coordinates for dimension "
                                f"'{self.model.matrix_dimension}' in data for dataset "
                                f"'{label}'")
            if self.model.global_dimension not in dataset.dims:
                raise Exception("Missing coordinates for dimension "
                                f"'{self.model.global_dimension}' in data for dataset "
                                f"'{label}'")
            if isinstance(dataset, xr.DataArray):
                dataset = dataset.to_dataset(name="data")

            if 'weight' in dataset and 'weighted_data' not in dataset:
                dataset['weighted_data'] = np.multiply(dataset.data, dataset.weight)
            data[label] = dataset.transpose(
                self.model.matrix_dimension, self.model.global_dimension,
                *[dim for dim in dataset.dims
                  if dim is not self.model.matrix_dimension
                  and dim is not self.model.global_dimension])
        return data
