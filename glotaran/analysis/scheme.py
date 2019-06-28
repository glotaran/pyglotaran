import functools
import pathlib
import typing
import yaml
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

    @classmethod
    def from_yml_file(cls, filename: str) -> 'Scheme':

        try:
            with open(filename) as f:
                try:
                    scheme = yaml.load(f, Loader=yaml.FullLoader)
                except Exception as e:
                    raise Exception(f"Error parsing scheme: {e}")
        except Exception as e:
            raise Exception(f"Error opening scheme: {e}")

        if 'model' not in scheme:
            raise Exception('Model file not specified.')

        try:
            model = glotaran.read_model_from_yml_file(scheme['model'])
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

        if 'parameter' not in scheme:
            raise Exception('Parameter file not specified.')

        path = scheme['parameter']
        fmt = scheme.get('parameter_format', None)
        try:
            parameter = glotaran.parameter.ParameterGroup.from_file(path, fmt)
        except Exception as e:
            raise Exception(f"Error loading parameter: {e}")

        if 'data' not in scheme:
            raise Exception('No data specified.')

        data = {}
        for label, path in scheme['data'].items():
            path = pathlib.Path(path)

            fmt = path.suffix[1:] if path.suffix != '' else 'nc'
            if 'data_format' in scheme:
                fmt = scheme['data_format']

            try:
                data[label] = glotaran.io.read_data_file(path, fmt=fmt)
            except Exception as e:
                raise Exception(f"Error loading dataset '{label}': {e}")

        nnls = scheme.get('nnls', False)
        nfev = scheme.get('nfev', None)
        group_tolerance = scheme.get('group_tolerance', 0.0)
        return cls(model=model, parameter=parameter, data=data,
                   nnls=nnls, nfev=nfev, group_tolerance=group_tolerance)

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

            # This protects transposing when getting data with svd in it

            if 'data_singular_values' in dataset:
                if dataset.coords['right_singular_value_index'].size != \
                  dataset.coords[self.model.global_dimension].size:
                    dataset = dataset.rename(
                        right_singular_value_index='right_singular_value_indexTMP')
                    dataset = dataset.rename(
                        left_singular_value_index='right_singular_value_index')
                    dataset = dataset.rename(
                        right_singular_value_indexTMP='left_singular_value_index')
                    dataset = dataset.rename(
                        right_singular_vectors='right_singular_value_vectorsTMP')
                    dataset = dataset.rename(
                        left_singular_value_vectors='right_singular_value_vectors')
                    dataset = dataset.rename(
                        right_singular_value_vectorsTMP='left_singular_value_vectors')
            new_dims = [self.model.matrix_dimension, self.model.global_dimension]
            new_dims += [dim for dim in dataset.dims
                         if dim != self.model.matrix_dimension
                         and dim != self.model.global_dimension]
            data[label] = dataset.transpose(*new_dims)
        return data

    def markdown(self):
        s = self.model.markdown(parameter=self.parameter)

        s += "\n\n"
        s += "__Scheme__\n\n"

        s += f"* *nnls*: {self.nnls}\n"
        s += f"* *nfev*: {self.nfev}\n"
        s += f"* *group_tolerance*: {self.group_tolerance}\n"

        return s
