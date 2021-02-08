from __future__ import annotations

import functools
import pathlib
import warnings
from typing import Literal

import numpy as np
import xarray as xr
import yaml

import glotaran
from glotaran.model import Model
from glotaran.parameter import ParameterGroup


def _not_none(f):
    @functools.wraps(f)
    def decorator(self, value):
        if value is None:
            raise ValueError(f"{f.__name__} cannot be None")
        f(self, value)


class Scheme:
    def __init__(
        self,
        model: Model = None,
        parameters: ParameterGroup = None,
        data: dict[str, xr.DataArray | xr.Dataset] = None,
        group_tolerance: float = 0.0,
        non_negative_least_squares: bool = False,
        maximum_number_function_evaluations: int = None,
        ftol: float = 1e-8,
        gtol: float = 1e-8,
        xtol: float = 1e-8,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
    ):

        self._model = model
        self._parameters = parameters
        self._group_tolerance = group_tolerance
        self._non_negative_least_squares = non_negative_least_squares
        self._maximum_number_function_evaluations = maximum_number_function_evaluations
        self._ftol = ftol
        self._gtol = gtol
        self._xtol = xtol
        self._optimization_method = optimization_method
        self._prepare_data(data)

    @classmethod
    def from_yaml_file(cls, filename: str) -> Scheme:

        try:
            with open(filename) as f:
                try:
                    scheme = yaml.safe_load(f)
                except Exception as e:
                    raise ValueError(f"Error parsing scheme: {e}")
        except Exception as e:
            raise OSError(f"Error opening scheme: {e}")

        if "model" not in scheme:
            raise ValueError("Model file not specified.")

        try:
            model = glotaran.read_model_from_yaml_file(scheme["model"])
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

        if "parameters" not in scheme:
            raise ValueError("Parameters file not specified.")

        path = scheme["parameters"]
        fmt = scheme.get("parameter_format", None)
        try:
            parameters = glotaran.parameter.ParameterGroup.from_file(path, fmt)
        except Exception as e:
            raise ValueError(f"Error loading parameters: {e}")

        if "data" not in scheme:
            raise ValueError("No data specified.")

        data = {}
        for label, path in scheme["data"].items():
            path = pathlib.Path(path)

            fmt = path.suffix[1:] if path.suffix != "" else "nc"
            if "data_format" in scheme:
                fmt = scheme["data_format"]

            try:
                data[label] = glotaran.io.read_data_file(path, fmt=fmt)
            except Exception as e:
                raise ValueError(f"Error loading dataset '{label}': {e}")

        optimization_method = scheme.get("optimization_method", "TrustRegionReflection")
        nnls = scheme.get("non-negative-least-squares", False)
        nfev = scheme.get("maximum-number-function-evaluations", None)
        ftol = scheme.get("ftol", 1e-8)
        gtol = scheme.get("gtol", 1e-8)
        xtol = scheme.get("xtol", 1e-8)
        group_tolerance = scheme.get("group_tolerance", 0.0)
        return cls(
            model=model,
            parameters=parameters,
            data=data,
            non_negative_least_squares=nnls,
            maximum_number_function_evaluations=nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            group_tolerance=group_tolerance,
            optimization_method=optimization_method,
        )

    @property
    def model(self) -> Model:
        return self._model

    @property
    def parameters(self) -> ParameterGroup:
        return self._parameters

    @property
    def data(self) -> dict[str, xr.DataArray | xr.Dataset]:
        return self._data

    @property
    def non_negative_least_squares(self) -> bool:
        return self._non_negative_least_squares

    @property
    def maximum_number_function_evaluations(self) -> int:
        return self._maximum_number_function_evaluations

    @property
    def group_tolerance(self) -> float:
        return self._group_tolerance

    @property
    def ftol(self) -> float:
        return self._ftol

    @property
    def gtol(self) -> float:
        return self._gtol

    @property
    def xtol(self) -> float:
        return self._xtol

    @property
    def optimization_method(self) -> str:
        return self._optimization_method

    def problem_list(self) -> list[str]:
        """Returns a list with all problems in the model and missing parameters."""
        return self.model.problem_list(self.parameters)

    def validate(self) -> str:
        """Returns a string listing all problems in the model and missing parameters."""
        return self.model.validate(self.parameters)

    def valid(self, parameters: ParameterGroup = None) -> bool:
        """Returns `True` if there are no problems with the model or the parameters,
        else `False`."""
        return self.model.valid(parameters)

    def _transpose_dataset(self, dataset):
        new_dims = [self.model.model_dimension, self.model.global_dimension]
        new_dims += [
            dim
            for dim in dataset.dims
            if dim not in [self.model.model_dimension, self.model.global_dimension]
        ]

        return dataset.transpose(*new_dims)

    def _prepare_data(self, data: dict[str, xr.DataArray | xr.Dataset]):
        self._data = {}
        for label, dataset in data.items():
            if self.model.model_dimension not in dataset.dims:
                raise ValueError(
                    "Missing coordinates for dimension "
                    f"'{self.model.model_dimension}' in data for dataset "
                    f"'{label}'"
                )
            if self.model.global_dimension not in dataset.dims:
                raise ValueError(
                    "Missing coordinates for dimension "
                    f"'{self.model.global_dimension}' in data for dataset "
                    f"'{label}'"
                )
            if isinstance(dataset, xr.DataArray):
                dataset = dataset.to_dataset(name="data")

            dataset = self._transpose_dataset(dataset)
            self._add_weight(label, dataset)

            # This protects transposing when getting data with svd in it
            if "data_singular_values" in dataset and (
                dataset.coords["right_singular_value_index"].size
                != dataset.coords[self.model.global_dimension].size
            ):
                dataset = dataset.rename(
                    right_singular_value_index="right_singular_value_indexTMP"
                )
                dataset = dataset.rename(left_singular_value_index="right_singular_value_index")
                dataset = dataset.rename(right_singular_value_indexTMP="left_singular_value_index")
                dataset = dataset.rename(right_singular_vectors="right_singular_value_vectorsTMP")
                dataset = dataset.rename(
                    left_singular_value_vectors="right_singular_value_vectors"
                )
                dataset = dataset.rename(
                    right_singular_value_vectorsTMP="left_singular_value_vectors"
                )
            new_dims = [self.model.model_dimension, self.model.global_dimension]
            new_dims += [
                dim
                for dim in dataset.dims
                if dim not in [self.model.model_dimension, self.model.global_dimension]
            ]

            self._data[label] = dataset

    def markdown(self):
        s = self.model.markdown(parameters=self.parameters)

        s += "\n\n"
        s += "__Scheme__\n\n"

        s += f"* *nnls*: {self.nnls}\n"
        s += f"* *nfev*: {self.nfev}\n"
        s += f"* *group_tolerance*: {self.group_tolerance}\n"

        return s

    def _add_weight(self, label, dataset):

        # if the user supplies a weight we ignore modeled weights
        if "weight" in dataset:
            if any(label in weight.datasets for weight in self.model.weights):
                warnings.warn(
                    f"Ignoring model weight for dataset '{label}'"
                    " because weight is already supplied by dataset."
                )
            return

        global_axis = dataset.coords[self.model.global_dimension]
        model_axis = dataset.coords[self.model.model_dimension]

        for weight in self.model.weights:
            if label in weight.datasets:
                if "weight" not in dataset:
                    dataset["weight"] = xr.DataArray(
                        np.ones_like(dataset.data), coords=dataset.data.coords
                    )

                idx = {}
                if weight.global_interval is not None:
                    idx[self.model.global_dimension] = _get_min_max_from_interval(
                        weight.global_interval, global_axis
                    )
                if weight.model_interval is not None:
                    idx[self.model.model_dimension] = _get_min_max_from_interval(
                        weight.model_interval, model_axis
                    )
                dataset.weight[idx] *= weight.value


def _get_min_max_from_interval(interval, axis):
    minimum = np.abs(axis.values - interval[0]).argmin() if not np.isinf(interval[0]) else 0
    maximum = (
        np.abs(axis.values - interval[1]).argmin() + 1 if not np.isinf(interval[1]) else axis.size
    )
    return slice(minimum, maximum)
