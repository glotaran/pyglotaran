import warnings
from numbers import Number

import numpy as np
import xarray as xr

from glotaran.model import DatasetGroup
from glotaran.model import Model
from glotaran.project import Scheme


class DataProvider:
    def __init__(self, scheme: Scheme, dataset_group: DatasetGroup):

        self._data = {}
        self._weight = {}
        self._model_axes = {}
        self._global_axes = {}

        for label, dataset_model in dataset_group.dataset_models.items():

            dataset = scheme.data[label]
            model_dimension = dataset_model.get_model_dimension()
            global_dimension = self.infer_global_dimension(model_dimension, dataset.data.dims)

            self._data[label] = self.get_from_dataset(
                dataset, "data", model_dimension, global_dimension
            )
            self._weight[label] = self.get_from_dataset(
                dataset, "weight", model_dimension, global_dimension
            )
            self._model_axes[label] = dataset.coords[model_dimension].data
            self._global_axes[label] = dataset.coords[global_dimension].data

            self.add_model_weight(scheme.model, label, model_dimension, global_dimension)

    @staticmethod
    def infer_global_dimension(model_dimension: str, dimensions: tuple[str]) -> str:
        return next(dim for dim in dimensions if dim != model_dimension)

    @staticmethod
    def get_from_dataset(
        dataset: xr.Dataset, name: str, model_dimension: str, global_dimension: str
    ) -> np.typing.ArrayLike | None:
        data = None
        if name in dataset:
            data = dataset[name].data
            if dataset[name].dims != (model_dimension, global_dimension):
                data = data.T
        return data

    @staticmethod
    def get_axis_slice_from_interval(
        interval: tuple[Number, Number], axis: np.typing.ArrayLike
    ) -> slice:
        minimum = np.abs(axis.values - interval[0]).argmin() if not np.isinf(interval[0]) else 0
        maximum = (
            np.abs(axis.values - interval[1]).argmin() + 1
            if not np.isinf(interval[1])
            else axis.size
        )
        return slice(minimum, maximum)

    def add_model_weight(
        self, model: Model, label: str, model_dimension: str, global_dimension: str
    ):
        model_weights = [weight for weight in model.weights if label in weight.dataset]

        if not model_weights:
            return

        if self._weights[label]:
            warnings.warn(
                f"Ignoring model weight for dataset '{label}'"
                " because weight is already supplied by dataset."
            )
            return
        model_axis = self._model_axes[label]
        global_axis = self._global_axes[label]
        weight = xr.DataArray(
            np.ones((model_axis.size, global_axis.size)),
            coords=(
                (model_dimension, model_axis),
                (global_dimension, global_axis),
            ),
        )
        for model_weight in model_weights:

            idx = {}
            if model_weight.global_interval is not None:
                idx[global_dimension] = self.get_axis_slice_from_interval(
                    model_weight.global_interval, global_axis
                )
            if model_weight.model_interval is not None:
                idx[model_dimension] = self.get_axis_slice_from_interval(
                    model_weight.model_interval, model_axis
                )
            weight[idx] *= model_weight.value

        self._weights[label] = weight.data

    def get_data(self, dataset_label: str) -> np.typing.ArrayLike:
        return self._data[dataset_label]

    def get_weight(self, dataset_label: str) -> np.typing.ArrayLike | None:
        return self._weight[dataset_label]

    def get_model_axis(self, dataset_label: str) -> np.typing.ArrayLike:
        return self._model_axes[dataset_label]

    def get_global_axis(self, dataset_label: str) -> np.typing.ArrayLike:
        return self._global_axes[dataset_label]
