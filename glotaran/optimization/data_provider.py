import warnings
from numbers import Number
from typing import Literal

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
            self._model_axes[label] = dataset.coords[model_dimension].data
            self._global_axes[label] = dataset.coords[global_dimension].data

            self._weight[label] = self.get_from_dataset(
                dataset, "weight", model_dimension, global_dimension
            )
            self.add_model_weight(scheme.model, label, model_dimension, global_dimension)

            self._data[label] = self.get_from_dataset(
                dataset, "data", model_dimension, global_dimension
            )
            if self._weight[label] is not None:
                self._data[label] *= self._weight[label]

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


class DataProviderLinked(DataProvider):
    def __init__(
        self,
        scheme: Scheme,
        dataset_group: DatasetGroup,
    ):
        super().__init__(scheme, dataset_group)
        aligned_global_axes = self.create_aligned_global_axes(scheme)
        self.align_data(aligned_global_axes)
        self.align_global_indices(aligned_global_axes)
        self.align_groups(aligned_global_axes)
        self.align_weights(aligned_global_axes)

    @staticmethod
    def align_index(
        index: Number,
        target_axis: np.typing.ArrayLike,
        tolerance: Number,
        method: Literal["nearest", "backward", "forward"],
    ) -> Number:
        diff = target_axis - index

        if method == "forward":
            diff = diff[diff >= 0]
        elif method == "backward":
            diff = diff[diff <= 0]

        diff = np.abs(diff)

        if len(diff) > 0 and diff.min() <= tolerance:
            index = target_axis[diff.argmin()]
        return index

    @property
    def aligned_global_axis(self) -> np.typing.ArrayLike:
        return self._aligned_global_axis

    @property
    def group_definitions(self) -> dict[str, list[str]]:
        return self._group_definitions

    def get_aligned_group_labels(self, index: int) -> str:
        return self._aligned_group_labels[index]

    def get_aligned_dataset_indices(self, index: int) -> list[Number]:
        return self._aligned_dataset_indices[index]

    def get_aligned_data(self, index: int) -> np.typing.ArrayLike:
        return self._aligned_data[index]

    def get_aligned_weight(self, index: int) -> np.typing.ArrayLike | None:
        return self._aligned_weights[index]

    def create_aligned_global_axes(self, scheme: Scheme) -> dict[str, np.typing.ArrayLike]:
        aligned_axis_values = None
        aligned_global_axes = {}
        for label, global_axis in self._global_axes.items():

            aligned_global_axis = global_axis
            if aligned_axis_values is None:
                aligned_axis_values = aligned_global_axis
            else:
                aligned_global_axis = [
                    self.align_index(
                        index,
                        aligned_axis_values,
                        scheme.clp_link_tolerance,
                        scheme.clp_link_method,
                    )
                    for index in aligned_global_axis
                ]
                if len(np.unique(aligned_global_axis)) != len(aligned_global_axis):
                    raise ValueError(
                        "Cannot link datasets, aligning is ambiguous. \n\n"
                        "Try to lower link tolerance or change the alignment method."
                    )
                aligned_axis_values = np.unique(
                    np.concatenate([aligned_axis_values, aligned_global_axis])
                )
            aligned_global_axes[label] = aligned_global_axis
        return aligned_global_axes

    def align_data(self, aligned_global_axes: dict[str, np.typing.ArrayLike]):

        aligned_data = xr.concat(
            [
                xr.DataArray(
                    self.get_data(label), dims=["model", "global"], coords={"global": axis}
                )
                for label, axis in aligned_global_axes.items()
            ],
            dim="model",
        )
        self._aligned_global_axis = aligned_data.coords["global"].data
        self._aligned_data = [
            aligned_data.isel({"global": i}).dropna(dim="model").data
            for i in range(self._aligned_global_axis.size)
        ]

    def align_global_indices(self, aligned_global_axes: dict[str, np.typing.ArrayLike]):
        aligned_indices = xr.concat(
            [
                xr.DataArray(self.get_global_axis(label), dims=["global"], coords={"global": axis})
                for label, axis in aligned_global_axes.items()
            ],
            dim="dataset",
        )
        self._aligned_dataset_indices = [
            aligned_indices.isel({"global": i}).dropna(dim="dataset").data
            for i in range(self._aligned_global_axis.size)
        ]

    def align_groups(self, aligned_global_axes: dict[str, np.typing.ArrayLike]):
        aligned_group_labels = xr.concat(
            [
                xr.DataArray(np.full(len(axis), label), dims=["global"], coords={"global": axis})
                for label, axis in aligned_global_axes.items()
            ],
            dim="dataset",
            fill_value="",
        )
        self._aligned_group_labels = aligned_group_labels.str.join(dim="dataset").data
        self._group_definitions = {}
        for i, group_label in enumerate(self._aligned_group_labels):
            if group_label not in self._group_definitions:
                self._group_definitions[group_label] = list(
                    filter(lambda l: l != "", aligned_group_labels.isel({"global": i}).data)
                )

    def align_weights(self, aligned_global_axes: dict[str, np.typing.ArrayLike]):
        aligned_weights = {
            label: xr.DataArray(
                weight, dims=["model", "global"], coords={"global": aligned_global_axes[label]}
            )
            for label, weight in self._weight.items()
            if weight is not None
        }
        if not aligned_weights:
            return

        self._aligned_weights = [None] * self._aligned_global_axis.size
        for i, group_label in enumerate(self._aligned_group_labels):
            group_dataset_labels = self._group_definitions[group_label]
            if any(label in aligned_weights for label in group_dataset_labels):
                index_weights = []
                for label in group_dataset_labels:
                    if label in aligned_weights:
                        index_weights.append(
                            aligned_weights[label]
                            .sel({"global": self._aligned_global_axis[i]})
                            .data
                        )
                    else:
                        size = self.get_model_axis(label).size
                        index_weights.append(np.ones(size))
                self._aligned_weights[i] = np.concatenate(index_weights)
