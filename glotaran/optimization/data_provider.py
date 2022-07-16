"""Module containing the data provider classes."""
from __future__ import annotations

import warnings
from numbers import Number
from typing import Literal

import numpy as np
import xarray as xr

from glotaran.model import DatasetGroup
from glotaran.model import Model
from glotaran.project import Scheme


class AlignDatasetError(ValueError):
    """Indicates that datasets can not be aligned."""

    def __init__(self):
        """Initialize a AlignDatasetError."""
        super().__init__(
            "Cannot link datasets, aligning is ambiguous. \n\n"
            "Try to lower link tolerance or change the alignment method."
        )


class DataProvider:
    """A class to provide prepared data for optimization."""

    def __init__(self, scheme: Scheme, dataset_group: DatasetGroup):
        """Initialize a data provider for a scheme and a dataset_group.

        Parameters
        ----------
        scheme : Scheme
            The optimization scheme.
        dataset_group : DatasetGroup
            The dataset group.
        """
        self._data: dict[str, np.typing.ArrayLike] = {}
        self._weight: dict[str, np.typing.ArrayLike] = {}
        self._flattened_data: dict[str, np.typing.ArrayLike] = {}
        self._flattened_weight: dict[str, np.typing.ArrayLike] = {}
        self._model_axes: dict[str, np.typing.ArrayLike] = {}
        self._model_dimensions: dict[str, str] = {}
        self._global_axes: dict[str, np.typing.ArrayLike] = {}
        self._global_dimensions: dict[str, str] = {}

        for label, dataset_model in dataset_group.dataset_models.items():

            dataset = scheme.data[label]
            model_dimension = dataset_model.get_model_dimension()
            self._model_axes[label] = dataset.coords[model_dimension].data
            self._model_dimensions[label] = model_dimension
            global_dimension = self.infer_global_dimension(model_dimension, dataset.data.dims)
            self._global_axes[label] = dataset.coords[global_dimension].data
            self._global_dimensions[label] = global_dimension

            self._weight[label] = self.get_from_dataset(
                dataset, "weight", model_dimension, global_dimension
            )
            self.add_model_weight(scheme.model, label, model_dimension, global_dimension)

            self._data[label] = self.get_from_dataset(
                dataset, "data", model_dimension, global_dimension
            )
            if self._weight[label] is not None:
                self._data[label] *= self._weight[label]

            if dataset_model.has_global_model():
                self._flattened_data[label] = self._data[label].T.flatten()
                self._flattened_weight[label] = (
                    self._weight[label].T.flatten() if self._weight[label] is not None else None
                )

    @staticmethod
    def infer_global_dimension(model_dimension: str, dimensions: tuple[str]) -> str:
        """Infer the name of the global dimension from tuple of dimensions.

        Parameters
        ----------
        model_dimension : str
            The model dimension.
        dimensions : tuple[str]
            The dimensions tuple to infer from.

        Returns
        -------
        str
            The inferred name of the global dimension.
        """
        return next(dim for dim in dimensions if dim != model_dimension)

    @staticmethod
    def get_from_dataset(
        dataset: xr.Dataset, name: str, model_dimension: str, global_dimension: str
    ) -> np.typing.ArrayLike | None:
        """Get a copy of data from a dataset with dimensions (model, global).

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to retrieve from.
        name : str
            The name of the data to retrieve.
        model_dimension : str
            The model dimension.
        global_dimension : str
            The global dimension.

        Returns
        -------
        np.typing.ArrayLike | None
            The copy of the data. None if name is not present in dataset.
        """
        data = None
        if name in dataset:
            data = dataset[name].data.copy()
            if dataset[name].dims != (model_dimension, global_dimension):
                data = data.T
        return data

    @staticmethod
    def get_axis_slice_from_interval(
        interval: tuple[Number, Number], axis: np.typing.ArrayLike
    ) -> slice:
        """Get a slice of indices from a min max tuple and for an axis.

        Parameters
        ----------
        interval : tuple[Number, Number]
            The min max tuple.
        axis : np.typing.ArrayLike
            The axis to slice.

        Returns
        -------
        slice
            The slice of indices.
        """
        minimum = 0 if np.isinf(interval[0]) else np.abs(axis - interval[0]).argmin()
        maximum = axis.size if np.isinf(interval[1]) else np.abs(axis - interval[1]).argmin() + 1

        return slice(minimum, maximum)

    def add_model_weight(
        self, model: Model, dataset_label: str, model_dimension: str, global_dimension: str
    ):
        """Add model weight to data.

        Parameters
        ----------
        model : Model
            The model.
        dataset_label : str
            The label of the data.
        model_dimension : str
            The model dimension.
        global_dimension : str
            The global dimension.
        """
        model_weights = [
            weight
            for weight in model.weights  # type:ignore[attr-defined]
            if dataset_label in weight.datasets  # type:ignore[attr-defined]
        ]
        if not model_weights:
            return

        if self._weight[dataset_label]:
            warnings.warn(
                f"Ignoring model weight for dataset '{dataset_label}'"
                " because weight is already supplied by dataset."
            )
            return
        model_axis = self._model_axes[dataset_label]
        global_axis = self._global_axes[dataset_label]
        weight = xr.DataArray(
            np.ones((model_axis.size, global_axis.size)),
            coords=(
                (model_dimension, model_axis),
                (global_dimension, global_axis),
            ),
        )
        for model_weight in model_weights:

            idx = {}
            if model_weight.global_interval is not None:  # type:ignore[attr-defined]
                idx[global_dimension] = self.get_axis_slice_from_interval(
                    model_weight.global_interval, global_axis  # type:ignore[attr-defined]
                )
            if model_weight.model_interval is not None:  # type:ignore[attr-defined]
                idx[model_dimension] = self.get_axis_slice_from_interval(
                    model_weight.model_interval, model_axis  # type:ignore[attr-defined]
                )
            weight[idx] *= model_weight.value  # type:ignore[attr-defined]

        self._weight[dataset_label] = weight.data

    def get_data(self, dataset_label: str) -> np.typing.ArrayLike:
        """Get data for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        np.typing.ArrayLike
            The data.
        """
        return self._data[dataset_label]

    def get_weight(self, dataset_label: str) -> np.typing.ArrayLike | None:
        """Get weight for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        np.typing.ArrayLike | None
            The weight.
        """
        return self._weight[dataset_label]

    def get_flattened_data(self, dataset_label: str) -> np.typing.ArrayLike:
        """Get flattened data for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        np.typing.ArrayLike
            The flattened data.
        """
        return self._flattened_data[dataset_label]

    def get_flattened_weight(self, dataset_label: str) -> np.typing.ArrayLike | None:
        """Get flattened weight for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        np.typing.ArrayLike | None
            The flattened weight.
        """
        return self._flattened_weight[dataset_label]

    def get_model_axis(self, dataset_label: str) -> np.typing.ArrayLike:
        """Get the model axis for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        np.typing.ArrayLike
            The model axis.
        """
        return self._model_axes[dataset_label]

    def get_model_dimension(self, dataset_label: str) -> str:
        """Get the model dimension for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        str
            The model dimension.
        """
        return self._model_dimensions[dataset_label]

    def get_global_axis(self, dataset_label: str) -> np.typing.ArrayLike:
        """Get the global axis for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        np.typing.ArrayLike
            The global axis.
        """
        return self._global_axes[dataset_label]

    def get_global_dimension(self, dataset_label: str) -> str:
        """Get the global dimension for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        str
            The global dimension.
        """
        return self._global_dimensions[dataset_label]


class DataProviderLinked(DataProvider):
    """A class to provide aligned data for optimization."""

    def __init__(
        self,
        scheme: Scheme,
        dataset_group: DatasetGroup,
    ):
        """Initialize a linked data provider for a scheme and a dataset_group.

        Parameters
        ----------
        scheme : Scheme
            The optimization scheme.
        dataset_group : DatasetGroup
            The dataset group.
        """
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
        """Align an index on a target axis.

        Parameters
        ----------
        index : Number
            The index to align.
        target_axis : np.typing.ArrayLike
            The axis to align the index on.
        tolerance : Number
            The alignment tolerance.
        method : Literal["nearest", "backward", "forward"]
            The alignment method.

        Returns
        -------
        Number
            The aligned index.
        """
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
        """Get the aligned global axis for the dataset group.

        Returns
        -------
        np.typing.ArrayLike
            The aligned global axis.
        """
        return self._aligned_global_axis

    @property
    def group_definitions(self) -> dict[str, list[str]]:
        """Get the group definitions for the dataset group.

        Returns
        -------
        dict[str, list[str]]
            The group definitions.
        """
        return self._group_definitions

    def get_aligned_group_label(self, index: int) -> str:
        """Get the group label for an index.

        Parameters
        ----------
        index : int
            The index on the aligned global axis.

        Returns
        -------
        str
            The aligned group label.
        """
        return self._aligned_group_labels[index]

    def get_aligned_dataset_indices(self, index: int) -> list[int]:
        """Get the aligned dataset indices for an index.

        Parameters
        ----------
        index : int
            The index on the aligned global axis.

        Returns
        -------
        list[int]
            The aligned dataset indices.
        """
        return self._aligned_dataset_indices[index]

    def get_aligned_data(self, index: int) -> np.typing.ArrayLike:
        """Get the aligned data for an index.

        Parameters
        ----------
        index : int
            The index on the aligned global axis.

        Returns
        -------
        np.typing.ArrayLike
            The aligned data.
        """
        return self._aligned_data[index]

    def get_aligned_weight(self, index: int) -> np.typing.ArrayLike | None:
        """Get the aligned weight for an index.

        Parameters
        ----------
        index : int
            The index on the aligned global axis.

        Returns
        -------
        np.typing.ArrayLike | None
            The aligned weight.
        """
        return self._aligned_weights[index]

    def create_aligned_global_axes(self, scheme: Scheme) -> dict[str, np.typing.ArrayLike]:
        """Create aligned global axes for the dataset group.

        Parameters
        ----------
        scheme : Scheme
            The optimization scheme.

        Returns
        -------
        dict[str, np.typing.ArrayLike]
            The aligned global axes.

        Raises
        ------
        AlignDatasetError
            Raised when dataset alignment is ambiguous.
        """
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
                    raise AlignDatasetError()
                aligned_axis_values = np.unique(
                    np.concatenate([aligned_axis_values, aligned_global_axis])
                )
            aligned_global_axes[label] = aligned_global_axis
        return aligned_global_axes

    def align_data(self, aligned_global_axes: dict[str, np.typing.ArrayLike]):
        """Align the data in a dataset group.

        Parameters
        ----------
        aligned_global_axes : dict[str, np.typing.ArrayLike]
            The aligned global axes.
        """
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
        """Align the global indices in a dataset group.

        Parameters
        ----------
        aligned_global_axes : dict[str, np.typing.ArrayLike]
            The aligned global axes.
        """
        aligned_indices = xr.concat(
            [
                xr.DataArray(
                    np.arange(len(axis), dtype=int),
                    dims=["global"],
                    coords={"global": axis},
                )
                for axis in aligned_global_axes.values()
            ],
            dim="dataset",
        )
        self._aligned_dataset_indices = [
            aligned_indices.isel({"global": i}).dropna(dim="dataset").data.astype(int)
            for i in range(self._aligned_global_axis.size)
        ]

    def align_groups(self, aligned_global_axes: dict[str, np.typing.ArrayLike]):
        """Align the groups in a dataset group.

        Parameters
        ----------
        aligned_global_axes : dict[str, np.typing.ArrayLike]
            The aligned global axes.
        """
        aligned_group_labels = xr.concat(
            [
                xr.DataArray(np.full(len(axis), label), dims=["global"], coords={"global": axis})
                for label, axis in aligned_global_axes.items()
            ],
            dim="dataset",
            fill_value="",
        )
        self._aligned_group_labels = aligned_group_labels.str.join(dim="dataset").data
        self._group_definitions: dict[str, list[str]] = {}
        for i, group_label in enumerate(self._aligned_group_labels):
            if group_label not in self._group_definitions:
                self._group_definitions[group_label] = list(
                    filter(lambda l: l != "", aligned_group_labels.isel({"global": i}).data)
                )

    def align_weights(self, aligned_global_axes: dict[str, np.typing.ArrayLike]):
        """Align the weights in a dataset group.

        Parameters
        ----------
        aligned_global_axes : dict[str, np.typing.ArrayLike]
            The aligned global axes.
        """
        self._aligned_weights = [None] * self._aligned_global_axis.size
        aligned_weights = {
            label: xr.DataArray(
                weight, dims=["model", "global"], coords={"global": aligned_global_axes[label]}
            )
            for label, weight in self._weight.items()
            if weight is not None
        }
        if not aligned_weights:
            return

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
