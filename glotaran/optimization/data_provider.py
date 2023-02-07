"""Module containing the data provider classes."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model import DatasetGroup
from glotaran.model import Model
from glotaran.model.dataset_model import get_dataset_model_model_dimension
from glotaran.model.dataset_model import has_dataset_model_global_model
from glotaran.project import Scheme

if TYPE_CHECKING:
    from typing import Literal

    from glotaran.typing.types import ArrayLike


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
        self._data: dict[str, ArrayLike] = {}
        self._weight: dict[str, ArrayLike | None] = {}
        self._flattened_data: dict[str, ArrayLike] = {}
        self._flattened_weight: dict[str, ArrayLike | None] = {}
        self._model_axes: dict[str, ArrayLike] = {}
        self._model_dimensions: dict[str, str] = {}
        self._global_axes: dict[str, ArrayLike] = {}
        self._global_dimensions: dict[str, str] = {}

        for label, dataset_model in dataset_group.dataset_models.items():
            dataset = scheme.data[label]
            model_dimension = get_dataset_model_model_dimension(dataset_model)
            self._model_axes[label] = dataset.coords[model_dimension].data
            self._model_dimensions[label] = model_dimension
            global_dimension = self.infer_global_dimension(model_dimension, dataset.data.dims)
            self._global_axes[label] = dataset.coords[global_dimension].data
            self._global_dimensions[label] = global_dimension

            self._weight[label] = self.get_from_dataset(
                dataset, "weight", model_dimension, global_dimension
            )
            self.add_model_weight(scheme.model, label, model_dimension, global_dimension)

            self._data[label] = self.get_from_dataset(  # type:ignore[assignment]
                dataset, "data", model_dimension, global_dimension
            )
            if self._weight[label] is not None:
                self._data[label] *= self._weight[label]

            if has_dataset_model_global_model(dataset_model):
                self._flattened_data[label] = self._data[label].T.flatten()
                self._flattened_weight[label] = (
                    self._weight[label].T.flatten()  # type:ignore[union-attr]
                    if self._weight[label] is not None
                    else None
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
    ) -> ArrayLike | None:
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
        ArrayLike | None
            The copy of the data. None if name is not present in dataset.
        """
        data = None
        if name in dataset:
            data = dataset[name].data.copy()
            if dataset[name].dims != (model_dimension, global_dimension):
                data = data.T
        return data

    @staticmethod
    def get_axis_slice_from_interval(interval: tuple[float, float], axis: ArrayLike) -> slice:
        """Get a slice of indices from a min max tuple and for an axis.

        Parameters
        ----------
        interval : tuple[float, float]
            The min max tuple.
        axis : ArrayLike
            The axis to slice.

        Returns
        -------
        slice
            The slice of indices.
        """
        interval_min = interval[0]
        interval_max = interval[1]

        if interval_min > interval_max:
            interval_min, interval_max = interval_max, interval_min

        minimum = 0 if np.isinf(interval_min) else np.abs(axis - interval_min).argmin()
        maximum = (
            axis.size - 1 if np.isinf(interval_max) else np.abs(axis - interval_max).argmin() + 1
        )

        return slice(minimum, maximum)

    def add_model_weight(
        self,
        model: Model,
        dataset_label: str,
        model_dimension: str,
        global_dimension: str,
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
        model_weights = [weight for weight in model.weights if dataset_label in weight.datasets]
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
            if model_weight.global_interval is not None:
                idx[global_dimension] = self.get_axis_slice_from_interval(
                    model_weight.global_interval, global_axis
                )
            if model_weight.model_interval is not None:
                idx[model_dimension] = self.get_axis_slice_from_interval(
                    model_weight.model_interval, model_axis
                )
            weight[idx] *= model_weight.value

        self._weight[dataset_label] = weight.data

    def get_data(self, dataset_label: str) -> ArrayLike:
        """Get data for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        ArrayLike
            The data.
        """
        return self._data[dataset_label]

    def get_weight(self, dataset_label: str) -> ArrayLike | None:
        """Get weight for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        ArrayLike | None
            The weight.
        """
        return self._weight[dataset_label]

    def get_flattened_data(self, dataset_label: str) -> ArrayLike:
        """Get flattened data for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        ArrayLike
            The flattened data.
        """
        return self._flattened_data[dataset_label]

    def get_flattened_weight(self, dataset_label: str) -> ArrayLike | None:
        """Get flattened weight for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        ArrayLike | None
            The flattened weight.
        """
        return self._flattened_weight[dataset_label]

    def get_model_axis(self, dataset_label: str) -> ArrayLike:
        """Get the model axis for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        ArrayLike
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

    def get_global_axis(self, dataset_label: str) -> ArrayLike:
        """Get the global axis for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the data.

        Returns
        -------
        ArrayLike
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
        self._aligned_global_axis, self._aligned_data = self.align_data(aligned_global_axes)
        self._aligned_dataset_indices = self.align_dataset_indices(aligned_global_axes)
        self._aligned_group_labels, self._group_definitions = self.align_groups(
            aligned_global_axes
        )
        self._aligned_weights = self.align_weights(aligned_global_axes)

    @staticmethod
    def align_index(
        index: int,
        target_axis: ArrayLike,
        tolerance: float,
        method: Literal["nearest", "backward", "forward"],
    ) -> int:
        """Align an index on a target axis.

        Parameters
        ----------
        index : int
            The index to align.
        target_axis : ArrayLike
            The axis to align the index on.
        tolerance : float
            The alignment tolerance.
        method : Literal["nearest", "backward", "forward"]
            The alignment method.

        Returns
        -------
        int
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
    def aligned_global_axis(self) -> ArrayLike:
        """Get the aligned global axis for the dataset group.

        Returns
        -------
        ArrayLike
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

    def get_aligned_dataset_indices(self, index: int) -> ArrayLike:
        """Get the aligned dataset indices for an index.

        Parameters
        ----------
        index : int
            The index on the aligned global axis.

        Returns
        -------
        ArrayLike
            The aligned dataset indices.
        """
        return self._aligned_dataset_indices[index]

    def get_aligned_data(self, index: int) -> ArrayLike:
        """Get the aligned data for an index.

        Parameters
        ----------
        index : int
            The index on the aligned global axis.

        Returns
        -------
        ArrayLike
            The aligned data.
        """
        return self._aligned_data[index]

    def get_aligned_weight(self, index: int) -> ArrayLike | None:
        """Get the aligned weight for an index.

        Parameters
        ----------
        index : int
            The index on the aligned global axis.

        Returns
        -------
        ArrayLike | None
            The aligned weight.
        """
        return self._aligned_weights[index]

    def create_aligned_global_axes(self, scheme: Scheme) -> dict[str, ArrayLike]:
        """Create aligned global axes for the dataset group.

        Parameters
        ----------
        scheme : Scheme
            The optimization scheme.

        Returns
        -------
        dict[str, ArrayLike]
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

    def align_data(
        self, aligned_global_axes: dict[str, ArrayLike]
    ) -> tuple[ArrayLike, list[ArrayLike]]:
        """Align the data in a dataset group.

        Parameters
        ----------
        aligned_global_axes : dict[str, ArrayLike]
            The aligned global axes.

        Returns
        -------
        tuple[ArrayLike, list[ArrayLike]]
            The aligned global axis and data.
        """
        aligned_data = xr.concat(
            [
                xr.DataArray(
                    self.get_data(label),
                    dims=["model", "global"],
                    coords={"global": axis},
                )
                for label, axis in aligned_global_axes.items()
            ],
            dim="model",
        )
        aligned_global_axis = aligned_data.coords["global"].data
        return (
            aligned_global_axis,
            [
                aligned_data.isel({"global": i}).dropna(dim="model").data
                for i in range(aligned_global_axis.size)
            ],
        )

    def align_dataset_indices(self, aligned_global_axes: dict[str, ArrayLike]) -> list[ArrayLike]:
        """Align the global indices in a dataset group.

        Parameters
        ----------
        aligned_global_axes : dict[str, ArrayLike]
            The aligned global axes.

        Returns
        -------
        list[ArrayLike]
        The aligned dataset indices.
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
        return [
            aligned_indices.isel({"global": i}).dropna(dim="dataset").data.astype(int)
            for i in range(self._aligned_global_axis.size)
        ]

    @staticmethod
    def align_groups(
        aligned_global_axes: dict[str, ArrayLike]
    ) -> tuple[ArrayLike, dict[str, list[str]]]:
        """Align the groups in a dataset group.

        Parameters
        ----------
        aligned_global_axes : dict[str, ArrayLike]
            The aligned global axes.

        Returns
        -------
        tuple[ArrayLike, dict[str, list[str]]]
            The aligned grouplabels and group definitions.
        """
        aligned_groups = xr.concat(
            [
                xr.DataArray(np.full(len(axis), label), dims=["global"], coords={"global": axis})
                for label, axis in aligned_global_axes.items()
            ],
            dim="dataset",
            fill_value="",
        )
        # for every element along the global axis, concatenate all dataset labels
        # into an ndarray of shape (len(global,)
        # as an alternative to the more elegant xarray built-in which is limited to 32 datasets
        # aligned_group_labels = aligned_groups.str.join(dim="dataset").data
        aligned_group_labels = np.asarray(
            ["".join(sub_arr.values) for _, sub_arr in aligned_groups.groupby("global")]
        )

        group_definitions: dict[str, list[str]] = {}
        for i, group_label in enumerate(aligned_group_labels):
            if group_label not in group_definitions:
                group_definitions[group_label] = list(
                    filter(lambda label: label != "", aligned_groups.isel({"global": i}).data)
                )
        return aligned_group_labels, group_definitions

    def align_weights(self, aligned_global_axes: dict[str, ArrayLike]) -> list[ArrayLike | None]:
        """Align the weights in a dataset group.

        Parameters
        ----------
        aligned_global_axes : dict[str, ArrayLike]
            The aligned global axes.

        Returns
        -------
        list[ArrayLike | None]
            The aligned weights.
        """
        all_weights = {
            label: xr.DataArray(
                weight,
                dims=["model", "global"],
                coords={"global": aligned_global_axes[label]},
            )
            for label, weight in self._weight.items()
            if weight is not None
        }

        aligned_weights: list[ArrayLike | None] = [None] * self._aligned_global_axis.size
        if all_weights:
            for i, group_label in enumerate(self._aligned_group_labels):
                group_dataset_labels = self._group_definitions[group_label]
                if any(label in all_weights for label in group_dataset_labels):
                    index_weights = []
                    for label in group_dataset_labels:
                        if label in all_weights:
                            index_weights.append(
                                all_weights[label]
                                .sel({"global": self._aligned_global_axis[i]})
                                .data
                            )
                        else:
                            size = self.get_model_axis(label).size
                            index_weights.append(np.ones(size))
                    aligned_weights[i] = np.concatenate(index_weights)

        return aligned_weights
