"""Module containing the data provider classes."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
import xarray as xr

from glotaran.model.data_model import DataModel
from glotaran.model.data_model import get_data_model_dimension
from glotaran.model.data_model import is_data_model_global

if TYPE_CHECKING:
    from glotaran.model.experiment_model import ExperimentModel
    from glotaran.parameter import Parameter
    from glotaran.typing.types import ArrayLike


class AlignDatasetError(ValueError):
    """Indicates that datasets can not be aligned."""

    def __init__(self) -> None:
        """Initialize a AlignDatasetError."""
        super().__init__(
            "Cannot link datasets, aligning is ambiguous. \n\n"
            "Try to lower link tolerance or change the alignment method."
        )


class OptimizationDataProvider:
    @abc.abstractproperty
    def data_slices(self) -> list[ArrayLike]:
        pass

    @abc.abstractproperty
    def global_axis(self) -> ArrayLike:
        pass


class OptimizationData(OptimizationDataProvider):
    """A class to provide prepared data for optimization."""

    def __init__(self, model: DataModel) -> None:
        """Initialize a data provider for an experiment.

        Parameters
        ----------
        scheme : Scheme
            The optimization scheme.
        dataset_group : DatasetGroup
            The dataset group.
        """
        data = model.data
        assert isinstance(data, xr.Dataset)

        self._model = model
        self._model_dimension = get_data_model_dimension(model)
        self._model_axis = data.coords[self._model_dimension].data
        self._global_dimension = self.infer_global_dimension(self._model_dimension, data.data.dims)
        self._global_axis = data.coords[self._global_dimension].data

        self._data: ArrayLike = self.get_from_dataset(data, "data")  # type:ignore[assignment]
        self._flat_data = None

        self._weight = self.get_from_dataset(data, "weight")
        self._flat_weight = None
        if self._weight is None:
            self._weight = self.get_model_weight(model)
        if self._weight is not None:
            self._data *= self._weight

        if self.is_global:
            self._flat_data = self._data.T.flatten()
            if self._weight is not None:
                self._flat_weight = self._weight.T.flatten()
            self._data_slices = []
        else:
            self._data_slices = [self._data[:, i] for i in range(self.global_axis.size)]

    @property
    def data(self) -> ArrayLike:
        return self._data

    @property
    def flat_data(self) -> ArrayLike | None:
        return self._flat_data

    @property
    def data_slices(self) -> list[ArrayLike]:
        return self._data_slices

    @property
    def is_global(self) -> bool:
        return is_data_model_global(self._model)

    @property
    def global_axis(self) -> ArrayLike:
        return self._global_axis

    @property
    def global_dimension(self) -> str:
        return self._global_dimension

    @property
    def model(self) -> DataModel:
        return self._model

    @property
    def model_axis(self) -> ArrayLike:
        return self._model_axis

    @property
    def model_dimension(self) -> str:
        return self._model_dimension

    @property
    def weight(self) -> ArrayLike | None:
        return self._weight

    @property
    def flat_weight(self) -> ArrayLike | None:
        return self._flat_weight

    def get_model_weight(self, model: DataModel) -> ArrayLike | None:
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
        if not model.weights:
            return None
        weight = xr.DataArray(
            np.ones((self._model_axis.size, self._global_axis.size)),
            coords=(
                (self._model_dimension, self._model_axis),
                (self._global_dimension, self._global_axis),
            ),
        )

        for model_weight in model.weights:
            idx = {}
            if model_weight.global_interval is not None:
                idx[self._global_dimension] = self.get_axis_slice_from_interval(
                    model_weight.global_interval, self._global_axis
                )
            if model_weight.model_interval is not None:
                idx[self._model_dimension] = self.get_axis_slice_from_interval(
                    model_weight.model_interval, self._model_axis
                )
            weight[idx] *= model_weight.value

        return weight.data

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

    def get_from_dataset(self, dataset: xr.Dataset, name: str) -> ArrayLike | None:
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
            if dataset[name].dims != (self.model_dimension, self.global_dimension):
                data = data.T
        return data

    def unweight_result_dataset(self, result_dataset: xr.Dataset) -> None:
        if self.weight is None:
            return

        if "weight" not in result_dataset:
            result_dataset["weight"] = xr.DataArray(self.weight, coords=result_dataset.data.coords)
        result_dataset["weighted_residual"] = result_dataset["residual"]
        result_dataset["residual"] = result_dataset["residual"] / self.weight
        result_dataset.attrs["weighted_root_mean_square_error"] = result_dataset.attrs[
            "root_mean_square_error"
        ]
        result_dataset.attrs["root_mean_square_error"] = np.sqrt(
            (result_dataset.residual**2).sum() / sum(result_dataset.residual.shape)
        ).to_numpy()


class LinkedOptimizationData(OptimizationDataProvider):
    def __init__(
        self,
        datasets: dict[str, OptimizationData],
        tolerance: float,
        method: Literal["nearest", "backward", "forward"],
        scales: dict[str, Parameter],
    ) -> None:
        self._datasets = datasets
        self._scales = {label: scales.get(label, 1.0) for label in self._datasets}
        aligned_global_axes = self.align_global_axes(tolerance, method)
        self._global_axis, self._data = self.align_data(aligned_global_axes)
        self._data_indices = self.align_dataset_indices(aligned_global_axes)
        self._group_labels, self._group_definitions, self._group_sizes = self.align_groups(
            aligned_global_axes, datasets
        )

    @classmethod
    def from_experiment_model(cls, model: ExperimentModel) -> LinkedOptimizationData:
        return cls(
            {k: OptimizationData(d) for k, d in model.datasets.items()},
            model.clp_link_tolerance,
            model.clp_link_method,
            model.scale,  # type:ignore[arg-type]
        )

    @property
    def global_axis(self) -> ArrayLike:
        return self._global_axis

    @property
    def group_definitions(self) -> dict[str, list[str]]:
        return self._group_definitions

    @property
    def group_sizes(self) -> dict[str, list[int]]:
        return self._group_sizes

    @property
    def group_labels(self) -> ArrayLike:
        return self._group_labels

    @property
    def data_slices(self) -> list[ArrayLike]:
        return self._data

    @property
    def data_indices(self) -> list[ArrayLike]:
        return self._data_indices

    @property
    def data(self) -> dict[str, OptimizationData]:
        return self._datasets

    @property
    def scales(self) -> dict[str, Parameter | float]:
        return self._scales

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

    def align_global_axes(
        self,
        tolerance: float,
        method: Literal["nearest", "backward", "forward"],
    ) -> dict[str, ArrayLike]:
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
        for label, data in self._datasets.items():
            aligned_global_axis = data.global_axis
            if aligned_axis_values is None:
                aligned_axis_values = aligned_global_axis
            else:
                aligned_global_axis = [  # type:ignore[assignment]
                    self.align_index(index, aligned_axis_values, tolerance, method)
                    for index in aligned_global_axis
                ]
                if len(np.unique(aligned_global_axis)) != len(aligned_global_axis):
                    raise AlignDatasetError
                aligned_axis_values = np.unique(
                    np.concatenate([aligned_axis_values, aligned_global_axis])
                )
            aligned_global_axes[label] = aligned_global_axis
        return aligned_global_axes

    def align_data(
        self,
        aligned_global_axes: dict[str, ArrayLike],
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
                    self._datasets[label].data,
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
            for i in range(self._global_axis.size)
        ]

    @staticmethod
    def align_groups(
        aligned_global_axes: dict[str, ArrayLike],
        datasets: dict[str, OptimizationData],
    ) -> tuple[ArrayLike, dict[str, list[str]], dict[str, list[int]]]:
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
            tuple(
                "".join(sub_arr.to_numpy().flatten())
                for _, sub_arr in aligned_groups.groupby("global", squeeze=False)
            )
        )

        group_definitions: dict[str, list[str]] = {}
        group_sizes: dict[str, list[int]] = {}
        for i, group_label in enumerate(aligned_group_labels):
            if group_label not in group_definitions:
                group_definitions[group_label] = list(
                    filter(
                        lambda label: label != "",
                        aligned_groups.isel({"global": i}).data,
                    )
                )
                group_sizes[group_label] = [
                    datasets[label].model_axis.size for label in group_definitions[group_label]
                ]
        return aligned_group_labels, group_definitions, group_sizes
