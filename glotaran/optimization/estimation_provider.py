"""Module containing the estimation provider classes."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model import DatasetGroup
from glotaran.model import DatasetModel
from glotaran.model import EqualAreaPenalty
from glotaran.model.dataset_model import has_dataset_model_global_model
from glotaran.model.item import fill_item
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderUnlinked
from glotaran.optimization.nnls import residual_nnls
from glotaran.optimization.variable_projection import residual_variable_projection

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike
SUPPORTED_RESIUDAL_FUNCTIONS = {
    "variable_projection": residual_variable_projection,
    "non_negative_least_squares": residual_nnls,
}


class UnsupportedResidualFunctionError(ValueError):
    """Inidcates that the residual function is unsupported."""

    def __init__(self, residual_function: str):
        """Initialize an UnsupportedMethodError.

        Parameters
        ----------
        residual_function : str
            The unsupported residual_function.
        """
        super().__init__(
            f"Unknown residual function '{residual_function}', "
            f"supported functions are: {list(SUPPORTED_RESIUDAL_FUNCTIONS.keys())}."
        )


class EstimationProvider:
    """A class to provide estimation for optimization."""

    def __init__(self, dataset_group: DatasetGroup):
        """Initialize an estimation provider for a dataset group.

        Parameters
        ----------
        dataset_group : DatasetGroup
            The dataset group.

        Raises
        ------
        UnsupportedResidualFunctionError
            Raised when residual function of the group dataset group is unsupported.
        """
        self._group = dataset_group
        self._clp_penalty: list[float] = []
        try:
            self._residual_function = SUPPORTED_RESIUDAL_FUNCTIONS[dataset_group.residual_function]
        except KeyError as e:
            raise UnsupportedResidualFunctionError(dataset_group.residual_function) from e

    @property
    def group(self) -> DatasetGroup:
        """Get the dataset group.

        Returns
        -------
        DatasetGroup
            The dataset group.
        """
        return self._group

    def calculate_residual(
        self, matrix: ArrayLike, data: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """Calculate the clps and the residual for a matrix and data.

        Parameters
        ----------
        matrix : ArrayLike
            The matrix.
        data : ArrayLike
            The data.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            The estimated clp and residual.
        """
        return self._residual_function(matrix, data)

    def retrieve_clps(
        self,
        clp_labels: list[str],
        reduced_clp_labels: list[str],
        reduced_clps: ArrayLike,
        index: int,
    ) -> ArrayLike:
        """Retrieve clp from reduced clp.

        Parameters
        ----------
        clp_labels : list[str]
            The original clp labels.
        reduced_clp_labels : list[str]
            The reduced clp labels.
        reduced_clps : ArrayLike
            The reduced clps.
        index : int
            The index on the global axis.

        Returns
        -------
        ArrayLike
            The retrieved clps.
        """
        model = self.group.model
        parameters = self.group.parameters
        if len(model.clp_relations) == 0 and len(model.clp_constraints) == 0:
            return reduced_clps

        clps = np.zeros(len(clp_labels))

        for i, label in enumerate(reduced_clp_labels):
            idx = clp_labels.index(label)
            clps[idx] = reduced_clps[i]

        for relation in model.clp_relations:
            relation = fill_item(relation, model, parameters)  # type:ignore[arg-type]
            if (
                relation.target in clp_labels
                and relation.applies(index)
                and relation.source in clp_labels
            ):
                source_idx = clp_labels.index(relation.source)
                target_idx = clp_labels.index(relation.target)
                clps[target_idx] = relation.parameter * clps[source_idx]
        return clps

    def get_additional_penalties(self) -> list[float]:
        """Get the additional penalty.

        Returns
        -------
        list[float]
            The additional penalty.
        """
        return self._clp_penalty

    def calculate_clp_penalties(
        self,
        clp_labels: list[list[str]],
        clps: list[np.ndarray],
        global_axis: np.ndarray,
    ) -> list[float]:
        """Calculate the clp penalty.

        Parameters
        ----------
        clp_labels : list[list[str]]
            The clp labels.
        clps : list[ArrayLike]
            The clps.
        global_axis : ArrayLike
            The global axis.

        Returns
        -------
        list[float]
            The clp penalty.
        """
        model = self.group.model
        parameters = self.group.parameters
        penalties = []
        for penalty in model.clp_penalties:
            if not isinstance(penalty, EqualAreaPenalty):
                continue
            penalty = fill_item(penalty, model, parameters)  # type:ignore[arg-type]

            source_area = _get_area(
                penalty.source,
                clp_labels,
                clps,
                penalty.source_intervals,
                global_axis,
            )
            target_area = _get_area(
                penalty.target,
                clp_labels,
                clps,
                penalty.target_intervals,
                global_axis,
            )

            if len(target_area) == 0 and len(source_area) == 0:
                continue
            elif len(target_area) == 0:
                warnings.warn(
                    "Ignoring equal area penalty, target clp " f"{penalty.target} not present."
                )
                continue
            elif len(source_area) == 0:
                warnings.warn(
                    "Ignoring equal area penalty, source clp " f"{penalty.source} not present."
                )
                continue

            area_penalty = np.abs(np.sum(source_area) - penalty.parameter * np.sum(target_area))

            penalties.append(area_penalty * penalty.weight)

        return penalties

    def estimate(self):
        """Calculate the estimation.

        .. # noqa: DAR401
        """
        raise NotImplementedError

    def get_full_penalty(self) -> ArrayLike:
        """Get the full penalty.

        Returns
        -------
        ArrayLike
            The clp penalty.

        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError

    def get_result(
        self,
    ) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray],]:
        """Get the results of the estimation.

        Returns
        -------
        tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
            A tuple of the estimated clps and residuals.

        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError


class EstimationProviderUnlinked(EstimationProvider):
    """A class to provide estimation for optimization of an unlinked dataset group."""

    def __init__(
        self,
        dataset_group: DatasetGroup,
        data_provider: DataProvider,
        matrix_provider: MatrixProviderUnlinked,
    ):
        """Initialize an estimation provider for an unlinked dataset group.

        Parameters
        ----------
        dataset_group : DatasetGroup
            The dataset group.
        data_provider : DataProvider
            The data provider.
        matrix_provider : MatrixProviderUnlinked
            The matrix provider.
        """
        super().__init__(dataset_group)
        self._data_provider = data_provider
        self._matrix_provider = matrix_provider
        self._clps: dict[str, list[ArrayLike] | ArrayLike] = {
            label: [] for label in self.group.dataset_models
        }
        self._residuals: dict[str, list[ArrayLike] | ArrayLike] = {
            label: [] for label in self.group.dataset_models
        }

    def estimate(self):
        """Calculate the estimation."""
        self._clp_penalty.clear()

        for dataset_model in self.group.dataset_models.values():
            if has_dataset_model_global_model(dataset_model):
                self.calculate_full_model_estimation(dataset_model)
            else:
                self.calculate_estimation(dataset_model)

    def get_full_penalty(self) -> ArrayLike:
        """Get the full penalty.

        Returns
        -------
        ArrayLike
            The clp penalty.
        """
        full_penalty = np.concatenate(
            [
                self._residuals[label]
                if has_dataset_model_global_model(dataset_model)
                else np.concatenate(self._residuals[label])
                for label, dataset_model in self.group.dataset_models.items()
            ]
        )
        if len(self._clp_penalty) != 0:
            full_penalty = np.concatenate([full_penalty, self._clp_penalty])
        return full_penalty

    def get_result(
        self,
    ) -> tuple[dict[str, list[xr.DataArray]], dict[str, list[xr.DataArray]],]:
        """Get the results of the estimation.

        Returns
        -------
        tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
            A tuple of the estimated clps and residuals.
        """
        clps, residuals = {}, {}
        for label, dataset_model in self.group.dataset_models.items():
            model_dimension = self._data_provider.get_model_dimension(label)
            model_axis = self._data_provider.get_model_axis(label)
            global_dimension = self._data_provider.get_global_dimension(label)
            global_axis = self._data_provider.get_global_axis(label)

            if has_dataset_model_global_model(dataset_model):
                residuals[label] = xr.DataArray(
                    np.array(self._residuals[label]).T.reshape(model_axis.size, global_axis.size),
                    coords={global_dimension: global_axis, model_dimension: model_axis},
                    dims=[model_dimension, global_dimension],
                )
                clp_labels = self._matrix_provider.get_matrix_container(label).clp_labels
                global_clp_labels = self._matrix_provider.get_global_matrix_container(
                    label
                ).clp_labels
                clps[label] = xr.DataArray(
                    np.array(self._clps[label]).reshape((len(global_clp_labels), len(clp_labels))),
                    coords={"global_clp_label": global_clp_labels, "clp_label": clp_labels},
                    dims=["global_clp_label", "clp_label"],
                )

            else:
                residuals[label] = xr.DataArray(
                    np.array(self._residuals[label]).T,
                    coords={global_dimension: global_axis, model_dimension: model_axis},
                    dims=[model_dimension, global_dimension],
                )
                clps[label] = xr.DataArray(
                    self._clps[label],
                    coords=(
                        (global_dimension, global_axis),
                        (
                            "clp_label",
                            self._matrix_provider.get_matrix_container(label).clp_labels,
                        ),
                    ),
                )
        return clps, residuals

    def calculate_full_model_estimation(self, dataset_model: DatasetModel):
        """Calculate the estimation for a dataset with a full model.

        Parameters
        ----------
        dataset_model : DatasetModel
            The dataset model.
        """
        label = dataset_model.label
        full_matrix = self._matrix_provider.get_full_matrix(label)
        data = self._data_provider.get_flattened_data(label)
        self._clps[label], self._residuals[label] = self.calculate_residual(full_matrix, data)

    def calculate_estimation(self, dataset_model: DatasetModel):
        """Calculate the estimation for a dataset.

        Parameters
        ----------
        dataset_model : DatasetModel
            The dataset model.
        """
        label = dataset_model.label
        self._clps[label].clear()  # type:ignore[union-attr]
        self._residuals[label].clear()  # type:ignore[union-attr]

        global_axis = self._data_provider.get_global_axis(label)
        data = self._data_provider.get_data(label)
        clp_labels = []

        for index, global_index_value in enumerate(global_axis):
            matrix_container = self._matrix_provider.get_prepared_matrix_container(label, index)
            reduced_clps, residual = self.calculate_residual(
                matrix_container.matrix, data[:, index]
            )
            clp_labels.append(self._matrix_provider.get_matrix_container(label).clp_labels)
            clp = self.retrieve_clps(
                clp_labels[index], matrix_container.clp_labels, reduced_clps, global_index_value
            )

            self._clps[label].append(clp)  # type:ignore[union-attr]
            self._residuals[label].append(residual)  # type:ignore[union-attr]

        self._clp_penalty += self.calculate_clp_penalties(
            clp_labels, self._clps[label], global_axis  # type:ignore[arg-type]
        )


class EstimationProviderLinked(EstimationProvider):
    """A class to provide estimation for optimization of a linked dataset group."""

    def __init__(
        self,
        dataset_group: DatasetGroup,
        data_provider: DataProviderLinked,
        matrix_provider: MatrixProviderLinked,
    ):
        """Initialize an estimation provider for a linked dataset group.

        Parameters
        ----------
        dataset_group : DatasetGroup
            The dataset group.
        data_provider : DataProviderLinked
            The data provider.
        matrix_provider : MatrixProviderLinked
            The matrix provider.
        """
        super().__init__(dataset_group)
        self._data_provider = data_provider
        self._matrix_provider = matrix_provider
        self._clps: list[ArrayLike] = [
            None  # type:ignore[list-item]
        ] * self._data_provider.aligned_global_axis.size
        self._residuals: list[ArrayLike] = [
            None  # type:ignore[list-item]
        ] * self._data_provider.aligned_global_axis.size

    def estimate(self):
        """Calculate the estimation."""
        for index, global_index_value in enumerate(self._data_provider.aligned_global_axis):
            matrix_container = self._matrix_provider.get_aligned_matrix_container(index)
            data = self._data_provider.get_aligned_data(index)
            reduced_clps, residual = self.calculate_residual(matrix_container.matrix, data)
            self._clps[index] = self.retrieve_clps(
                self._matrix_provider.aligned_full_clp_labels[index],
                matrix_container.clp_labels,
                reduced_clps,
                global_index_value,
            )
            self._residuals[index] = residual

        self._clp_penalty = self.calculate_clp_penalties(
            self._matrix_provider.aligned_full_clp_labels,
            self._clps,
            self._data_provider.aligned_global_axis,
        )

    def get_full_penalty(self) -> ArrayLike:
        """Get the full penalty.

        Returns
        -------
        ArrayLike
            The clp penalty.
        """
        return np.concatenate((np.concatenate(self._residuals), self._clp_penalty))

    def get_result(
        self,
    ) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray],]:
        """Get the results of the estimation.

        Returns
        -------
        tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
            A tuple of the estimated clps and residuals.
        """
        clps: dict[str, xr.DataArray] = {}
        residuals: dict[str, xr.DataArray] = {}
        for dataset_label in self.group.dataset_models:
            dataset_clps, dataset_residual = [], []
            for index in range(self._data_provider.aligned_global_axis.size):
                group_label = self._data_provider.get_aligned_group_label(index)
                if dataset_label not in group_label:
                    continue

                group_datasets = self._data_provider.group_definitions[group_label]
                dataset_index = group_datasets.index(dataset_label)

                clp_labels = self._matrix_provider.get_matrix_container(dataset_label).clp_labels

                dataset_clps.append(
                    xr.DataArray(
                        [
                            self._clps[index][
                                self._matrix_provider.aligned_full_clp_labels[index].index(label)
                            ]
                            for label in clp_labels
                        ],
                        coords={"clp_label": clp_labels},
                    )
                )

                start = sum(
                    self._data_provider.get_model_axis(label).size
                    for label in group_datasets[:dataset_index]
                )
                end = start + self._data_provider.get_model_axis(dataset_label).size
                dataset_residual.append(self._residuals[index][start:end])

            model_dimension = self._data_provider.get_model_dimension(dataset_label)
            model_axis = self._data_provider.get_model_axis(dataset_label)
            global_dimension = self._data_provider.get_global_dimension(dataset_label)
            global_axis = self._data_provider.get_global_axis(dataset_label)
            clps[dataset_label] = xr.concat(
                dataset_clps,
                dim=global_dimension,
            )
            clps[dataset_label].coords[global_dimension] = global_axis
            residuals[dataset_label] = xr.DataArray(
                np.array(dataset_residual).T,
                coords={global_dimension: global_axis, model_dimension: model_axis},
                dims=[model_dimension, global_dimension],
            )
        return clps, residuals


def _get_area(
    clp_label: str,
    clp_labels: list[str] | list[list[str]],
    clps: list[ArrayLike],
    intervals: list[tuple[float, float]],
    global_axis: ArrayLike,
) -> ArrayLike:
    """Get get slice of a clp on intervals on the global axis.

    Parameters
    ----------
    clp_label : str
        The label of the clp.
    clp_labels: list[str] | list[list[str]]
        The clp labels.
    clps : list[ArrayLike]
        The clps.
    intervals: list[tuple[float, float]]
        The intervals on the global axis.
    global_axis : ArrayLike
        The global axis.

    Returns
    -------
    ArrayLike:
        The concatenated slices.
    """
    area = []

    for interval in intervals:
        if interval[0] > global_axis[-1]:
            continue
        bounded_interval = (
            max(interval[0], np.min(global_axis)),
            min(interval[1], np.max(global_axis)),
        )

        interval_slice = DataProvider.get_axis_slice_from_interval(bounded_interval, global_axis)
        start_idx, end_idx = interval_slice.start, interval_slice.stop

        for i in range(start_idx, end_idx):
            index_clp_labels: list[str] = (
                clp_labels[i]
                if isinstance(clp_labels[0], list)
                else clp_labels  # type:ignore[assignment]
            )
            if clp_label in index_clp_labels:
                area.append(clps[i][index_clp_labels.index(clp_label)])

    return np.asarray(area)  # TODO: normalize for distance on global axis
