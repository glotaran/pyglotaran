from __future__ import annotations

from numbers import Number

import numpy as np
import xarray as xr

from glotaran.model import DatasetGroup
from glotaran.model import DatasetModel
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderUnlinked
from glotaran.optimization.nnls import residual_nnls
from glotaran.optimization.variable_projection import residual_variable_projection

residual_functions = {
    "variable_projection": residual_variable_projection,
    "non_negative_least_squares": residual_nnls,
}


class EstimationProvider:
    def __init__(self, group: DatasetGroup):
        self._group = group
        self._clp_penalty: list[Number] = []
        try:
            self._residual_function = residual_functions[group.residual_function]
        except KeyError as e:
            raise ValueError(
                f"Unknown residual function '{group.residual_function}', "
                f"allowed functions are: {list(residual_functions.keys())}."
            ) from e

    @property
    def group(self) -> DatasetGroup:
        return self._group

    def calculate_residual(
        self, matrix: np.typing.ArrayLike, data: np.typing.ArrayLike
    ) -> tuple[np.typing.ArrayLike, np.typing.ArrayLike]:
        return self._residual_function(matrix, data)

    def retrieve_clps(
        self,
        clp_labels: list[str],
        reduced_clp_labels: list[str],
        reduced_clps: np.typing.ArrayLike,
        index: Number,
    ) -> np.typing.ArrayLike:
        model = self.group.model
        parameters = self.group.parameters
        if len(model.clp_relations) == 0 and len(model.clp_constraints) == 0:
            return reduced_clps

        clps = np.zeros(len(clp_labels))

        for i, label in enumerate(reduced_clp_labels):
            idx = clp_labels.index(label)
            clps[idx] = reduced_clps[i]

        for relation in model.clp_relations:
            relation = relation.fill(model, parameters)  # type:ignore[attr-defined]
            if (
                relation.target in clp_labels  # type:ignore[attr-defined]
                and relation.applies(index)  # type:ignore[attr-defined]
                and relation.source in clp_labels  # type:ignore[attr-defined]
            ):
                source_idx = clp_labels.index(relation.source)  # type:ignore[attr-defined]
                target_idx = clp_labels.index(relation.target)  # type:ignore[attr-defined]
                clps[target_idx] = (
                    relation.parameter * clps[source_idx]  # type:ignore[attr-defined]
                )
        return clps

    def get_additional_penalties(self) -> list[Number]:
        return self._clp_penalty

    def calculate_clp_penalties(
        self,
        clp_labels: list[list[str]],
        clps: list[np.ndarray],
        global_axis: np.ndarray,
    ) -> list[Number]:

        # TODO: make a decision on how to handle clp_penalties per dataset
        # 1. sum up contributions per dataset on each dataset_axis (v0.4.1)
        # 2. sum up contributions on the global_axis (future?)

        model = self.group.model
        parameters = self.group.parameters
        penalties = []
        for penalty in model.clp_area_penalties:
            penalty = penalty.fill(model, parameters)  # type:ignore[attr-defined]
            source_area = np.array([])
            target_area = np.array([])

            source_area = np.concatenate(
                [
                    source_area,
                    _get_area(
                        penalty.source,  # type:ignore[attr-defined]
                        clp_labels,
                        clps,
                        penalty.source_intervals,  # type:ignore[attr-defined]
                        global_axis,
                    ),
                ]
            )

            target_area = np.concatenate(
                [
                    target_area,
                    _get_area(
                        penalty.target,  # type:ignore[attr-defined]
                        clp_labels,
                        clps,
                        penalty.target_intervals,  # type:ignore[attr-defined]
                        global_axis,
                    ),
                ]
            )
            area_penalty = np.abs(
                np.sum(source_area)
                - penalty.parameter * np.sum(target_area)  # type:ignore[attr-defined]
            )

            penalties.append(area_penalty * penalty.weight)  # type:ignore[attr-defined]

        return penalties

    def estimate(self):
        raise NotImplementedError

    def get_full_penalty(self) -> np.typing.ArrayLike:
        raise NotImplementedError

    def get_result(
        self,
    ) -> tuple[dict[str, list[np.typing.ArrayLike]], dict[str, list[np.typing.ArrayLike]],]:
        raise NotImplementedError


class EstimationProviderUnlinked(EstimationProvider):
    def __init__(
        self,
        group: DatasetGroup,
        data_provider: DataProvider,
        matrix_provider: MatrixProviderUnlinked,
    ):
        super().__init__(group)
        self._data_provider = data_provider
        self._matrix_provider = matrix_provider
        self._clps: dict[str, list[np.typing.ArrayLike] | np.typing.ArrayLike] = {
            label: [] for label in self.group.dataset_models
        }
        self._residuals: dict[str, list[np.typing.ArrayLike] | np.typing.ArrayLike] = {
            label: [] for label in self.group.dataset_models
        }

    def estimate(self):
        self._clp_penalty.clear()

        for label, dataset_model in self.group.dataset_models.items():
            if dataset_model.has_global_model():
                self.calculate_full_model_estimation(label, dataset_model)
            else:
                self.calculate_estimation(label, dataset_model)

    def get_full_penalty(self) -> np.typing.ArrayLike:
        full_penalty = np.concatenate(
            [
                self._residuals[label]
                if dataset_model.has_global_model()
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
        clps, residuals = {}, {}
        for label, dataset_model in self.group.dataset_models.items():
            model_dimension = self._data_provider.get_model_dimension(label)
            model_axis = self._data_provider.get_model_axis(label)
            global_dimension = self._data_provider.get_global_dimension(label)
            global_axis = self._data_provider.get_global_axis(label)

            if dataset_model.has_global_model():
                residuals[label] = xr.DataArray(
                    np.array(self._residuals[label]).T.reshape(model_axis.size, global_axis.size),
                    coords={global_dimension: global_axis, model_dimension: model_axis},
                    dims=[model_dimension, global_dimension],
                )
                clp_labels = self._matrix_provider.get_matrix_container(label, 0).clp_labels
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
                if dataset_model.is_index_dependent():
                    clps[label] = xr.concat(
                        [
                            xr.DataArray(
                                self._clps[label][i],
                                coords={
                                    "clp_label": self._matrix_provider.get_matrix_container(
                                        label, i
                                    ).clp_labels
                                },
                            )
                            for i in range(len(self._clps[label]))
                        ],
                        dim=global_dimension,
                    )
                    clps[label].coords[global_dimension] = global_axis

                else:
                    clps[label] = xr.DataArray(
                        self._clps[label],
                        coords=(
                            (global_dimension, global_axis),
                            (
                                "clp_label",
                                self._matrix_provider.get_matrix_container(label, 0).clp_labels,
                            ),
                        ),
                    )
        return clps, residuals

    def calculate_full_model_estimation(self, label: str, dataset_model: DatasetModel):
        full_matrix = self._matrix_provider.get_full_matrix(label)
        data = self._data_provider.get_flattened_data(label)
        self._clps[label], self._residuals[label] = self.calculate_residual(full_matrix, data)

    def calculate_estimation(self, label: str, dataset_model: DatasetModel):
        self._clps[label].clear()
        self._residuals[label].clear()

        global_axis = self._data_provider.get_global_axis(label)
        data = self._data_provider.get_data(label)
        clp_labels = []

        for index, global_index in enumerate(global_axis):
            matrix_container = self._matrix_provider.get_prepared_matrix_container(label, index)
            reduced_clps, residual = self.calculate_residual(
                matrix_container.matrix, data[:, index]
            )
            clp_labels.append(self._matrix_provider.get_matrix_container(label, index).clp_labels)
            clp = self.retrieve_clps(
                clp_labels[index], matrix_container.clp_labels, reduced_clps, global_index
            )

            self._clps[label].append(clp)
            self._residuals[label].append(residual)

        self._clp_penalty += self.calculate_clp_penalties(
            clp_labels, self._clps[label], global_axis
        )


class EstimationProviderLinked(EstimationProvider):
    def __init__(
        self,
        group: DatasetGroup,
        data_provider: DataProviderLinked,
        matrix_provider: MatrixProviderLinked,
    ):
        super().__init__(group)
        self._data_provider = data_provider
        self._matrix_provider = matrix_provider
        self._clps: list[np.typing.ArrayLike] = [
            None
        ] * self._data_provider.aligned_global_axis.size
        self._residuals: list[np.typing.ArrayLike] = [
            None
        ] * self._data_provider.aligned_global_axis.size

    def estimate(self):
        for index, global_index in enumerate(self._data_provider.aligned_global_axis):
            matrix_container = self._matrix_provider.get_aligned_matrix_container(index)
            data = self._data_provider.get_aligned_data(index)
            reduced_clps, residual = self.calculate_residual(matrix_container.matrix, data)
            self._clps[index] = self.retrieve_clps(
                self._matrix_provider.aligned_full_clp_labels[index],
                matrix_container.clp_labels,
                reduced_clps,
                global_index,
            )
            self._residuals[index] = residual

        self._clp_penalty = self.calculate_clp_penalties(
            self._matrix_provider.aligned_full_clp_labels,
            self._clps,
            self._data_provider.aligned_global_axis,
        )

    def get_full_penalty(self) -> np.typing.ArrayLike:
        return np.concatenate((np.concatenate(self._residuals), self._clp_penalty))

    def get_result(
        self,
    ) -> tuple[dict[str, list[np.typing.ArrayLike]], dict[str, list[np.typing.ArrayLike]],]:
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
                global_index = self._data_provider.get_aligned_dataset_indices(index)[
                    dataset_index
                ]

                clp_labels = self._matrix_provider.get_matrix_container(
                    dataset_label, global_index
                ).clp_labels

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
    clp_labels: list[list[str]],
    clps: list[np.ndarray],
    intervals: list[tuple[float, float]],
    global_axis: np.typing.ArrayLike,
) -> np.typing.ArrayLike:
    area = []

    for interval in intervals:
        if interval[0] > global_axis[-1]:
            continue
        bounded_interval = (
            max(interval[0], np.min(global_axis)),
            min(interval[1], np.max(global_axis)),
        )
        start_idx = (
            0
            if np.isinf(bounded_interval[0])
            else np.abs(global_axis - bounded_interval[0]).argmin()
        )

        end_idx = (
            global_axis.size - 1
            if np.isinf(bounded_interval[1])
            else np.abs(global_axis - bounded_interval[1]).argmin()
        )

        for i in range(start_idx, end_idx + 1):
            index_clp_labels = clp_labels[i] if isinstance(clp_labels[0], list) else clp_labels
            if clp_label in index_clp_labels:
                area.append(clps[i][index_clp_labels.index(clp_label)])

    return np.asarray(area)  # TODO: normalize for distance on global axis
