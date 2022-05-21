from numbers import Number

import numpy as np

from glotaran.model import DatasetGroup
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
        self._residual_function = (
            residual_variable_projection
            if group.residual_function == "variable_projection"
            else residual_nnls
        )
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
            relation = relation.fill(model, parameters)
            if (
                relation.target in clp_labels
                and relation.applies(index)
                and relation.source in clp_labels
            ):
                source_idx = clp_labels.index(relation.source)
                target_idx = clp_labels.index(relation.target)
                clps[target_idx] = relation.parameter * clps[source_idx]
        return clps

    def calculate_clp_penalties(
        self,
        clp_labels: list[list[str]],
        clps: list[np.ndarray],
        global_axis: np.ndarray,
    ) -> np.ndarray:

        # TODO: make a decision on how to handle clp_penalties per dataset
        # 1. sum up contributions per dataset on each dataset_axis (v0.4.1)
        # 2. sum up contributions on the global_axis (future?)

        model = self.group.model
        parameters = self.group.parameters
        penalties = []
        for penalty in model.clp_area_penalties:
            penalty = penalty.fill(model, parameters)
            source_area = np.array([])
            target_area = np.array([])
            for dataset_model in self.group.dataset_models.values():
                dataset_axis = dataset_model.get_global_axis()

                source_area = np.concatenate(
                    [
                        source_area,
                        _get_area(
                            penalty.source,
                            clp_labels,
                            clps,
                            penalty.source_intervals,
                            global_axis,
                            dataset_axis,
                        ),
                    ]
                )

                target_area = np.concatenate(
                    [
                        target_area,
                        _get_area(
                            penalty.target,
                            clp_labels,
                            clps,
                            penalty.target_intervals,
                            global_axis,
                            dataset_axis,
                        ),
                    ]
                )
            area_penalty = np.abs(np.sum(source_area) - penalty.parameter * np.sum(target_area))

            penalties.append(area_penalty * penalty.weight)

        return [np.asarray(penalties)]

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
        self._clps = {label: [] for label in self.group.dataset_models}
        self._residuals = {label: [] for label in self.group.dataset_models}
        self._clp_penalty = []

    def estimate(self):
        self._clp_penalty.clear()
        for label, dataset_model in self.group.dataset_models.items():
            self._clps[label].clear()
            self._residuals[label].clear()

            global_axis = self._data_provider.get_global_axis(label)
            data = self._data_provider.get_data(label)
            clp_labels = []

            for index, global_index in enumerate(global_axis):
                matrix_container = self._matrix_provider.get_prepared_matrix_container(
                    label, index
                )
                reduced_clps, residual = self.calculate_residual(
                    matrix_container.matrix, data[:, index]
                )
                clp_labels.append(
                    self._matrix_provider.get_matrix_container(label, index).clp_labels
                )
                clp = self.retrieve_clps(
                    clp_labels[index], matrix_container.clp_labels, reduced_clps, global_index
                )

                self._clps[label].append(clp)
                self._residuals[label].append(residual)

            self._clp_penalty += self.calculate_clp_penalties(
                clp_labels, self._clps[label], global_axis
            )

    def get_full_penalty(self) -> np.typing.ArrayLike:
        full_residual = np.concatenate([np.concatenate(r) for r in self._residuals.values()])
        return np.concatenate([full_residual, np.concatenate(self._clp_penalty)])

    def get_result(
        self,
    ) -> tuple[dict[str, list[np.typing.ArrayLike]], dict[str, list[np.typing.ArrayLike]],]:
        return self._clps, self._residuals


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
        self._clps = [None] * self._data_provider.aligned_global_axis.size
        self._residuals = [None] * self._data_provider.aligned_global_axis.size
        self._clp_penalty = []

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
        return np.concatenate((np.concatenate(self._residuals), np.concatenate(self._clp_penalty)))

    def get_result(
        self,
    ) -> tuple[dict[str, list[np.typing.ArrayLike]], dict[str, list[np.typing.ArrayLike]],]:
        clps = {label: [] for label in self.group.dataset_models}
        residuals = {label: [] for label in self.group.dataset_models}
        for dataset_label in self.group.dataset_models:
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

                clps[dataset_label].append(
                    [
                        self._clps[index][
                            self._matrix_provider.aligned_full_clp_labels[index].index(label)
                        ]
                        for label in clp_labels
                    ]
                )

                start = sum(
                    self._data_provider.get_model_axis(label).size
                    for label in group_datasets[:dataset_index]
                )
                end = start + self._data_provider.get_model_axis(dataset_label).size
                residuals[dataset_label].append(self._residuals[index][start:end])
        return clps, residuals


def _get_area(
    clp_label: str,
    clp_labels: list[list[str]],
    clps: list[np.ndarray],
    intervals: list[tuple[float, float]],
    global_axis: np.ndarray,
    dataset_axis: np.ndarray,
) -> np.ndarray:
    area = []

    for interval in intervals:
        if interval[0] > global_axis[-1]:
            continue
        bounded_interval = (
            max(interval[0], np.min(dataset_axis)),
            min(interval[1], np.max(dataset_axis)),
        )
        start_idx = (
            np.abs(global_axis - bounded_interval[0]).argmin()
            if not np.isinf(bounded_interval[0])
            else 0
        )
        end_idx = (
            np.abs(global_axis - bounded_interval[1]).argmin()
            if not np.isinf(bounded_interval[1])
            else global_axis.size - 1
        )
        for i in range(start_idx, end_idx + 1):
            index_clp_labels = clp_labels[i] if isinstance(clp_labels[0], list) else clp_labels
            if clp_label in index_clp_labels:
                area.append(clps[i][index_clp_labels.index(clp_label)])

    return np.asarray(area)  # TODO: normalize for distance on global axis
