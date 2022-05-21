from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from numbers import Number

import numba as nb
import numpy as np

from glotaran.model import DatasetGroup
from glotaran.model import DatasetModel
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked


@dataclass
class MatrixContainer:
    clp_labels: list[str]
    matrix: np.ndarray

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _create_weighted_matrix(
        matrix: np.typing.ArrayLike, weight: np.typing.ArrayLike
    ) -> np.typing.ArrayLike:
        matrix = matrix.copy()
        for i in nb.prange(matrix.shape[1]):
            matrix[:, i] *= weight
        return matrix

    def create_weighted_matrix(self, weight: np.typing.ArrayLike) -> MatrixContainer:
        return replace(self, matrix=self._create_weighted_matrix(self.matrix, weight))

    def create_scaled_matrix(self, scale: float) -> MatrixContainer:
        return replace(self, matrix=self.matrix * scale)


class MatrixProvider:
    def __init__(self, group: DatasetGroup):
        self._group = group
        self._matrix_containers: dict[str, MatrixContainer | list[MatrixContainer]] = {}

    @property
    def group(self) -> DatasetGroup:
        return self._group

    def get_matrix_container(self, dataset_label: str, index: int) -> MatrixContainer:
        matrix_container = self._matrix_containers[dataset_label]
        if self.group.dataset_models[dataset_label].is_index_dependent():
            matrix_container = matrix_container[index]
        return matrix_container

    def create_dataset_matrices(self):
        for label, dataset_model in self.group.dataset_models.items():
            model_axis = self._data_provider.get_model_axis(label)
            global_axis = self._data_provider.get_global_axis(label)

            if dataset_model.is_index_dependent():
                self._matrix_containers[label] = [
                    self.calculate_dataset_matrix(
                        dataset_model, global_index, global_axis, model_axis
                    )
                    for global_index in range(self._data_provider.get_global_axis(label).size)
                ]
            else:
                self._matrix_containers[label] = self.calculate_dataset_matrix(
                    dataset_model, None, global_axis, model_axis
                )

    @staticmethod
    def calculate_dataset_matrix(
        dataset_model: DatasetModel,
        global_index: int | None,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
        as_global_model: bool = False,
    ) -> MatrixContainer:

        clp_labels = None
        matrix = None

        megacomplex_iterator = dataset_model.iterate_megacomplexes

        if as_global_model:
            megacomplex_iterator = dataset_model.iterate_global_megacomplexes
            model_axis, global_axis = global_axis, model_axis

        for scale, megacomplex in megacomplex_iterator():
            this_clp_labels, this_matrix = megacomplex.calculate_matrix(
                dataset_model, global_index, global_axis, model_axis
            )

            if scale is not None:
                this_matrix *= scale

            if matrix is None:
                clp_labels = this_clp_labels
                matrix = this_matrix
            else:
                clp_labels, matrix = MatrixProvider.combine_megacomplex_matrices(
                    matrix, this_matrix, clp_labels, this_clp_labels
                )
        return MatrixContainer(clp_labels, matrix)

    @staticmethod
    def combine_megacomplex_matrices(
        matrix: np.typing.ArrayLike,
        this_matrix: np.typing.ArrayLike,
        clp_labels: list[str],
        this_clp_labels: list[str],
    ):
        tmp_clp_labels = clp_labels + [c for c in this_clp_labels if c not in clp_labels]
        tmp_matrix = np.zeros((matrix.shape[0], len(tmp_clp_labels)), dtype=np.float64)
        for idx, label in enumerate(tmp_clp_labels):
            if label in clp_labels:
                tmp_matrix[:, idx] += matrix[:, clp_labels.index(label)]
            if label in this_clp_labels:
                tmp_matrix[:, idx] += this_matrix[:, this_clp_labels.index(label)]
        return tmp_clp_labels, tmp_matrix

    def reduce_matrix(
        self,
        matrix: MatrixContainer,
        index: Number | None,
    ) -> MatrixContainer:
        matrix = self.apply_relations(matrix, index)
        matrix = self.apply_constraints(matrix, index)
        return matrix

    def apply_constraints(
        self,
        matrix: MatrixContainer,
        index: Number | None,
    ) -> MatrixContainer:

        model = self.group.model
        if len(model.clp_constraints) == 0:
            return matrix

        clp_labels = matrix.clp_labels
        removed_clp_labels = [
            c.target for c in model.clp_constraints if c.target in clp_labels and c.applies(index)
        ]
        reduced_clp_labels = [c for c in clp_labels if c not in removed_clp_labels]
        mask = [label in reduced_clp_labels for label in clp_labels]
        reduced_matrix = matrix.matrix[:, mask]
        return MatrixContainer(reduced_clp_labels, reduced_matrix)

    def apply_relations(
        self,
        matrix: MatrixContainer,
        index: Number | None,
    ) -> MatrixContainer:
        model = self.group.model
        parameters = self.group.parameters

        if len(model.clp_relations) == 0:
            return matrix

        clp_labels = matrix.clp_labels
        relation_matrix = np.diagflat([1.0 for _ in clp_labels])

        idx_to_delete = []
        for relation in model.clp_relations:
            if relation.target in clp_labels and relation.applies(index):

                if relation.source not in clp_labels:
                    continue

                relation = relation.fill(model, parameters)
                source_idx = clp_labels.index(relation.source)
                target_idx = clp_labels.index(relation.target)
                relation_matrix[target_idx, source_idx] = relation.parameter
                idx_to_delete.append(target_idx)

        reduced_clp_labels = [
            label for i, label in enumerate(clp_labels) if i not in idx_to_delete
        ]
        relation_matrix = np.delete(relation_matrix, idx_to_delete, axis=1)
        reduced_matrix = matrix.matrix @ relation_matrix
        return MatrixContainer(reduced_clp_labels, reduced_matrix)

    def calculate(self):
        raise NotImplementedError

    def get_result(self) -> tuple[dict[str, list[str]], dict[str, np.typing.ArrayLike]]:
        clp_labels, matrices = {}, {}
        for label, matrix_container in self._matrix_containers.items():
            if self.group.dataset_models[label].is_index_dependent():
                clp_labels[label] = [m.clp_labels for m in matrix_container]
                matrices[label] = [m.matrix for m in matrix_container]
            else:
                clp_labels[label] = matrix_container.clp_labels
                matrices[label] = matrix_container.matrix

        return clp_labels, matrices


class MatrixProviderUnlinked(MatrixProvider):
    def __init__(self, group: DatasetGroup, data_provider: DataProvider):
        super().__init__(group)
        self._data_provider = data_provider
        self._prepared_matrix_container: dict[str, list[MatrixContainer]] = {}

    def get_prepared_matrix_container(self, dataset_label: str, index: int) -> MatrixContainer:
        return self._prepared_matrix_container[dataset_label][index]

    def calculate(self):
        self.create_dataset_matrices()

        self.create_prepared_matrices()

    def create_prepared_matrices(self):
        for label, dataset_model in self.group.dataset_models.items():
            scale = dataset_model.scale or 1
            weight = self._data_provider.get_weight(label)
            if dataset_model.is_index_dependent():
                self._prepared_matrix_container[label] = [
                    self.reduce_matrix(
                        self.get_matrix_container(label, i).create_scaled_matrix(scale),
                        global_index,
                    )
                    for i, global_index in enumerate(self._data_provider.get_global_axis(label))
                ]
            else:
                self._prepared_matrix_container[label] = [
                    self.reduce_matrix(
                        self.get_matrix_container(label, 0).create_scaled_matrix(scale), None
                    )
                ] * self._data_provider.get_global_axis(label).size
            if weight is not None:
                self._prepared_matrix_container[label] = [
                    matrix.create_weighted_matrix(weight[:, i])
                    for i, matrix in enumerate(self._prepared_matrix_container[label])
                ]


class MatrixProviderLinked(MatrixProvider):
    def __init__(self, group: DatasetGroup, data_provider: DataProviderLinked):
        super().__init__(group)
        self._data_provider = data_provider
        self._aligned_full_clp_labels = [None] * self._data_provider.aligned_global_axis.size
        self._aligned_matrices = [None] * self._data_provider.aligned_global_axis.size

    @property
    def aligned_full_clp_labels(self) -> list[list[str]]:
        return self._aligned_full_clp_labels

    def get_aligned_matrix_container(self, index: int) -> MatrixContainer:
        return self._aligned_matrices[index]

    def create_aligned_matrices(self):

        for i, global_index in enumerate(self._data_provider.aligned_global_axis):
            group_label = self._data_provider.get_aligned_group_label(i)
            group_matrix = self.align_matrices(
                [
                    self.get_matrix_container(label, index)
                    for label, index in zip(
                        self._data_provider.group_definitions[group_label],
                        self._data_provider.get_aligned_dataset_indices(i),
                    )
                ],
                [
                    self.group.dataset_models[label].scale
                    if self.group.dataset_models[label].scale is not None
                    else 1
                    for label in self._data_provider.group_definitions[group_label]
                ],
            )

            self._aligned_full_clp_labels[i] = group_matrix.clp_labels
            group_matrix = self.reduce_matrix(group_matrix, global_index)
            weight = self._data_provider.get_aligned_weight(i)
            if weight is not None:
                group_matrix = group_matrix.create_weighted_matrix(weight)

            self._aligned_matrices[i] = group_matrix

    def calculate(self):

        self.create_dataset_matrices()

        self.create_aligned_matrices()

    @staticmethod
    def align_matrices(matrices: list[MatrixContainer], scales: list[Number]) -> MatrixContainer:
        if len(matrices) == 1:
            return matrices[0]
        masks = []
        full_clp_labels = None
        sizes = []
        dim1 = 0
        for matrix in matrices:
            clp_labels = matrix.clp_labels
            model_axis_size = matrix.matrix.shape[0]
            sizes.append(model_axis_size)
            dim1 += model_axis_size
            if full_clp_labels is None:
                full_clp_labels = clp_labels.copy()
                masks.append([i for i, _ in enumerate(clp_labels)])
            else:
                mask = []
                for c in clp_labels:
                    if c not in full_clp_labels:
                        full_clp_labels.append(c)
                    mask.append(full_clp_labels.index(c))
                masks.append(mask)
        dim2 = len(full_clp_labels)
        full_matrix = np.zeros((dim1, dim2), dtype=np.float64)
        start = 0
        for i, m in enumerate(matrices):
            end = start + sizes[i]
            full_matrix[start:end, masks[i]] = m.matrix * scales[i]
            start = end

        return MatrixContainer(full_clp_labels, full_matrix)
