from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from numbers import Number

import numba as nb
import numpy as np

from glotaran.model import DatasetGroup
from glotaran.model import DatasetModel
from glotaran.optimization.data_provider import DataProvider


@dataclass
class MatrixContainer:
    clp_labels: list[str]
    matrix: np.ndarray

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _create_weighted_matrix(matrix: np.typing.ArrayLike, weight: np.typing.ArrayLike):
        matrix = matrix.copy()
        for i in nb.prange(matrix.shape[1]):
            matrix[:, i] *= weight

    def create_weighted_matrix(self, weight: np.typing.ArrayLike) -> MatrixContainer:
        return replace(self, matrix=self._create_weighted_matrix(self.matrix, weight))


class MatrixProvider:
    def __init__(self, group: DatasetGroup):
        self._group = group

    @property
    def group(self) -> DatasetGroup:
        return self._group

    @staticmethod
    def calculate_dataset_matrix(
        dataset_model: DatasetModel,
        global_index: int,
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
        raise NotImplementedError


class MatrixProviderUnlinked(MatrixProvider):
    def __init__(self, group: DatasetGroup, data_provider: DataProvider):
        super().__init__(group)
        self._data_provider = data_provider
        self._matrices = {}
        self._global_matrices = {}
        self._reduced_matrices = {}

    def get_reduced_matrix(self, dataset_label: str, index: int) -> MatrixContainer:
        return self._reduced_matrices[dataset_label][index]

    def calculate(self):
        for label, dataset_model in self.group.dataset_models.items():
            model_axis = self._data_provider.get_model_axis(label)
            global_axis = self._data_provider.get_global_axis(label)
            if dataset_model.is_index_dependent():
                self._matrices[label] = [
                    self.calculate_dataset_matrix(
                        dataset_model, global_index, global_axis, model_axis
                    )
                    for global_index in range(self._data_provider.get_global_axis(label).size)
                ]
            else:
                self._matrices[label] = self.calculate_dataset_matrix(
                    dataset_model, None, global_axis, model_axis
                )

        if dataset_model.has_global_model():
            self._global_matrices[label] = self.calculate_dataset_matrix(
                dataset_model, None, global_axis, model_axis, as_global_model=True
            )
        else:
            self.create_reduced_matrices()

    def create_reduced_matrices(self):
        for label, dataset_model in self.group.dataset_models.items():
            weight = self._data_provider.get_weight(label)
            if dataset_model.is_index_dependent():
                self._reduced_matrices[label] = [
                    self.reduce_matrix(self._matrices[label][i], global_index)
                    for i, global_index in enumerate(self._data_provider.get_global_axis(label))
                ]
            else:
                self._reduced_matrices[label] = [
                    self.reduce_matrix(self._matrices[label], None)
                ] * self._data_provider.get_global_axis(label).size
            if weight is not None:
                self._reduced_matrices[label] = [
                    matrix.create_weighted_matrix(weight[:, i])
                    for i, matrix in enumerate(self._reduced_matrices[label])
                ]

    def get_result(self) -> tuple[dict[str, list[str]], dict[str, np.typing.ArrayLike]]:
        clp_labels, matrices = {}, {}
        for label, matrix_container in self._matrices.items():
            if self.group.dataset_models[label].is_index_dependent():
                clp_labels[label] = [m.clp_labels for m in matrix_container]
                matrices[label] = [m.matrix for m in matrix_container]
            else:
                clp_labels[label] = matrix_container.clp_labels
                matrices[label] = matrix_container.matrix

        return clp_labels, matrices
