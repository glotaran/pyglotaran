from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from glotaran.model import ClpConstraint
from glotaran.model import ClpRelation
from glotaran.model import DataModel
from glotaran.model import GlotaranUserError
from glotaran.model import Megacomplex
from glotaran.model import iterate_data_model_megacomplexes
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.parameter import Parameter


@dataclass
class OptimizationMatrix:
    """A container of matrix and the corresponding clp labels."""

    clp_labels: list[str]
    """The clp labels."""
    array: np.typing.ArrayLike

    @property
    def is_index_dependent(self) -> bool:
        """Check if the matrix is index dependent.

        Returns
        -------
        bool
            Whether the matrix is index dependent.
        """
        return len(self.array.shape) == 3

    @property
    def global_size(self) -> int | None:
        return self.array.shape[0] if self.is_index_dependent else None

    @property
    def model_size(self) -> int:
        return self.array.shape[1 if self.is_index_dependent else 0]

    @classmethod
    def from_megacomplex(
        cls,
        scale: Parameter | None,
        megacomplex: Megacomplex,
        data_model: DataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
    ) -> OptimizationMatrix:
        """"""
        clp_labels, array = megacomplex.calculate_matrix(data_model, global_axis, model_axis)

        if scale is not None:
            array *= scale
        return cls(clp_labels, array)

    @classmethod
    def combine(cls, matrices: list[OptimizationMatrix]) -> OptimizationMatrix:
        """"""
        clp_labels = list({c for m in matrices for c in m.clp_labels})
        clp_size = len(clp_labels)
        model_axis_size = next(matrices).model_axis_size
        index_dependent_matrices = [m for m in matrices if m.is_index_dependent]
        global_axis_size = (
            index_dependent_matrices[0].global_axis_size if index_dependent_matrices else None
        )
        shape = (
            (global_axis_size, model_axis_size, clp_size)
            if global_axis_size
            else (model_axis_size, clp_size)
        )
        array = np.zeros(shape, dtype=np.float64)

        for matrix in matrices:
            clp_mask = [clp_labels.index(c) for c in matrix.clp_labels]
            array[:, clp_mask] = matrix.array
        return cls(clp_labels, array)

    @classmethod
    def from_data(cls, data: OptimizationData) -> OptimizationMatrix:
        """"""
        matrices = [
            cls.from_megacomplex(scale, megacomplex, data.model, data.global_axis, data.model_axis)
            for scale, megacomplex in iterate_data_model_megacomplexes(data.model)
        ]
        matrix = matrices[0] if len(matrices) == 1 else cls.combine(matrices)
        if data.weight is not None:
            matrix.weight(data.weight)
        return matrix

    @classmethod
    def from_linked_data(cls, linked_data: LinkedOptimizationData) -> list[OptimizationMatrix]:
        data_matrices = {label: cls.from_data(data) for label, data in linked_data.data.items()}

        return [
            cls.combine(
                [
                    data_matrices[label].at_index(index)
                    for label, index in zip(
                        linked_data.group_definitions[linked_data.group_labels[global_index]],
                        linked_data.datas_indices[global_index],
                    )
                ]
            )
            for global_index in range(linked_data.global_axis.size)
        ]

    def reduce(
        self,
        index: float,
        constraints: list[ClpConstraint],
        relations: list[ClpRelation],
    ) -> OptimizationMatrix:
        if self.is_index_dependent:
            raise GlotaranUserError("Cannot reduce index dependent matrix.")
        constraints = [c for c in constraints if c.applies(index)]
        relations = [r for r in relations if r.applies(index)]
        if len(constraints) + len(relations) == 0:
            return self

        if len(relations) > 0:

            relation_matrix = np.diagflat([1.0] * len(self.clp_labels))
            idx_to_delete = []
            for relation in relations:
                if relation.target in self.clp_labels and relation.source in self.clp_labels:

                    source_idx = self.clp_labels.index(relation.source)
                    target_idx = self.clp_labels.index(relation.target)
                    relation_matrix[target_idx, source_idx] = relation.parameter
                    idx_to_delete.append(target_idx)

            if len(idx_to_delete) > 0:
                self.clp_labels = [
                    label for i, label in enumerate(self.clp_labels) if i not in idx_to_delete
                ]
                relation_matrix = np.delete(relation_matrix, idx_to_delete, axis=1)
                self.array @= relation_matrix

        if len(constraints) > 0:
            removed_clp_labels = [c.target for c in constraints if c.target in self.clp_labels]
            if len(removed_clp_labels) > 0:
                mask = [label not in removed_clp_labels for label in self.clp_labels]
                self.clp_labels = [
                    label for label in self.clp_labels if label not in removed_clp_labels
                ]
                self.array = self.array[:, mask]

        return self

    def weight(self, weight: np.typing.ArrayLike) -> OptimizationMatrix:
        """Create a matrix container with a weighted matrix.

        Parameters
        ----------
        weight : np.typing.ArrayLike
            The weight.

        Returns
        -------
        OptimizationMatrix
            The weighted matrix.
        """
        if self.is_index_dependent:
            self.array *= weight.T[:, :, np.newaxis]
        else:
            self.array = self.array[np.newaxis, :, :] * weight.T[:, :, np.newaxis]
        return self

    def scale(self, scale: float) -> OptimizationMatrix:
        """Create a matrix container with a scaled matrix.

        Parameters
        ----------
        scale : float
            The scale.

        Returns
        -------
        OptimizationMatrix
            The scaled matrix.
        """
        self.array *= scale
        return self

    def at_index(self, index: int) -> OptimizationMatrix:
        """Create a matrix container with a scaled matrix.

        Parameters
        ----------
        scale : float
            The scale.

        Returns
        -------
        OptimizationMatrix
            The scaled matrix.
        """
        index_matrix = self.array
        if self.is_index_dependent:
            index_matrix = index_matrix[index, :, :]
        return replace(self, matrix=index_matrix)
