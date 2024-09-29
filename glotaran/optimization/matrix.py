from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model.data_model import DataModel
from glotaran.model.data_model import iterate_data_model_elements
from glotaran.model.data_model import iterate_data_model_global_elements
from glotaran.model.errors import GlotaranModelError
from glotaran.model.errors import GlotaranUserError

if TYPE_CHECKING:
    from glotaran.model.clp_constraint import ClpConstraint
    from glotaran.model.clp_relation import ClpRelation
    from glotaran.model.element import Element
    from glotaran.optimization.data import LinkedOptimizationData
    from glotaran.optimization.data import OptimizationData
    from glotaran.parameter import Parameter
    from glotaran.typing.types import ArrayLike


@dataclass
class OptimizationMatrix:
    """A container of matrix and the corresponding clp labels."""

    clp_axis: list[str]
    """The clp labels."""
    array: ArrayLike
    constraints: list[ClpConstraint] = field(default_factory=list)

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
    def global_axis_size(self) -> int | None:
        return self.array.shape[0] if self.is_index_dependent else None

    @property
    def model_axis_size(self) -> int:
        return self.array.shape[1 if self.is_index_dependent else 0]

    @classmethod
    def from_element(
        cls,
        scale: Parameter | None,
        element: Element,
        data_model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
    ) -> OptimizationMatrix:
        """"""
        clp_axis, array = element.calculate_matrix(data_model, global_axis, model_axis)

        if scale is not None:
            array *= scale
        return cls(clp_axis, array, element.clp_constraints)

    @classmethod
    def combine(cls, matrices: list[OptimizationMatrix]) -> OptimizationMatrix:
        """"""
        clp_axis = list({c for m in matrices for c in m.clp_axis})
        clp_size = len(clp_axis)
        model_axis_size = matrices[0].model_axis_size
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
            clp_mask = [clp_axis.index(c) for c in matrix.clp_axis]
            array[..., clp_mask] += matrix.array
        return cls(clp_axis, array, [c for m in matrices for c in m.constraints])

    @classmethod
    def link(cls, matrices: list[OptimizationMatrix]) -> OptimizationMatrix:
        """"""
        clp_axis = list(dict.fromkeys(c for m in matrices for c in m.clp_axis))
        clp_size = len(clp_axis)
        model_axis_size = sum(chain([m.model_axis_size for m in matrices]))
        shape = (model_axis_size, clp_size)
        array = np.zeros(shape, dtype=np.float64)

        current_element_index, current_element_index_end = 0, 0
        for matrix in matrices:
            clp_mask = [clp_axis.index(c) for c in matrix.clp_axis]
            current_element_index_end = current_element_index + matrix.model_axis_size
            array[current_element_index:current_element_index_end, clp_mask] = matrix.array
            current_element_index = current_element_index_end
        return cls(clp_axis, array, [c for m in matrices for c in m.constraints])

    @classmethod
    def from_data_model(
        cls,
        model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        weight: ArrayLike | None,
        global_matrix: bool = False,
    ) -> OptimizationMatrix:
        """"""
        element_iterator = (
            iterate_data_model_global_elements if global_matrix else iterate_data_model_elements
        )
        if global_matrix:
            model_axis, global_axis = global_axis, model_axis

        matrices = [
            cls.from_element(scale, element, model, global_axis, model_axis)  # type:ignore[arg-type]
            for scale, element in element_iterator(model)
        ]
        matrix = matrices[0] if len(matrices) == 1 else cls.combine(matrices)
        if weight is not None:
            matrix.weight(weight.T if global_matrix else weight)
        return matrix

    @classmethod
    def from_data(
        cls,
        data: OptimizationData,
        apply_weight: bool = True,
        global_matrix: bool = False,
    ) -> OptimizationMatrix:
        """"""
        return cls.from_data_model(
            data.model,
            data.global_axis,
            data.model_axis,
            data.weight if apply_weight else None,
            global_matrix=global_matrix,
        )

    @classmethod
    def from_global_data(
        cls, data: OptimizationData
    ) -> tuple[
        OptimizationMatrix,
        OptimizationMatrix,
        OptimizationMatrix,
    ]:
        matrix = cls.from_data(data, apply_weight=False)
        global_matrix = cls.from_data(data, apply_weight=False, global_matrix=True)

        if global_matrix.is_index_dependent:
            raise GlotaranModelError("Index dependent global matrices are not supported.")

        clp_axis = [
            label
            for gl in global_matrix.clp_axis
            for label in [f"{gl}@{ml}" for ml in matrix.clp_axis]
        ]

        array = (
            np.concatenate(
                [
                    np.kron(global_matrix.array[i, :], matrix.array[i, :, :])
                    for i in range(data.global_axis.size)
                ]
            )
            if matrix.is_index_dependent
            else np.kron(global_matrix.array, matrix.array)
        )

        if data.flat_weight is not None:
            array *= data.flat_weight[:, np.newaxis]

        return matrix, global_matrix, cls(clp_axis, array)

    @classmethod
    def from_linked_data(
        cls,
        linked_data: LinkedOptimizationData,
        matrices: dict[str, OptimizationMatrix] | None = None,
    ) -> list[OptimizationMatrix]:
        matrices = matrices or {
            label: cls.from_data(data) for label, data in linked_data.data.items()
        }

        return [
            cls.link(
                [
                    matrices[label].at_index(index).scale(linked_data.scales[label])
                    for label, index in zip(
                        linked_data.group_definitions[linked_data.group_labels[global_index]],
                        linked_data.data_indices[global_index],
                    )
                ]
            )
            for global_index in range(linked_data.global_axis.size)
        ]

    def reduce(
        self,
        index: float,
        relations: list[ClpRelation],
        copy: bool = False,
    ) -> OptimizationMatrix:
        result = deepcopy(self) if copy else self
        if result.is_index_dependent:
            raise GlotaranUserError("Cannot reduce index dependent matrix.")
        constraints = [c for c in self.constraints if c.applies(index)]
        relations = [r for r in relations if r.applies(index)]
        if len(constraints) + len(relations) == 0:
            return result

        if len(relations) > 0:
            relation_matrix = np.diagflat([1.0] * len(result.clp_axis))
            idx_to_delete = []
            for relation in relations:
                if relation.target in result.clp_axis and relation.source in result.clp_axis:
                    source_idx = result.clp_axis.index(relation.source)
                    target_idx = result.clp_axis.index(relation.target)
                    relation_matrix[target_idx, source_idx] = relation.parameter
                    idx_to_delete.append(target_idx)

            if len(idx_to_delete) > 0:
                result.clp_axis = [
                    label for i, label in enumerate(result.clp_axis) if i not in idx_to_delete
                ]
                relation_matrix = np.delete(relation_matrix, idx_to_delete, axis=1)
                result.array = result.array @ relation_matrix

        if len(constraints) > 0:
            removed_clp_labels = [c.target for c in constraints if c.target in result.clp_axis]
            if len(removed_clp_labels) > 0:
                mask = [label not in removed_clp_labels for label in result.clp_axis]
                result.clp_axis = [
                    label for label in result.clp_axis if label not in removed_clp_labels
                ]
                result.array = result.array[:, mask]

        return result

    def weight(self, weight: ArrayLike) -> OptimizationMatrix:
        """Create a matrix container with a weighted matrix.

        Parameters
        ----------
        weight : ArrayLike
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

    def scale(self, scale: float | Parameter) -> OptimizationMatrix:
        """Create a matrix container with a scaled matrix.

        Parameters
        ----------
        scale : float | Parameter
            The scale.

        Returns
        -------
        OptimizationMatrix
            The scaled matrix.
        """
        self.array *= scale
        return self

    def at_index(self, index: int) -> OptimizationMatrix:
        """Get the matrix at a global index.

        Parameters
        ----------
        index : int
            The global index.

        Returns
        -------
        OptimizationMatrix
            The matrix at the index. A copy of the array if not index dependent.
        """
        if self.is_index_dependent:  # noqa: SIM108
            index_array = self.array[index, :, :]
        else:
            # necessary if relations are applied
            index_array = self.array.copy()
        return replace(self, array=index_array)

    def as_global_list(self, global_axis: ArrayLike) -> list[OptimizationMatrix]:
        """Get a list of matrices.

        Parameters
        ----------
        global_axis : ArrayLike
            The global axis.

        Returns
        -------
        list[OptimizationMatrix]
            A list of matrices.
        """
        return [self.at_index(i) for i in range(global_axis.size)]

    def to_data_array(
        self, global_dim: str, global_axis: ArrayLike, model_dim: str, model_axis: ArrayLike
    ) -> xr.DataArray:
        coords = {model_dim: model_axis, "amplitude_label": self.clp_axis}
        if self.is_index_dependent:
            coords = {global_dim: global_axis} | coords

        return xr.DataArray(self.array, dims=coords.keys(), coords=coords)
