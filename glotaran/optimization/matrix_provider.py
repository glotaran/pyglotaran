"""Module containing the matrix provider classes."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import xarray as xr

from glotaran.model import DatasetGroup
from glotaran.model import DatasetModel
from glotaran.model.dataset_model import has_dataset_model_global_model
from glotaran.model.dataset_model import iterate_dataset_model_global_megacomplexes
from glotaran.model.dataset_model import iterate_dataset_model_megacomplexes
from glotaran.model.interval_item import IntervalItem
from glotaran.model.item import fill_item
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


@dataclass
class MatrixContainer:
    """A container of matrix and the corresponding clp labels."""

    clp_labels: list[str]
    """The clp labels."""
    matrix: np.ndarray
    """The matrix."""

    @property
    def is_index_dependent(self) -> bool:
        """Check if the matrix is index dependent.

        Returns
        -------
        bool
            Whether the matrix is index dependent.
        """
        return len(self.matrix.shape) == 3

    @staticmethod
    def apply_weight(matrix: ArrayLike, weight: ArrayLike) -> ArrayLike:
        """Apply weight on a matrix.

        Parameters
        ----------
        matrix : ArrayLike
            The matrix.
        weight : ArrayLike
            The weight.

        Returns
        -------
        ArrayLike
            The weighted matrix.
        """
        return (matrix.T * weight).T

    def create_weighted_matrix(self, weight: ArrayLike) -> MatrixContainer:
        """Create a matrix container with a weighted matrix.

        Parameters
        ----------
        weight : ArrayLike
            The weight.

        Returns
        -------
        MatrixContainer
            The weighted matrix.
        """
        return replace(self, matrix=self.apply_weight(self.matrix, weight))

    def create_scaled_matrix(self, scale: float) -> MatrixContainer:
        """Create a matrix container with a scaled matrix.

        Parameters
        ----------
        scale : float
            The scale.

        Returns
        -------
        MatrixContainer
            The scaled matrix.
        """
        return replace(self, matrix=self.matrix * scale)


class MatrixProvider:
    """A class to provide matrix calculations for optimization."""

    def __init__(self, dataset_group: DatasetGroup):
        """Initialize a matrix provider for a dataset group.

        Parameters
        ----------
        dataset_group : DatasetGroup
            The dataset group.
        """
        self._group = dataset_group
        self._matrix_containers: dict[str, MatrixContainer] = {}
        self._global_matrix_containers: dict[str, MatrixContainer] = {}
        self._data_provider: DataProvider

    @property
    def group(self) -> DatasetGroup:
        """Get the dataset group.

        Returns
        -------
        DatasetGroup
            The dataset group.
        """
        return self._group

    def get_matrix_container(self, dataset_label: str) -> MatrixContainer:
        """Get the matrix container for a dataset on an index on the global axis.

        Parameters
        ----------
        dataset_label : str
            The label of the dataset.

        Returns
        -------
        MatrixContainer
            The matrix container.
        """
        return self._matrix_containers[dataset_label]

    def calculate_dataset_matrices(self):
        """Calculate the matrices of the datasets in the dataset group."""
        for label, dataset_model in self.group.dataset_models.items():
            model_axis = self._data_provider.get_model_axis(label)
            global_axis = self._data_provider.get_global_axis(label)

            self._matrix_containers[label] = self.calculate_dataset_matrix(
                dataset_model, global_axis, model_axis
            )

    @staticmethod
    def calculate_dataset_matrix(
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        global_matrix: bool = False,
    ) -> MatrixContainer:
        """Calculate the matrix for a dataset on an index on the global axis.

        Parameters
        ----------
        dataset_model : DatasetModel
            The dataset model.
        global_axis: ArrayLike
            The global axis.
        model_axis: ArrayLike
            The model axis.
        global_matrix: bool
            Calculate the global megacomplexes if `True`.

        Returns
        -------
        MatrixContainer
            The resulting matrix container.
        """
        clp_labels: list[str] = []
        matrix = None

        megacomplex_iterator = iterate_dataset_model_megacomplexes(dataset_model)

        if global_matrix:
            megacomplex_iterator = iterate_dataset_model_global_megacomplexes(dataset_model)
            model_axis, global_axis = global_axis, model_axis

        for scale, megacomplex in megacomplex_iterator:
            this_clp_labels, this_matrix = megacomplex.calculate_matrix(  # type:ignore[union-attr]
                dataset_model, global_axis, model_axis
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
        return MatrixContainer(clp_labels, matrix)  # type:ignore[arg-type]

    @staticmethod
    def combine_megacomplex_matrices(
        matrix_left: ArrayLike,
        matrix_right: ArrayLike,
        clp_labels_left: list[str],
        clp_labels_right: list[str],
    ) -> tuple[list[str], ArrayLike]:
        """Calculate the matrix for a dataset on an index on the global axis.

        Parameters
        ----------
        matrix_left: ArrayLike
            The left matrix.
        matrix_right: ArrayLike
            The right matrix.
        clp_labels_left: list[str]
            The left clp labels.
        clp_labels_right: list[str]
            The right clp labels.

        Returns
        -------
        tuple[list[str], ArrayLike]:
            The combined clp labels and matrix.
        """
        result_clp_labels = clp_labels_left + [
            c for c in clp_labels_right if c not in clp_labels_left
        ]
        result_clp_size = len(result_clp_labels)

        if len(matrix_left.shape) < len(matrix_right.shape):
            matrix_left, matrix_right = matrix_right, matrix_left

        left_index_dependent = len(matrix_left.shape) == 3
        right_index_dependent = len(matrix_right.shape) == 3

        result_shape = (
            (matrix_left.shape[0], matrix_left.shape[1], result_clp_size)
            if left_index_dependent
            else (matrix_left.shape[0], result_clp_size)
        )

        result_matrix = np.zeros(result_shape, dtype=np.float64)
        for idx, label in enumerate(result_clp_labels):
            if label in clp_labels_left:
                if left_index_dependent:
                    result_matrix[:, :, idx] += matrix_left[:, :, clp_labels_left.index(label)]
                else:
                    result_matrix[:, idx] += matrix_left[:, clp_labels_left.index(label)]
            if label in clp_labels_right:
                if left_index_dependent:
                    result_matrix[:, :, idx] += (
                        matrix_right[:, :, clp_labels_right.index(label)]
                        if right_index_dependent
                        else matrix_right[:, clp_labels_right.index(label)]
                    )
                else:
                    result_matrix[:, idx] += matrix_right[:, clp_labels_right.index(label)]
        return result_clp_labels, result_matrix

    @staticmethod
    def does_interval_item_apply(prop: IntervalItem, index: int | None) -> bool:
        """Check if an interval item applies on an index.

        Parameters
        ----------
        prop : IntervalItem
            The interval property.
        index: int | None
            The index to check.

        Returns
        -------
        bool
            Whether the property applies.
        """
        if prop.has_interval() and index is None:
            warnings.warn(
                f"Interval property '{prop}' applies on a matrix which is "
                f"not index dependent. This will be an error in 0.9.0. Set "
                "'index_dependent: true' on the dataset model to fix the issue."
            )
            return True
        return prop.applies(index)

    def reduce_matrix(
        self,
        matrix: MatrixContainer,
        global_axis: ArrayLike,
    ) -> list[MatrixContainer]:
        """Reduce a matrix.

        Applies constraints and relations.

        Parameters
        ----------
        matrix : MatrixContainer
            The matrix.
        global_axis: ArrayLike
            The global axis.

        Returns
        -------
        MatrixContainer
            The resulting matrix container.
        """
        result = (
            [
                MatrixContainer(matrix.clp_labels, matrix.matrix[i, :, :])
                for i in range(global_axis.size)
            ]
            if matrix.is_index_dependent
            else [matrix] * global_axis.size
        )
        result = self.apply_relations(result, global_axis)
        result = self.apply_constraints(result, global_axis)
        return result

    def apply_constraints(
        self,
        matrices: list[MatrixContainer],
        global_axis: ArrayLike,
    ) -> list[MatrixContainer]:
        """Apply constraints on a matrix.

        Parameters
        ----------
        matrices: list[MatrixContainer],
            The matrices.
        global_axis: ArrayLike
            The global axis.

        Returns
        -------
        MatrixContainer
            The resulting matrix container.
        """
        model = self.group.model
        if len(model.clp_constraints) == 0:
            return matrices

        for i, index in enumerate(global_axis):
            matrix = matrices[i]
            clp_labels = matrix.clp_labels
            removed_clp_labels = [
                c.target
                for c in model.clp_constraints
                if c.target in clp_labels and self.does_interval_item_apply(c, index)
            ]
            if len(removed_clp_labels) == 0:
                continue
            reduced_clp_labels = [c for c in clp_labels if c not in removed_clp_labels]
            mask = [label in reduced_clp_labels for label in clp_labels]
            reduced_matrix = matrix.matrix[:, mask]
            matrices[i] = MatrixContainer(reduced_clp_labels, reduced_matrix)
        return matrices

    def apply_relations(
        self,
        matrices: list[MatrixContainer],
        global_axis: ArrayLike,
    ) -> list[MatrixContainer]:
        """Apply relations on a matrix.

        Parameters
        ----------
        matrices: list[MatrixContainer],
            The matrices.
        global_axis: ArrayLike
            The global axis.

        Returns
        -------
        MatrixContainer
            The resulting matrix container.
        """
        model = self.group.model
        parameters = self.group.parameters

        if len(model.clp_relations) == 0:
            return matrices

        for i, index in enumerate(global_axis):
            matrix = matrices[i]

            clp_labels = matrix.clp_labels
            relation_matrix = np.diagflat([1.0 for _ in clp_labels])

            idx_to_delete = []
            for relation in model.clp_relations:
                if relation.target in clp_labels and self.does_interval_item_apply(
                    relation, index
                ):
                    if relation.source not in clp_labels:
                        continue

                    relation = fill_item(relation, model, parameters)  # type:ignore[arg-type]
                    source_idx = clp_labels.index(relation.source)
                    target_idx = clp_labels.index(relation.target)
                    relation_matrix[target_idx, source_idx] = relation.parameter
                    idx_to_delete.append(target_idx)

            if len(idx_to_delete) == 0:
                continue

            reduced_clp_labels = [
                label for i, label in enumerate(clp_labels) if i not in idx_to_delete
            ]
            relation_matrix = np.delete(relation_matrix, idx_to_delete, axis=1)
            reduced_matrix = matrix.matrix @ relation_matrix
            matrices[i] = MatrixContainer(reduced_clp_labels, reduced_matrix)
        return matrices

    def get_result(self) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
        """Get the results of the matrix calculations.

        Returns
        -------
        tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
            A tuple of the matrices and global matrices.

        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        matrices = {}
        global_matrices = {}
        for label, matrix_container in self._matrix_containers.items():
            model_dimension = self._data_provider.get_model_dimension(label)
            model_axis = self._data_provider.get_model_axis(label)
            matrix_coords: tuple[tuple[str, Any], tuple[str, Any], tuple[str, list[str]]] | tuple[
                tuple[str, Any], tuple[str, list[str]]
            ] = (
                (model_dimension, model_axis),
                ("clp_label", matrix_container.clp_labels),
            )
            if matrix_container.is_index_dependent:
                global_dimension = self._data_provider.get_global_dimension(label)
                global_axis = self._data_provider.get_global_axis(label)
                matrix_coords = (
                    (global_dimension, global_axis),
                    matrix_coords[0],
                    matrix_coords[1],
                )
            matrices[label] = xr.DataArray(matrix_container.matrix, coords=matrix_coords)

        for label, matrix_container in self._global_matrix_containers.items():
            global_dimension = self._data_provider.get_global_dimension(label)
            global_axis = self._data_provider.get_global_axis(label)
            global_matrices[label] = xr.DataArray(
                matrix_container.matrix,
                coords=(
                    (global_dimension, global_axis),
                    ("global_clp_label", matrix_container.clp_labels),
                ),
            )

        return global_matrices, matrices

    def calculate(self):
        """Calculate the matrices for optimization.

        .. # noqa: DAR401
        """
        raise NotImplementedError

    @property
    def number_of_clps(self) -> int:
        """Return number of conditionally linear parameters.

        Raises
        ------
        NotImplementedError
            This property needs to be implemented by subclasses.

        See Also
        --------
        MatrixProviderUnlinked
        MatrixProviderLinked
        """
        raise NotImplementedError


class MatrixProviderUnlinked(MatrixProvider):
    """A class to provide matrix calculations for optimization of unlinked dataset groups."""

    def __init__(self, group: DatasetGroup, data_provider: DataProvider):
        """Initialize a matrix provider for an unlinked dataset group.

        Parameters
        ----------
        dataset_group : DatasetGroup
            The dataset group.
        data_provider : DataProvider
            The data provider.
        """
        super().__init__(group)
        self._data_provider = data_provider
        self._prepared_matrix_container: dict[str, list[MatrixContainer]] = {}
        self._full_matrices: dict[str, ArrayLike] = {}

    def get_global_matrix_container(self, dataset_label: str) -> MatrixContainer:
        """Get the global matrix container for a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the dataset.

        Returns
        -------
        MatrixContainer
            The matrix container.
        """
        return self._global_matrix_containers[dataset_label]

    def get_prepared_matrix_container(
        self, dataset_label: str, global_index: int
    ) -> MatrixContainer:
        """Get the prepared matrix container for a dataset on an index on the global axis.

        Parameters
        ----------
        dataset_label : str
            The label of the dataset.
        global_index : int
            The index on the global axis.

        Returns
        -------
        MatrixContainer
            The matrix container.
        """
        return self._prepared_matrix_container[dataset_label][global_index]

    def get_full_matrix(self, dataset_label: str) -> ArrayLike:
        """Get the full matrix of a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the dataset.

        Returns
        -------
        ArrayLike
            The matrix.
        """
        return self._full_matrices[dataset_label]

    def calculate(self):
        """Calculate the matrices for optimization."""
        self.calculate_dataset_matrices()
        self.calculate_global_matrices()
        self.calculate_prepared_matrices()
        self.calculate_full_matrices()

    def calculate_global_matrices(self):
        """Calculate the global matrices of the datasets in the dataset group."""
        for label, dataset_model in self.group.dataset_models.items():
            if has_dataset_model_global_model(dataset_model):
                model_axis = self._data_provider.get_model_axis(label)
                global_axis = self._data_provider.get_global_axis(label)
                self._global_matrix_containers[label] = self.calculate_dataset_matrix(
                    dataset_model, global_axis, model_axis, global_matrix=True
                )

    def calculate_prepared_matrices(self):
        """Calculate the prepared matrices of the datasets in the dataset group."""
        for label, dataset_model in self.group.dataset_models.items():
            if has_dataset_model_global_model(dataset_model):
                continue
            scale = float(dataset_model.scale or 1)
            weight = self._data_provider.get_weight(label)
            self._prepared_matrix_container[label] = self.reduce_matrix(
                self.get_matrix_container(label).create_scaled_matrix(scale),
                self._data_provider.get_global_axis(label),
            )

            if weight is not None:
                self._prepared_matrix_container[label] = [
                    matrix.create_weighted_matrix(weight[:, i])
                    for i, matrix in enumerate(self._prepared_matrix_container[label])
                ]

    def calculate_full_matrices(self):
        """Calculate the full matrices of the datasets in the dataset group."""
        for label, dataset_model in self.group.dataset_models.items():
            if has_dataset_model_global_model(dataset_model):
                global_matrix_container = self.get_global_matrix_container(label)
                global_matrix = global_matrix_container.matrix
                matrix_container = self.get_matrix_container(label)
                matrix = matrix_container.matrix

                if matrix_container.is_index_dependent:
                    full_matrix = np.concatenate(
                        [
                            np.kron(global_matrix[i, :], matrix[i, :, :])
                            for i in range(matrix.shape[0])
                        ]
                    )
                else:
                    full_matrix = np.kron(global_matrix, matrix)

                weight = self._data_provider.get_flattened_weight(label)
                if weight is not None:
                    full_matrix = MatrixContainer.apply_weight(full_matrix, weight)

                self._full_matrices[label] = full_matrix

    @property
    def number_of_clps(self) -> int:
        """Return number of conditionally linear parameters.

        Returns
        -------
        int
        """
        nr_of_clps = 0
        for dataset_label, dataset_model in self.group.dataset_models.items():
            if has_dataset_model_global_model(dataset_model):
                model_clp_labels = self.get_matrix_container(dataset_label).clp_labels
                global_clp_labels = self.get_global_matrix_container(dataset_label).clp_labels
                nr_of_clps += len(model_clp_labels) * len(global_clp_labels)
            else:
                global_axis_indexes = range(
                    len(self._data_provider.get_global_axis(dataset_label))
                )
                nr_of_clps += sum(
                    len(self.get_prepared_matrix_container(dataset_label, index).clp_labels)
                    for index in global_axis_indexes
                )

        return nr_of_clps


class MatrixProviderLinked(MatrixProvider):
    """A class to provide matrix calculations for optimization of linked dataset groups."""

    def __init__(self, group: DatasetGroup, data_provider: DataProviderLinked):
        """Initialize a matrix provider for a linked dataset group.

        Parameters
        ----------
        dataset_group : DatasetGroup
            The dataset group.
        data_provider : DataProviderLinked
            The data provider.
        """
        super().__init__(group)
        self._data_provider: DataProviderLinked = data_provider
        self._aligned_full_clp_labels: list[list[str]] = [
            None  # type:ignore[list-item]
        ] * self._data_provider.aligned_global_axis.size
        self._aligned_matrices: list[MatrixContainer] = [
            None  # type:ignore[list-item]
        ] * self._data_provider.aligned_global_axis.size

    @property
    def aligned_full_clp_labels(self) -> list[list[str]]:
        """Get the aligned full clp labels.

        Returns
        -------
        list[list[str]]
            The full aligned clp labels.
        """
        return self._aligned_full_clp_labels

    def get_aligned_matrix_container(self, global_index: int) -> MatrixContainer:
        """Get the aligned matrix container for an index on the aligned global axis.

        Parameters
        ----------
        global_index : int
            The index on the global axis.

        Returns
        -------
        MatrixContainer
            The matrix container.
        """
        return self._aligned_matrices[global_index]

    def calculate(self):
        """Calculate the matrices for optimization."""
        self.calculate_dataset_matrices()
        self.calculate_aligned_matrices()

    def calculate_aligned_matrices(self):
        """Calculate the aligned matrices of the dataset group."""
        full_clp_labels = self.align_full_clp_labels()
        for i, global_index_value in enumerate(self._data_provider.aligned_global_axis):
            matrix_containers = []
            group_label = self._data_provider.get_aligned_group_label(i)
            for label, index in zip(
                self._data_provider.group_definitions[group_label],
                self._data_provider.get_aligned_dataset_indices(i),
            ):
                matrix_container_temp = self._matrix_containers[label]
                if matrix_container_temp.is_index_dependent:
                    matrix_containers.append(
                        MatrixContainer(
                            clp_labels=matrix_container_temp.clp_labels,
                            matrix=matrix_container_temp.matrix[index],
                        )
                    )
                else:
                    matrix_containers.append(matrix_container_temp)

            matrix_scales = [
                self.group.dataset_models[label].scale
                if self.group.dataset_models[label].scale is not None
                else 1
                for label in self._data_provider.group_definitions[group_label]
            ]

            group_matrix = self.align_matrices(
                matrix_containers, matrix_scales  # type:ignore[arg-type]
            )

            self._aligned_full_clp_labels[i] = full_clp_labels[group_label]
            group_matrix_single = self.reduce_matrix(
                group_matrix, np.array([self._data_provider.aligned_global_axis[i]])
            )[0]

            weight = self._data_provider.get_aligned_weight(i)
            if weight is not None:
                group_matrix_single = group_matrix_single.create_weighted_matrix(weight)

            self._aligned_matrices[i] = group_matrix_single

    def align_full_clp_labels(self) -> dict[str, list[str]]:
        """Align the unreduced clp labels.

        Returns
        -------
        dict[str, list[str]]
            The aligned clp for every group.
        """
        aligned_full_clp_labels: dict[str, list[str]] = {}

        for (
            group_label,
            dataset_labels,
        ) in self._data_provider.group_definitions.items():
            aligned_full_clp_labels[group_label] = []
            for dataset_label in dataset_labels:
                aligned_full_clp_labels[group_label] += [
                    label
                    for label in self.get_matrix_container(dataset_label).clp_labels
                    if label not in aligned_full_clp_labels[group_label]
                ]
        return aligned_full_clp_labels

    @staticmethod
    def align_matrices(matrices: list[MatrixContainer], scales: list[float]) -> MatrixContainer:
        """Align matrices.

        Parameters
        ----------
        matrices : list[MatrixContainer]
            The matrices to align.
        scales : list[float]
            The scales of the matrices.

        Returns
        -------
        MatrixContainer
            The aligned matrix container.
        """
        if len(matrices) == 1:
            return matrices[0]
        masks = []
        full_clp_labels: list[str] = []
        sizes = []
        dim1 = 0
        for matrix in matrices:
            clp_labels = matrix.clp_labels
            model_axis_size = matrix.matrix.shape[0]
            sizes.append(model_axis_size)
            dim1 += model_axis_size
            if len(full_clp_labels) == 0:
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

    @property
    def number_of_clps(self) -> int:
        """Return number of conditionally linear parameters.

        Returns
        -------
        int
        """
        global_axis_indexes = range(len(self._data_provider.aligned_global_axis))
        return sum(
            len(self.get_aligned_matrix_container(index).clp_labels)
            for index in global_axis_indexes
        )
