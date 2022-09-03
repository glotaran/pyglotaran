"""Module containing the matrix provider classes."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import replace

import numba as nb
import numpy as np
import xarray as xr

from glotaran.model import DatasetGroup
from glotaran.model import DatasetModel
from glotaran.model.interval_property import IntervalProperty
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked


@dataclass
class MatrixContainer:
    """A container of matrix and the corresponding clp labels."""

    clp_labels: list[str]
    """The clp labels."""
    matrix: np.ndarray
    """The matrix."""

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def apply_weight(
        matrix: np.typing.ArrayLike, weight: np.typing.ArrayLike
    ) -> np.typing.ArrayLike:
        """Apply weight on a matrix.

        Parameters
        ----------
        matrix : np.typing.ArrayLike
            The matrix.
        weight : np.typing.ArrayLike
            The weight.

        Returns
        -------
        np.typing.ArrayLike
            The weighted matrix.
        """
        matrix = matrix.copy()
        for i in nb.prange(matrix.shape[1]):
            matrix[:, i] *= weight
        return matrix

    def create_weighted_matrix(self, weight: np.typing.ArrayLike) -> MatrixContainer:
        """Create a matrix container with a weighted matrix.

        Parameters
        ----------
        weight : np.typing.ArrayLike
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
        self._matrix_containers: dict[str, MatrixContainer | list[MatrixContainer]] = {}
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

    def get_matrix_container(self, dataset_label: str, global_index: int) -> MatrixContainer:
        """Get the matrix container for a dataset on an index on the global axis.

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
        matrix_container = self._matrix_containers[dataset_label]
        if self.group.dataset_models[dataset_label].is_index_dependent():
            matrix_container = matrix_container[global_index]  # type:ignore[index]
        return matrix_container  # type:ignore[return-value]

    def calculate_dataset_matrices(self):
        """Calculate the matrices of the datasets in the dataset group."""
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
        global_matrix: bool = False,
    ) -> MatrixContainer:
        """Calculate the matrix for a dataset on an index on the global axis.

        Parameters
        ----------
        dataset_model : DatasetModel
            The dataset model.
        global_index : int | None
            The index on the global axis.
        global_axis: np.typing.ArrayLike
            The global axis.
        model_axis: np.typing.ArrayLike
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

        megacomplex_iterator = dataset_model.iterate_megacomplexes

        if global_matrix:
            megacomplex_iterator = dataset_model.iterate_global_megacomplexes
            model_axis, global_axis = global_axis, model_axis

        for scale, megacomplex in megacomplex_iterator():
            this_clp_labels, this_matrix = megacomplex.calculate_matrix(  # type:ignore[union-attr]
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
        matrix_left: np.typing.ArrayLike,
        matrix_right: np.typing.ArrayLike,
        clp_labels_left: list[str],
        clp_labels_right: list[str],
    ) -> tuple[list[str], np.typing.ArrayLike]:
        """Calculate the matrix for a dataset on an index on the global axis.

        Parameters
        ----------
        matrix_left: np.typing.ArrayLike
            The left matrix.
        matrix_right: np.typing.ArrayLike
            The right matrix.
        clp_labels_left: list[str]
            The left clp labels.
        clp_labels_right: list[str]
            The right clp labels.

        Returns
        -------
        tuple[list[str], np.typing.ArrayLike]:
            The combined clp labels and matrix.
        """
        tmp_clp_labels = clp_labels_left + [
            c for c in clp_labels_right if c not in clp_labels_left
        ]
        tmp_matrix = np.zeros((matrix_left.shape[0], len(tmp_clp_labels)), dtype=np.float64)
        for idx, label in enumerate(tmp_clp_labels):
            if label in clp_labels_left:
                tmp_matrix[:, idx] += matrix_left[:, clp_labels_left.index(label)]
            if label in clp_labels_right:
                tmp_matrix[:, idx] += matrix_right[:, clp_labels_right.index(label)]
        return tmp_clp_labels, tmp_matrix

    @staticmethod
    def does_interval_property_apply(prop: IntervalProperty, index: int | None) -> bool:
        """Check if an interval property applies on an index.

        Parameters
        ----------
        prop : IntervalProperty
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
        index: int | None,
    ) -> MatrixContainer:
        """Reduce a matrix.

        Applies constraints and relations.

        Parameters
        ----------
        matrix : MatrixContainer
            The matrix.
        index : int | None
            The index on the global axis.

        Returns
        -------
        MatrixContainer
            The resulting matrix container.
        """
        matrix = self.apply_relations(matrix, index)
        matrix = self.apply_constraints(matrix, index)
        return matrix

    def apply_constraints(
        self,
        matrix: MatrixContainer,
        index: int | None,
    ) -> MatrixContainer:
        """Apply constraints on a matrix.

        Parameters
        ----------
        matrix : MatrixContainer
            The matrix.
        index : int | None
            The index on the global axis.

        Returns
        -------
        MatrixContainer
            The resulting matrix container.
        """
        model = self.group.model
        if len(model.clp_constraints) == 0:  # type:ignore[attr-defined]
            return matrix

        clp_labels = matrix.clp_labels
        removed_clp_labels = [
            c.target  # type:ignore[attr-defined]
            for c in model.clp_constraints  # type:ignore[attr-defined]
            if c.target in clp_labels  # type:ignore[attr-defined]
            and self.does_interval_property_apply(c, index)  # type:ignore[arg-type]
        ]
        reduced_clp_labels = [c for c in clp_labels if c not in removed_clp_labels]
        mask = [label in reduced_clp_labels for label in clp_labels]
        reduced_matrix = matrix.matrix[:, mask]
        return MatrixContainer(reduced_clp_labels, reduced_matrix)

    def apply_relations(
        self,
        matrix: MatrixContainer,
        index: int | None,
    ) -> MatrixContainer:
        """Apply relations on a matrix.

        Parameters
        ----------
        matrix : MatrixContainer
            The matrix.
        index : int | None
            The index on the global axis.

        Returns
        -------
        MatrixContainer
            The resulting matrix container.
        """
        model = self.group.model
        parameters = self.group.parameters

        if len(model.clp_relations) == 0:
            return matrix

        clp_labels = matrix.clp_labels
        relation_matrix = np.diagflat([1.0 for _ in clp_labels])

        idx_to_delete = []
        for relation in model.clp_relations:  # type:ignore[attr-defined]
            if (
                relation.target in clp_labels  # type:ignore[attr-defined]
                and self.does_interval_property_apply(
                    relation, index  # type:ignore[arg-type]
                )
            ):

                if relation.source not in clp_labels:  # type:ignore[attr-defined]
                    continue

                relation = relation.fill(model, parameters)  # type:ignore[attr-defined]
                source_idx = clp_labels.index(relation.source)  # type:ignore[attr-defined]
                target_idx = clp_labels.index(relation.target)  # type:ignore[attr-defined]
                relation_matrix[
                    target_idx, source_idx
                ] = relation.parameter  # type:ignore[attr-defined]
                idx_to_delete.append(target_idx)

        reduced_clp_labels = [
            label for i, label in enumerate(clp_labels) if i not in idx_to_delete
        ]
        relation_matrix = np.delete(relation_matrix, idx_to_delete, axis=1)
        reduced_matrix = matrix.matrix @ relation_matrix
        return MatrixContainer(reduced_clp_labels, reduced_matrix)

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
            if self.group.dataset_models[label].is_index_dependent():
                global_dimension = self._data_provider.get_global_dimension(label)
                global_axis = self._data_provider.get_global_axis(label)
                matrices[label] = xr.concat(
                    [
                        xr.DataArray(
                            container.matrix,
                            coords=(
                                (model_dimension, model_axis),
                                ("clp_label", container.clp_labels),
                            ),
                        )
                        for container in matrix_container  # type:ignore[union-attr]
                    ],
                    dim=global_dimension,
                )
                matrices[label].coords[global_dimension] = global_axis
            else:
                matrices[label] = xr.DataArray(
                    matrix_container.matrix,  # type:ignore[union-attr]
                    coords=(
                        (model_dimension, model_axis),
                        ("clp_label", matrix_container.clp_labels),  # type:ignore[union-attr]
                    ),
                )
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
        self._full_matrices: dict[str, np.ArrayLike] = {}

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

    def get_full_matrix(self, dataset_label: str) -> np.ArrayLike:
        """Get the full matrix of a dataset.

        Parameters
        ----------
        dataset_label : str
            The label of the dataset.

        Returns
        -------
        np.typing.ArrayLike
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
            if dataset_model.has_global_model():
                model_axis = self._data_provider.get_model_axis(label)
                global_axis = self._data_provider.get_global_axis(label)
                self._global_matrix_containers[label] = self.calculate_dataset_matrix(
                    dataset_model, None, global_axis, model_axis, global_matrix=True
                )

    def calculate_prepared_matrices(self):
        """Calculate the prepared matrices of the datasets in the dataset group."""
        for label, dataset_model in self.group.dataset_models.items():
            if dataset_model.has_global_model():
                continue
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

    def calculate_full_matrices(self):
        """Calculate the full matrices of the datasets in the dataset group."""
        for label, dataset_model in self.group.dataset_models.items():
            if dataset_model.has_global_model():
                global_matrix_container = self.get_global_matrix_container(label)

                if dataset_model.is_index_dependent():
                    global_axis = self._data_provider.get_global_axis(label)
                    full_matrix = np.concatenate(
                        [
                            np.kron(
                                global_matrix_container.matrix[i, :],
                                self.get_matrix_container(label, i).matrix,
                            )
                            for i in range(global_axis.size)
                        ]
                    )
                else:
                    full_matrix = np.kron(
                        global_matrix_container.matrix,
                        self.get_matrix_container(label, 0).matrix,
                    )

                weight = self._data_provider.get_flattened_weight(label)
                if weight is not None:
                    full_matrix = MatrixContainer.apply_weight(full_matrix, weight)

                self._full_matrices[label] = full_matrix


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
        self._data_provider = data_provider
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
        for i, global_index_value in enumerate(self._data_provider.aligned_global_axis):
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
            group_matrix = self.reduce_matrix(group_matrix, global_index_value)
            weight = self._data_provider.get_aligned_weight(i)
            if weight is not None:
                group_matrix = group_matrix.create_weighted_matrix(weight)

            self._aligned_matrices[i] = group_matrix

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
