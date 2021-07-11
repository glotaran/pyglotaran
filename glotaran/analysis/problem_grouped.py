from __future__ import annotations

import collections
import itertools
from typing import Deque

import numba as nb
import numpy as np
import xarray as xr

from glotaran.analysis.problem import GroupedProblemDescriptor
from glotaran.analysis.problem import ParameterError
from glotaran.analysis.problem import Problem
from glotaran.analysis.problem import ProblemGroup
from glotaran.analysis.util import CalculatedMatrix
from glotaran.analysis.util import apply_weight
from glotaran.analysis.util import calculate_clp_penalties
from glotaran.analysis.util import calculate_matrix
from glotaran.analysis.util import find_closest_index
from glotaran.analysis.util import find_overlap
from glotaran.analysis.util import reduce_matrix
from glotaran.analysis.util import retrieve_clps
from glotaran.model import DatasetModel
from glotaran.project import Scheme


class GroupedProblem(Problem):
    """Represents a problem where the data is grouped."""

    def __init__(self, scheme: Scheme):
        """Initializes the Problem class from a scheme (:class:`glotaran.analysis.scheme.Scheme`)

        Args:
            scheme (Scheme): An instance of :class:`glotaran.analysis.scheme.Scheme`
                which defines your model, parameters, and data
        """
        super().__init__(scheme=scheme)

        # TODO: grouping should be user controlled not inferred automatically
        global_dimensions = {d.get_global_dimension() for d in self.dataset_models.values()}
        model_dimensions = {d.get_model_dimension() for d in self.dataset_models.values()}
        if len(global_dimensions) != 1:
            raise ValueError(
                f"Cannot group datasets. Global dimensions '{global_dimensions}' do not match."
            )
        if len(model_dimensions) != 1:
            raise ValueError(
                f"Cannot group datasets. Model dimension '{model_dimensions}' do not match."
            )
        self._index_dependent = any(d.index_dependent() for d in self.dataset_models.values())
        self._global_dimension = global_dimensions.pop()
        self._model_dimension = model_dimensions.pop()
        self._group_clp_labels = None
        self._groups = None
        self._has_weights = any("weight" in d for d in self._data.values())

    def init_bag(self):
        """Initializes a grouped problem bag."""
        self._bag = None
        datasets = None
        for label, dataset_model in self.dataset_models.items():

            data = dataset_model.get_data()
            weight = dataset_model.get_weight()
            if weight is None and self._has_weights:
                weight = np.ones_like(data)

            global_axis = dataset_model.get_global_axis()
            model_axis = dataset_model.get_model_axis()
            has_scaling = dataset_model.scale is not None

            if self._bag is None:
                self._bag = collections.deque(
                    ProblemGroup(
                        data=data[:, i],
                        weight=weight[:, i] if weight is not None else None,
                        has_scaling=has_scaling,
                        group=label,
                        data_sizes=[model_axis.size],
                        descriptor=[
                            GroupedProblemDescriptor(
                                label,
                                {
                                    self._global_dimension: i,
                                },
                                {
                                    self._model_dimension: model_axis,
                                    self._global_dimension: global_axis,
                                },
                            )
                        ],
                    )
                    for i, value in enumerate(global_axis)
                )
                datasets = collections.deque([label] for _, _ in enumerate(global_axis))
                self._full_axis = collections.deque(global_axis)
            else:
                self._append_to_grouped_bag(
                    label, datasets, global_axis, model_axis, data, weight, has_scaling
                )
        self._full_axis = np.asarray(self._full_axis)
        self._groups = {"".join(d): d for d in datasets}

    def _append_to_grouped_bag(
        self,
        label: str,
        datasets: Deque[str],
        global_axis: np.ndarray,
        model_axis: np.ndarray,
        data: xr.DataArray,
        weight: xr.DataArray,
        has_scaling: bool,
    ):
        i1, i2 = find_overlap(self._full_axis, global_axis, atol=self._scheme.group_tolerance)

        for i, j in enumerate(i1):
            datasets[j].append(label)
            data_stripe = data[:, i2[i]]
            self._bag[j] = ProblemGroup(
                data=np.concatenate(
                    [
                        self._bag[j].data,
                        data_stripe,
                    ]
                ),
                weight=np.concatenate([self._bag[j].weight, weight[:, i2[i]]])
                if weight is not None
                else None,
                has_scaling=has_scaling or self._bag[j].has_scaling,
                group=self._bag[j].group + label,
                data_sizes=self._bag[j].data_sizes + [data_stripe.size],
                descriptor=self._bag[j].descriptor
                + [
                    GroupedProblemDescriptor(
                        label,
                        {
                            self._global_dimension: i2[i],
                        },
                        {
                            self._model_dimension: model_axis,
                            self._global_dimension: global_axis,
                        },
                    )
                ],
            )

        # Add non-overlaping regions
        begin_overlap = i2[0] if len(i2) != 0 else 0
        end_overlap = i2[-1] + 1 if len(i2) != 0 else 0
        for i in itertools.chain(range(begin_overlap), range(end_overlap, len(global_axis))):
            data_stripe = data[:, i]
            problem = ProblemGroup(
                data=data_stripe,
                weight=weight[:, i] if weight is not None else None,
                has_scaling=has_scaling,
                group=label,
                data_sizes=[data_stripe.size],
                descriptor=[
                    GroupedProblemDescriptor(
                        label,
                        {
                            self._global_dimension: i,
                        },
                        {
                            self._model_dimension: model_axis,
                            self._global_dimension: global_axis,
                        },
                    )
                ],
            )
            if i < end_overlap:
                datasets.appendleft([label])
                self._full_axis.appendleft(global_axis[i])
                self._bag.appendleft(problem)
            else:
                datasets.append([label])
                self._full_axis.append(global_axis[i])
                self._bag.append(problem)

    @property
    def groups(self) -> dict[str, list[str]]:
        if not self._groups:
            self.init_bag()
        return self._groups

    def calculate_matrices(self):
        if self._parameters is None:
            raise ParameterError
        if self._index_dependent:
            self.calculate_index_dependent_matrices()
        else:
            self.calculate_index_independent_matrices()

    def calculate_index_dependent_matrices(
        self,
    ) -> tuple[dict[str, list[CalculatedMatrix]], list[CalculatedMatrix],]:
        """Calculates the index dependent model matrices."""

        def calculate_group(
            group: ProblemGroup, descriptors: dict[str, DatasetModel]
        ) -> tuple[list[CalculatedMatrix], list[str], CalculatedMatrix]:
            matrices = [
                calculate_matrix(
                    descriptors[problem.label],
                    problem.indices,
                )
                for problem in group.descriptor
            ]
            global_index = group.descriptor[0].indices[self._global_dimension]
            global_index = group.descriptor[0].axis[self._global_dimension][global_index]
            combined_matrix = combine_matrices(matrices)
            group_clp_labels = combined_matrix.clp_labels
            reduced_matrix = reduce_matrix(
                combined_matrix, self.model, self.parameters, global_index
            )
            return matrices, group_clp_labels, reduced_matrix

        results = list(map(lambda group: calculate_group(group, self.dataset_models), self.bag))

        matrices = list(map(lambda result: result[0], results))

        self._matrices = {}
        for i, grouped_problem in enumerate(self._bag):
            for j, descriptor in enumerate(grouped_problem.descriptor):
                if descriptor.label not in self._matrices:
                    self._matrices[descriptor.label] = []
                self._matrices[descriptor.label].append(matrices[i][j])

        self._group_clp_labels = list(map(lambda result: result[1], results))
        self._reduced_matrices = list(map(lambda result: result[2], results))
        return self._matrices, self._reduced_matrices

    def calculate_index_independent_matrices(
        self,
    ) -> tuple[dict[str, CalculatedMatrix], dict[str, CalculatedMatrix],]:
        """Calculates the index independent model matrices."""
        self._matrices = {}
        self._group_clp_labels = {}
        self._reduced_matrices = {}

        for label, dataset_model in self.dataset_models.items():
            self._matrices[label] = calculate_matrix(
                dataset_model,
                {},
            )
            self._group_clp_labels[label] = self._matrices[label].clp_labels
            self._reduced_matrices[label] = reduce_matrix(
                self._matrices[label],
                self.model,
                self.parameters,
                None,
            )

        for group_label, group in self.groups.items():
            if group_label not in self._matrices:
                self._reduced_matrices[group_label] = combine_matrices(
                    [self._reduced_matrices[label] for label in group]
                )
                self._group_clp_labels[group_label] = list(
                    set(itertools.chain(*(self._matrices[label].clp_labels for label in group)))
                )

        return self._matrices, self._reduced_matrices

    def calculate_residual(self):
        results = (
            list(
                map(
                    self._index_dependent_residual,
                    self.bag,
                    self.reduced_matrices,
                    self._group_clp_labels,
                    self._full_axis,
                )
            )
            if self._index_dependent
            else list(map(self._index_independent_residual, self.bag, self._full_axis))
        )

        self._clp_labels = list(map(lambda result: result[0], results))
        self._grouped_clps = list(map(lambda result: result[1], results))

        self._weighted_residuals = list(map(lambda result: result[2], results))
        self._residuals = list(map(lambda result: result[3], results))
        self._additional_penalty = calculate_clp_penalties(
            self.model, self.parameters, self._clp_labels, self._grouped_clps, self._full_axis
        )

        return self._reduced_clps, self._clps, self._weighted_residuals, self._residuals

    def _index_dependent_residual(
        self,
        problem: ProblemGroup,
        matrix: CalculatedMatrix,
        clp_labels: str,
        index: any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        reduced_clp_labels = matrix.clp_labels
        matrix = matrix.matrix
        if problem.weight is not None:
            apply_weight(matrix, problem.weight)
        data = problem.data
        if problem.has_scaling:
            for i, descriptor in enumerate(problem.descriptor):
                label = descriptor.label
                if self.dataset_models[label] is not None:
                    start = sum(problem.data_sizes[0:i])
                    end = start + problem.data_sizes[i]
                    matrix[start:end, :] *= self.dataset_models[label].scale

        reduced_clps, weighted_residual = self._residual_function(matrix, data)
        clps = retrieve_clps(
            self.model,
            self.parameters,
            clp_labels,
            reduced_clp_labels,
            reduced_clps,
            index,
        )
        residual = (
            weighted_residual / problem.weight if problem.weight is not None else weighted_residual
        )
        return clp_labels, clps, weighted_residual, residual

    def _index_independent_residual(self, problem: ProblemGroup, index: any):
        matrix = self.reduced_matrices[problem.group]
        reduced_clp_labels = matrix.clp_labels
        matrix = matrix.matrix.copy()
        if problem.weight is not None:
            apply_weight(matrix, problem.weight)
        data = problem.data
        if problem.has_scaling:
            for i, descriptor in enumerate(problem.descriptor):
                label = descriptor.label
                if self.dataset_models[label] is not None:
                    start = sum(problem.data_sizes[0:i])
                    end = start + problem.data_sizes[i]
                    matrix[start:end, :] *= self.dataset_models[label].scale
        reduced_clps, weighted_residual = self._residual_function(matrix, data)
        clp_labels = self._group_clp_labels[problem.group]
        clps = retrieve_clps(
            self.model,
            self.parameters,
            clp_labels,
            reduced_clp_labels,
            reduced_clps,
            index,
        )
        residual = (
            weighted_residual / problem.weight if problem.weight is not None else weighted_residual
        )
        return clp_labels, clps, weighted_residual, residual

    def prepare_result_creation(self):
        if self._residuals is None:
            self.calculate_residual()
        full_clp_labels = self._clp_labels
        full_clps = self._grouped_clps
        self._clps = {}
        for label, matrix in self.matrices.items():
            # TODO deal with different clps at indices
            clp_labels = matrix[0].clp_labels if self._index_dependent else matrix.clp_labels

            # find offset in the full axis
            global_axis = self.dataset_models[label].get_global_axis()
            offset = find_closest_index(global_axis[0], self._full_axis)

            clps = []
            for i in range(global_axis.size):
                full_index_clp_labels = full_clp_labels[i + offset]
                index_clps = full_clps[i + offset]
                mask = [full_index_clp_labels.index(clp_label) for clp_label in clp_labels]
                clps.append(index_clps[mask])

            self._clps[label] = xr.DataArray(
                clps,
                coords=((self._global_dimension, global_axis), ("clp_label", clp_labels)),
            )

    def create_index_dependent_result_dataset(self, label: str, dataset: xr.Dataset) -> xr.Dataset:
        """Creates a result datasets for index dependent matrices."""

        for index, grouped_problem in enumerate(self.bag):

            if label in grouped_problem.group:
                group_index = [
                    descriptor.label for descriptor in grouped_problem.descriptor
                ].index(label)
                group_descriptor = grouped_problem.descriptor[group_index]
                global_index = group_descriptor.indices[self._global_dimension]
                global_index = group_descriptor.axis[self._global_dimension][global_index]

                self._add_grouped_residual_to_dataset(
                    dataset, grouped_problem, index, group_index, global_index
                )

        dataset["matrix"] = (
            (
                (self._global_dimension),
                (self._model_dimension),
                ("clp_label"),
            ),
            np.asarray([m.matrix for m in self.matrices[label]]),
        )
        dataset["clp"] = self.clps[label]

        return dataset

    def create_index_independent_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:
        """Creates a result datasets for index independent matrices."""

        dataset["matrix"] = (
            (
                (self._model_dimension),
                ("clp_label"),
            ),
            self.matrices[label].matrix,
        )
        dataset["clp"] = self.clps[label]

        for index, grouped_problem in enumerate(self.bag):

            if label in grouped_problem.group:
                group_index = [
                    descriptor.label for descriptor in grouped_problem.descriptor
                ].index(label)
                group_descriptor = grouped_problem.descriptor[group_index]
                global_index = group_descriptor.indices[self._global_dimension]
                global_index = group_descriptor.axis[self._global_dimension][global_index]

                self._add_grouped_residual_to_dataset(
                    dataset, grouped_problem, index, group_index, global_index
                )

        return dataset

    def _add_grouped_residual_to_dataset(
        self,
        dataset: xr.Dataset,
        grouped_problem: ProblemGroup,
        index: int,
        group_index: int,
        global_index: int,
    ):
        if "residual" not in dataset:
            dim1 = dataset.coords[self._model_dimension].size
            dim2 = dataset.coords[self._global_dimension].size
            dataset["weighted_residual"] = (
                (self._model_dimension, self._global_dimension),
                np.zeros((dim1, dim2), dtype=np.float64),
            )
            dataset["residual"] = (
                (self._model_dimension, self._global_dimension),
                np.zeros((dim1, dim2), dtype=np.float64),
            )

        start = sum(
            self.data[grouped_problem.descriptor[i].label].coords[self._model_dimension].size
            for i in range(group_index)
        )

        end = start + dataset.coords[self._model_dimension].size
        dataset.weighted_residual.loc[
            {self._global_dimension: global_index}
        ] = self.weighted_residuals[index][start:end]
        dataset.residual.loc[{self._global_dimension: global_index}] = self.residuals[index][
            start:end
        ]

    @property
    def full_penalty(self) -> np.ndarray:
        if self._full_penalty is None:
            residuals = self.weighted_residuals
            additional_penalty = self.additional_penalty

            self._full_penalty = (
                np.concatenate((np.concatenate(residuals), additional_penalty))
                if additional_penalty is not None
                else np.concatenate(residuals)
            )
        return self._full_penalty


@nb.jit(nopython=True, parallel=True)
def _apply_weight(matrix, weight):
    for i in range(matrix.shape[1]):
        matrix[:, i] *= weight


def combine_matrices(matrices: list[CalculatedMatrix]) -> CalculatedMatrix:
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
        full_matrix[start:end, masks[i]] = m.matrix
        start = end

    return CalculatedMatrix(full_clp_labels, full_matrix)
