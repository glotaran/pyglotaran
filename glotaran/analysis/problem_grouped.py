from __future__ import annotations

import collections
import itertools
from typing import Deque

import numpy as np
import xarray as xr

from glotaran.analysis.problem import GroupedProblemDescriptor
from glotaran.analysis.problem import ParameterError
from glotaran.analysis.problem import Problem
from glotaran.analysis.problem import ProblemGroup
from glotaran.analysis.util import calculate_matrix
from glotaran.analysis.util import find_closest_index
from glotaran.analysis.util import find_overlap
from glotaran.analysis.util import reduce_matrix
from glotaran.analysis.util import retrieve_clps
from glotaran.model import DatasetDescriptor
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
        global_dimensions = {
            d.get_global_dimension() for d in self.filled_dataset_descriptors.values()
        }
        model_dimensions = {
            d.get_model_dimension() for d in self.filled_dataset_descriptors.values()
        }
        if len(global_dimensions) != 1:
            raise ValueError(
                f"Cannot group datasets. Global dimensions '{global_dimensions}' do not match."
            )
        if len(model_dimensions) != 1:
            raise ValueError(
                f"Cannot group datasets. Model dimension '{model_dimensions}' do not match."
            )
        self._index_dependent = any(
            d.index_dependent() for d in self.filled_dataset_descriptors.values()
        )
        self._global_dimension = global_dimensions.pop()
        self._model_dimension = model_dimensions.pop()

    def init_bag(self):
        """Initializes a grouped problem bag."""
        datasets = None
        for label in self._model.dataset:
            dataset = self._data[label]
            if "weight" in dataset:
                weight = dataset.weight
                data = dataset.data * weight
                dataset["weighted_data"] = data
            else:
                weight = xr.DataArray(np.ones_like(dataset.data), coords=dataset.data.coords)
                data = dataset.data
            global_axis = dataset.coords[self._global_dimension].values
            model_axis = dataset.coords[self._model_dimension].values
            has_scaling = self._model.dataset[label].scale is not None
            if self._bag is None:
                self._bag = collections.deque(
                    ProblemGroup(
                        data=data.isel({self._global_dimension: i}).values,
                        weight=weight.isel({self._global_dimension: i}).values,
                        has_scaling=has_scaling,
                        group=label,
                        data_sizes=[data.isel({self._global_dimension: i}).values.size],
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
            data_stripe = data.isel({self._global_dimension: i2[i]}).values
            self._bag[j] = ProblemGroup(
                data=np.concatenate(
                    [
                        self._bag[j].data,
                        data_stripe,
                    ]
                ),
                weight=np.concatenate(
                    [
                        self._bag[j].weight,
                        weight.isel({self._global_dimension: i2[i]}).values,
                    ]
                ),
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
            data_stripe = data.isel({self._global_dimension: i}).values
            problem = ProblemGroup(
                data=data_stripe,
                weight=weight.isel({self._global_dimension: i}).values,
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

    def calculate_matrices(self):
        if self._parameters is None:
            raise ParameterError
        if self._index_dependent:
            self.calculate_index_dependent_matrices()
        else:
            self.calculate_index_independent_matrices()

    def calculate_index_dependent_matrices(
        self,
    ) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray],]:
        """Calculates the index dependent model matrices."""

        def calculate_group(
            group: ProblemGroup, descriptors: dict[str, DatasetDescriptor]
        ) -> tuple[list[xr.DataArray], float]:
            matrices = [
                calculate_matrix(
                    descriptors[problem.label],
                    problem.indices,
                )
                for problem in group.descriptor
            ]
            global_index = group.descriptor[0].indices[self._global_dimension]
            global_index = group.descriptor[0].axis[self._global_dimension][global_index]
            return matrices, global_index

        def reduce_and_combine_matrices(results: tuple[list[xr.DataArray], any]) -> xr.DataArray:
            matrix = xr.concat(results[0], dim=self._model_dimension).fillna(0)
            matrix = reduce_matrix(
                matrix, self.model, self.parameters, self._model_dimension, results[1]
            )
            return matrix

        results = list(
            map(lambda group: calculate_group(group, self._filled_dataset_descriptors), self._bag)
        )

        matrices = list(map(lambda result: result[0], results))

        self._matrices = {}

        for i, grouped_problem in enumerate(self._bag):
            for j, descriptor in enumerate(grouped_problem.descriptor):
                if descriptor.label not in self._matrices:
                    self._matrices[descriptor.label] = []
                self._matrices[descriptor.label].append(matrices[i][j])

        self._reduced_matrices = list(map(reduce_and_combine_matrices, results))
        return self._matrices, self._reduced_matrices

    def calculate_index_independent_matrices(
        self,
    ) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray],]:
        """Calculates the index independent model matrices."""
        self._matrices = {}
        self._reduced_matrices = {}

        for label, dataset_model in self._filled_dataset_descriptors.items():
            self._matrices[label] = calculate_matrix(
                dataset_model,
                {},
            )
            self._reduced_matrices[label] = reduce_matrix(
                self._matrices[label],
                self.model,
                self.parameters,
                self._model_dimension,
                None,
            )

        for group_label, group in self.groups.items():
            if group_label not in self._matrices:
                self._reduced_matrices[group_label] = xr.concat(
                    [self._reduced_matrices[label] for label in group], dim=self._model_dimension
                ).fillna(0)

        return self._matrices, self._reduced_matrices

    def calculate_residual(self):
        if self._index_dependent:
            self.calculate_index_dependent_residual()
        else:
            self.calculate_index_independent_residual()

    def calculate_index_dependent_residual(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray],]:
        """Calculates the index dependent residuals."""

        def residual_function(
            problem: ProblemGroup, matrix: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

            matrix = matrix.copy()
            for i in range(matrix.shape[1]):
                matrix[:, i] *= problem.weight
            data = problem.data
            if problem.has_scaling:
                for i, descriptor in enumerate(problem.descriptor):
                    label = descriptor.label
                    if self.filled_dataset_descriptors[label] is not None:
                        start = sum(problem.data_sizes[0:i])
                        end = start + problem.data_sizes[i]
                        matrix[start:end, :] *= self.filled_dataset_descriptors[label].scale

            clp, residual = self._residual_function(matrix.values, data)
            clp = xr.DataArray(
                clp, dims=["clp_label"], coords={"clp_label": matrix.coords["clp_label"]}
            )
            residual = xr.DataArray(
                residual,
                dims=[self._model_dimension],
                coords={self._model_dimension: matrix.coords[self._model_dimension]},
            )
            return clp, residual, residual / problem.weight

        results = list(map(residual_function, self.bag, self.reduced_matrices))

        self._weighted_residuals = list(map(lambda result: result[1], results))
        self._residuals = list(map(lambda result: result[2], results))

        reduced_clps = list(map(lambda result: result[0], results))
        self._ungroup_clps(reduced_clps)

        return self._reduced_clps, self._clps, self._weighted_residuals, self._residuals

    def calculate_index_independent_residual(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray],]:
        """Calculates the index independent residuals."""

        def residual_function(problem: ProblemGroup):
            matrix = self.reduced_matrices[problem.group].copy()
            for i in range(matrix.shape[1]):
                matrix[:, i] *= problem.weight
            data = problem.data
            if problem.has_scaling:
                for i, descriptor in enumerate(problem.descriptor):
                    label = descriptor.label
                    if self.filled_dataset_descriptors[label] is not None:
                        start = sum(problem.data_sizes[0:i])
                        end = start + problem.data_sizes[i]
                        matrix[start:end, :] *= self.filled_dataset_descriptors[label].scale
            clp, residual = self._residual_function(matrix.values, data)
            clp = xr.DataArray(
                clp, dims=["clp_label"], coords={"clp_label": matrix.coords["clp_label"]}
            )
            residual = xr.DataArray(
                residual,
                dims=[self._model_dimension],
                coords={self._model_dimension: matrix.coords[self._model_dimension]},
            )
            return clp, residual, residual / problem.weight

        results = list(map(residual_function, self.bag))

        self._weighted_residuals = list(map(lambda result: result[1], results))
        self._residuals = list(map(lambda result: result[2], results))

        reduced_clps = list(map(lambda result: result[0], results))
        self._ungroup_clps(reduced_clps)

        return self._reduced_clps, self._clps, self._weighted_residuals, self._residuals

    def _ungroup_clps(self, reduced_clps: list(xr.DataArray)):
        self._reduced_clps = {}
        self._clps = {}
        for label, matrix in self.matrices.items():
            clp_labels = (
                [m.coords["clp_label"] for m in self.matrices[label]]
                if self._index_dependent
                else self.matrices[label].coords["clp_label"]
            )

            # find offset in the full axis
            offset = find_closest_index(
                self.data[label].coords[self._global_dimension][0].values, self._full_axis
            )

            self._reduced_clps[label] = []
            self._clps[label] = []

            for i, index in enumerate(self.data[label].coords[self._global_dimension]):

                index_clp_labels = clp_labels[i] if self._index_dependent else clp_labels
                index_reduced_clps = reduced_clps[i + offset]
                index_reduced_clp_labels, _ = xr.align(
                    index_clp_labels, index_reduced_clps.coords["clp_label"]
                )

                index_reduced_clps = index_reduced_clps.sel(
                    {"clp_label": index_reduced_clp_labels}
                )
                self._reduced_clps[label].append(index_reduced_clps)

                self._clps[label].append(
                    retrieve_clps(
                        self.model,
                        self.parameters,
                        index_clp_labels,
                        index_reduced_clps,
                        index.values,
                    )
                )

            self._reduced_clps[label] = xr.concat(
                self.reduced_clps[label], dim=self._global_dimension
            )
            self._reduced_clps[label].coords[self._global_dimension] = self.data[label].coords[
                self._global_dimension
            ]

            self._clps[label] = xr.concat(self._clps[label], dim=self._global_dimension)
            self._clps[label].coords[self._global_dimension] = self.data[label].coords[
                self._global_dimension
            ]

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
            self.matrices[label],
        )
        dataset["clp"] = self.clps[label]

        return dataset

    def create_index_independent_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:
        """Creates a result datasets for index independent matrices."""

        dataset["matrix"] = self.matrices[label]
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
