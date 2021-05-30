from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.analysis.problem import Problem
from glotaran.analysis.problem import UngroupedProblemDescriptor
from glotaran.analysis.util import calculate_matrix
from glotaran.analysis.util import reduce_matrix


class UngroupedProblem(Problem):
    """Represents a problem where the data is not grouped."""

    def init_bag(self):
        """Initializes an ungrouped problem bag."""
        self._bag = {}
        for label in self._scheme.model.dataset:
            dataset = self._data[label]
            data = dataset.data
            weight = dataset.weight if "weight" in dataset else None
            if weight is not None:
                data = data * weight
                dataset["weighted_data"] = data
            self._bag[label] = UngroupedProblemDescriptor(
                self._scheme.model.dataset[label],
                data,
                dataset.coords[self._model_dimension].values,
                dataset.coords[self._global_dimension].values,
                weight,
            )

    def calculate_index_dependent_matrices(
        self,
    ) -> tuple[
        dict[str, list[list[str]]],
        dict[str, list[np.ndarray]],
        dict[str, list[str]],
        dict[str, list[np.ndarray]],
    ]:
        """Calculates the index dependent model matrices."""

        self._clp_labels = {}
        self._matrices = {}
        self._reduced_clp_labels = {}
        self._reduced_matrices = {}

        for label, problem in self.bag.items():
            self._clp_labels[label] = []
            self._matrices[label] = []
            self._reduced_clp_labels[label] = []
            self._reduced_matrices[label] = []
            descriptor = self._filled_dataset_descriptors[label]

            for i, index in enumerate(problem.global_axis):
                result = calculate_matrix(
                    self._model,
                    descriptor,
                    {self._global_dimension: i},
                    {
                        self._model_dimension: problem.model_axis,
                        self._global_dimension: problem.global_axis,
                    },
                )

                self._clp_labels[label].append(result.clp_label)
                self._matrices[label].append(result.matrix)
                reduced_labels_and_matrix = reduce_matrix(
                    self._model, label, self._parameters, result, index
                )
                self._reduced_clp_labels[label].append(reduced_labels_and_matrix.clp_label)
                self._reduced_matrices[label].append(reduced_labels_and_matrix.matrix)

        return self._clp_labels, self._matrices, self._reduced_clp_labels, self._reduced_matrices

    def calculate_index_independent_matrices(
        self,
    ) -> tuple[
        dict[str, list[str]],
        dict[str, np.ndarray],
        dict[str, list[str]],
        dict[str, np.ndarray],
    ]:
        """Calculates the index independent model matrices."""

        self._clp_labels = {}
        self._matrices = {}
        self._reduced_clp_labels = {}
        self._reduced_matrices = {}

        for label, descriptor in self._filled_dataset_descriptors.items():
            model_axis = self._data[label].coords[self._model_dimension].values
            global_axis = self._data[label].coords[self._global_dimension].values
            result = calculate_matrix(
                self._model,
                descriptor,
                {},
                {
                    self._model_dimension: model_axis,
                    self._global_dimension: global_axis,
                },
            )

            self._clp_labels[label] = result.clp_label
            self._matrices[label] = result.matrix
            reduced_result = reduce_matrix(self._model, label, self._parameters, result, None)
            self._reduced_clp_labels[label] = reduced_result.clp_label
            self._reduced_matrices[label] = reduced_result.matrix

        return self._clp_labels, self._matrices, self._reduced_clp_labels, self._reduced_matrices

    def calculate_index_dependent_residual(
        self,
    ) -> tuple[
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
    ]:
        """Calculates the index dependent residuals."""

        self._reduced_clps = {}
        self._weighted_residuals = {}
        self._residuals = {}

        for label, problem in self.bag.items():
            self._calculate_residual_for_problem(label, problem)

        self._clps = (
            self.model.retrieve_clp_function(
                self.parameters,
                self.clp_labels,
                self.reduced_clp_labels,
                self.reduced_clps,
                self.data,
            )
            if callable(self.model.retrieve_clp_function)
            else self.reduced_clps
        )

        return self._reduced_clps, self._clps, self._weighted_residuals, self._residuals

    def calculate_index_independent_residual(
        self,
    ) -> tuple[
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
    ]:
        """Calculates the index independent residuals."""
        return self.calculate_index_dependent_residual()

    def _calculate_residual_for_problem(self, label: str, problem: UngroupedProblemDescriptor):
        self._reduced_clps[label] = []
        self._weighted_residuals[label] = []
        self._residuals[label] = []
        data = problem.data

        for i in range(len(problem.global_axis)):
            matrix = (
                self.reduced_matrices[label][i]
                if self.index_dependent
                else self.reduced_matrices[label].copy()
            )  # TODO: .copy() or not
            if problem.dataset.scale is not None:
                matrix *= self.filled_dataset_descriptors[label].scale

            if problem.weight is not None:
                for j in range(matrix.shape[1]):
                    matrix[:, j] *= problem.weight.isel({self._global_dimension: i}).values

            clp, residual = self._residual_function(
                matrix, data.isel({self._global_dimension: i}).values
            )
            self._reduced_clps[label].append(clp)
            self._weighted_residuals[label].append(residual)
            if problem.weight is not None:
                self._residuals[label].append(
                    residual / problem.weight.isel({self._global_dimension: i})
                )
            else:
                self._residuals[label].append(residual)

    def create_index_dependent_result_dataset(self, label: str, dataset: xr.Dataset) -> xr.Dataset:
        """Creates a result datasets for index dependent matrices."""

        self._add_index_dependent_matrix_to_dataset(label, dataset)

        self._add_residual_and_full_clp_to_dataset(label, dataset)

        return dataset

    def create_index_independent_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:
        """Creates a result datasets for index independent matrices."""

        self._add_index_independent_matrix_to_dataset(label, dataset)

        self._add_residual_and_full_clp_to_dataset(label, dataset)

        return dataset

    def _add_index_dependent_matrix_to_dataset(self, label: str, dataset: xr.Dataset):
        # we assume that the labels are the same, this might not be true in
        # future models
        dataset.coords["clp_label"] = self.clp_labels[label][0]

        dataset["matrix"] = (
            (
                (self._global_dimension),
                (self._model_dimension),
                ("clp_label"),
            ),
            np.asarray(self.matrices[label]),
        )

    def _add_index_independent_matrix_to_dataset(self, label: str, dataset: xr.Dataset):
        dataset.coords["clp_label"] = self.clp_labels[label]
        dataset["matrix"] = (
            (
                (self._model_dimension),
                ("clp_label"),
            ),
            np.asarray(self.matrices[label]),
        )

    def _add_residual_and_full_clp_to_dataset(self, label: str, dataset: xr.Dataset):
        dataset["clp"] = (
            (
                (self._global_dimension),
                ("clp_label"),
            ),
            np.asarray(self.clps[label]),
        )
        dataset["weighted_residual"] = (
            (
                (self._model_dimension),
                (self._global_dimension),
            ),
            np.transpose(np.asarray(self.weighted_residuals[label])),
        )
        dataset["residual"] = (
            (
                (self._model_dimension),
                (self._global_dimension),
            ),
            np.transpose(np.asarray(self.residuals[label])),
        )
