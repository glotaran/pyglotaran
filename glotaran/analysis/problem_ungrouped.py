from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.analysis.problem import ParameterError
from glotaran.analysis.problem import Problem
from glotaran.analysis.problem import UngroupedProblemDescriptor
from glotaran.analysis.util import calculate_matrix
from glotaran.analysis.util import reduce_matrix
from glotaran.model import DatasetDescriptor


class UngroupedProblem(Problem):
    """Represents a problem where the data is not grouped."""

    def init_bag(self):
        """Initializes an ungrouped problem bag."""
        self._bag = {}
        for label, dataset_model in self.filled_dataset_descriptors.items():
            dataset = self._data[label]
            data = dataset.data
            weight = dataset.weight if "weight" in dataset else None
            if weight is not None:
                data = data * weight
                dataset["weighted_data"] = data
            self._bag[label] = UngroupedProblemDescriptor(
                dataset_model,
                data,
                dataset.coords[dataset_model.get_model_dimension()].values,
                dataset.coords[dataset_model.get_global_dimension()].values,
                weight,
            )

    def calculate_matrices(
        self,
    ) -> tuple[
        dict[str, list[list[str]] | list[str]],
        dict[str, list[np.ndarray]] | np.ndarray,
        dict[str, list[str]],
        dict[str, list[np.ndarray] | np.ndarray],
    ]:
        """Calculates the model matrices."""
        if self._parameters is None:
            raise ParameterError

        self._clp_labels = {}
        self._matrices = {}
        self._reduced_clp_labels = {}
        self._reduced_matrices = {}

        for label, problem in self.bag.items():
            self._clp_labels[label] = []
            self._matrices[label] = []
            self._reduced_clp_labels[label] = []
            self._reduced_matrices[label] = []
            dataset_model = self._filled_dataset_descriptors[label]

            if dataset_model.index_dependent():
                self._calculate_index_dependent_matrix(label, problem, dataset_model)
            else:
                self._calculate_index_independent_matrix(label, problem, dataset_model)

        return self._clp_labels, self._matrices, self._reduced_clp_labels, self._reduced_matrices

    def _calculate_index_dependent_matrix(
        self, label: str, problem: UngroupedProblemDescriptor, dataset_model: DatasetDescriptor
    ):
        for i, index in enumerate(problem.global_axis):
            result = calculate_matrix(
                self._model,
                dataset_model,
                {dataset_model.get_global_dimension(): i},
                {
                    dataset_model.get_model_dimension(): problem.model_axis,
                    dataset_model.get_global_dimension(): problem.global_axis,
                },
            )

            self._clp_labels[label].append(result.clp_label)
            self._matrices[label].append(result.matrix)
            reduced_labels_and_matrix = reduce_matrix(
                self._model, label, self._parameters, result, index
            )
            self._reduced_clp_labels[label].append(reduced_labels_and_matrix.clp_label)
            self._reduced_matrices[label].append(reduced_labels_and_matrix.matrix)

    def _calculate_index_independent_matrix(
        self, label: str, problem: UngroupedProblemDescriptor, dataset_model: DatasetDescriptor
    ):

        model_dimension = dataset_model.get_model_dimension()
        global_dimension = dataset_model.get_global_dimension()
        result = calculate_matrix(
            self._model,
            dataset_model,
            {},
            {
                model_dimension: problem.model_axis,
                global_dimension: problem.global_axis,
            },
        )

        self._clp_labels[label] = result.clp_label
        self._matrices[label] = result.matrix
        reduced_result = reduce_matrix(self._model, label, self._parameters, result, None)
        self._reduced_clp_labels[label] = reduced_result.clp_label
        self._reduced_matrices[label] = reduced_result.matrix

    def calculate_residual(
        self,
    ) -> tuple[
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
    ]:
        """Calculates the residuals."""

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

    def _calculate_residual_for_problem(self, label: str, problem: UngroupedProblemDescriptor):
        self._reduced_clps[label] = []
        self._weighted_residuals[label] = []
        self._residuals[label] = []
        data = problem.data
        dataset_model = self._filled_dataset_descriptors[label]
        global_dimension = dataset_model.get_global_dimension()

        for i in range(len(problem.global_axis)):
            matrix = (
                self.reduced_matrices[label][i]
                if dataset_model.index_dependent()
                else self.reduced_matrices[label].copy()
            )  # TODO: .copy() or not
            if problem.dataset.scale is not None:
                matrix *= self.filled_dataset_descriptors[label].scale

            if problem.weight is not None:
                for j in range(matrix.shape[1]):
                    matrix[:, j] *= problem.weight.isel({global_dimension: i}).values

            clp, residual = self._residual_function(
                matrix, data.isel({global_dimension: i}).values
            )
            self._reduced_clps[label].append(clp)
            self._weighted_residuals[label].append(residual)
            if problem.weight is not None:
                self._residuals[label].append(
                    residual / problem.weight.isel({global_dimension: i})
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
        model_dimension = self.filled_dataset_descriptors[label].get_model_dimension()
        global_dimension = self.filled_dataset_descriptors[label].get_global_dimension()

        dataset["matrix"] = (
            (
                (global_dimension),
                (model_dimension),
                ("clp_label"),
            ),
            np.asarray(self.matrices[label]),
        )

    def _add_index_independent_matrix_to_dataset(self, label: str, dataset: xr.Dataset):
        dataset.coords["clp_label"] = self.clp_labels[label]
        model_dimension = self.filled_dataset_descriptors[label].get_model_dimension()
        dataset["matrix"] = (
            (
                (model_dimension),
                ("clp_label"),
            ),
            np.asarray(self.matrices[label]),
        )

    def _add_residual_and_full_clp_to_dataset(self, label: str, dataset: xr.Dataset):
        model_dimension = self.filled_dataset_descriptors[label].get_model_dimension()
        global_dimension = self.filled_dataset_descriptors[label].get_global_dimension()
        dataset["clp"] = (
            (
                (global_dimension),
                ("clp_label"),
            ),
            np.asarray(self.clps[label]),
        )
        dataset["weighted_residual"] = (
            (
                (model_dimension),
                (global_dimension),
            ),
            np.transpose(np.asarray(self.weighted_residuals[label])),
        )
        dataset["residual"] = (
            (
                (model_dimension),
                (global_dimension),
            ),
            np.transpose(np.asarray(self.residuals[label])),
        )
