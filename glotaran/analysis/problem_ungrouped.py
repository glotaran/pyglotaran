from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.analysis.problem import ParameterError
from glotaran.analysis.problem import Problem
from glotaran.analysis.problem import UngroupedProblemDescriptor
from glotaran.analysis.util import calculate_clp_penalties
from glotaran.analysis.util import calculate_matrix
from glotaran.analysis.util import reduce_matrix
from glotaran.analysis.util import retrieve_clps
from glotaran.model import DatasetDescriptor


class UngroupedProblem(Problem):
    """Represents a problem where the data is not grouped."""

    def init_bag(self):
        """Initializes an ungrouped problem bag."""
        self._bag = {}
        for label, dataset_model in self.dataset_models.items():
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
        dict[str, list[xr.DataArray] | xr.DataArray],
        dict[str, list[xr.DataArray] | xr.DataArray],
    ]:
        """Calculates the model matrices."""
        if self._parameters is None:
            raise ParameterError

        self._matrices = {}
        self._reduced_matrices = {}

        for label, problem in self.bag.items():
            dataset_model = self.dataset_models[label]

            if dataset_model.index_dependent():
                self._calculate_index_dependent_matrix(label, problem, dataset_model)
            else:
                self._calculate_index_independent_matrix(label, problem, dataset_model)

        return self._matrices, self._reduced_matrices

    def _calculate_index_dependent_matrix(
        self, label: str, problem: UngroupedProblemDescriptor, dataset_model: DatasetDescriptor
    ):
        self._matrices[label] = []
        self._reduced_matrices[label] = []
        for i, index in enumerate(problem.global_axis):
            matrix = calculate_matrix(
                dataset_model,
                {dataset_model.get_global_dimension(): i},
            )
            self._matrices[label].append(matrix)
            reduced_matrix = reduce_matrix(
                matrix, self.model, self.parameters, dataset_model.get_model_dimension(), index
            )
            self._reduced_matrices[label].append(reduced_matrix)

    def _calculate_index_independent_matrix(
        self, label: str, problem: UngroupedProblemDescriptor, dataset_model: DatasetDescriptor
    ):

        matrix = calculate_matrix(
            dataset_model,
            {},
        )
        self._matrices[label] = matrix
        reduced_matrix = reduce_matrix(
            matrix, self.model, self.parameters, dataset_model.get_model_dimension(), None
        )
        self._reduced_matrices[label] = reduced_matrix

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
        self._clps = {}
        self._weighted_residuals = {}
        self._residuals = {}
        self._additional_penalty = []

        for label, problem in self.bag.items():
            self._calculate_residual_for_problem(label, problem)

        self._additional_penalty = (
            np.concatenate(self._additional_penalty) if len(self._additional_penalty) != 0 else []
        )
        return self._reduced_clps, self._clps, self._weighted_residuals, self._residuals

    def _calculate_residual_for_problem(self, label: str, problem: UngroupedProblemDescriptor):
        self._reduced_clps[label] = []
        self._clps[label] = []
        self._weighted_residuals[label] = []
        self._residuals[label] = []
        data = problem.data
        dataset_model = self.dataset_models[label]
        model_dimension = dataset_model.get_model_dimension()
        global_dimension = dataset_model.get_global_dimension()

        for i, index in enumerate(problem.global_axis):
            clp_labels = (
                self.matrices[label][i].coords["clp_label"]
                if dataset_model.index_dependent()
                else self.matrices[label].coords["clp_label"]
            )
            reduced_matrix = (
                self.reduced_matrices[label][i]
                if dataset_model.index_dependent()
                else self.reduced_matrices[label]
            )
            if problem.dataset.scale is not None:
                reduced_matrix *= self.dataset_models[label].scale

            if problem.weight is not None:
                for j in range(reduced_matrix.shape[1]):
                    reduced_matrix[:, j] *= problem.weight.isel({global_dimension: i}).values

            reduced_clps, residual = self._residual_function(
                reduced_matrix.values, data.isel({global_dimension: i}).values
            )
            reduced_clps = xr.DataArray(
                reduced_clps,
                dims=["clp_label"],
                coords={"clp_label": reduced_matrix.coords["clp_label"]},
            )
            self._reduced_clps[label].append(reduced_clps)
            self._clps[label].append(
                retrieve_clps(self.model, self.parameters, clp_labels, reduced_clps, index)
            )
            residual = xr.DataArray(
                residual,
                dims=[model_dimension],
                coords={model_dimension: reduced_matrix.coords[model_dimension]},
            )
            self._weighted_residuals[label].append(residual)
            if problem.weight is not None:
                self._residuals[label].append(
                    residual / problem.weight.isel({global_dimension: i})
                )
            else:
                self._residuals[label].append(residual)

        self._reduced_clps[label] = xr.concat(self._reduced_clps[label], dim=global_dimension)
        self._reduced_clps[label].coords[global_dimension] = data.coords[global_dimension]
        self._clps[label] = xr.concat(self._clps[label], dim=global_dimension)
        self._clps[label].coords[global_dimension] = data.coords[global_dimension]
        additional_penalty = calculate_clp_penalties(
            self.model, self.parameters, self._clps[label], global_dimension
        )
        if additional_penalty.size != 0:
            self._additional_penalty.append(additional_penalty)

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
        model_dimension = self.dataset_models[label].get_model_dimension()
        global_dimension = self.dataset_models[label].get_global_dimension()

        matrix = xr.concat(self.matrices[label], dim=global_dimension)
        matrix.coords[global_dimension] = dataset.coords[global_dimension]
        dataset.coords["clp_label"] = matrix.coords["clp_label"]
        dataset["matrix"] = (
            (
                (global_dimension),
                (model_dimension),
                ("clp_label"),
            ),
            matrix.data,
        )

    def _add_index_independent_matrix_to_dataset(self, label: str, dataset: xr.Dataset):
        dataset.coords["clp_label"] = self.matrices[label].coords["clp_label"]
        model_dimension = self.dataset_models[label].get_model_dimension()
        dataset["matrix"] = (
            (
                (model_dimension),
                ("clp_label"),
            ),
            self.matrices[label].data,
        )

    def _add_residual_and_full_clp_to_dataset(self, label: str, dataset: xr.Dataset):
        model_dimension = self.dataset_models[label].get_model_dimension()
        global_dimension = self.dataset_models[label].get_global_dimension()
        dataset["clp"] = self.clps[label]
        dataset["weighted_residual"] = (
            (
                (model_dimension),
                (global_dimension),
            ),
            xr.concat(self.weighted_residuals[label], dim=global_dimension).T.data,
        )
        dataset["residual"] = (
            (
                (model_dimension),
                (global_dimension),
            ),
            xr.concat(self.residuals[label], dim=global_dimension).T.data,
        )

    @property
    def full_penalty(self) -> np.ndarray:
        if self._full_penalty is None:
            residuals = self.weighted_residuals
            additional_penalty = self.additional_penalty
            print(residuals)
            residuals = [np.concatenate(residuals[label]) for label in residuals.keys()]

            self._full_penalty = (
                np.concatenate((np.concatenate(residuals), additional_penalty))
                if additional_penalty is not None
                else np.concatenate(residuals)
            )
        return self._full_penalty
