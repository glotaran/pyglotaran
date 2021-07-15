from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.analysis.problem import ParameterError
from glotaran.analysis.problem import Problem
from glotaran.analysis.util import CalculatedMatrix
from glotaran.analysis.util import apply_weight
from glotaran.analysis.util import calculate_clp_penalties
from glotaran.analysis.util import calculate_matrix
from glotaran.analysis.util import reduce_matrix
from glotaran.analysis.util import retrieve_clps
from glotaran.model import DatasetModel


class UngroupedProblem(Problem):
    """Represents a problem where the data is not grouped."""

    def calculate_matrices(
        self,
    ) -> tuple[
        dict[str, CalculatedMatrix | list[CalculatedMatrix]],
        dict[str, CalculatedMatrix | list[CalculatedMatrix]],
    ]:
        """Calculates the model matrices."""
        if self._parameters is None:
            raise ParameterError

        self._matrices = {}
        self._reduced_matrices = {}

        for label, dataset_model in self.dataset_models.items():

            if dataset_model.index_dependent():
                self._calculate_index_dependent_matrix(label, dataset_model)
            else:
                self._calculate_index_independent_matrix(label, dataset_model)

        return self._matrices, self._reduced_matrices

    def _calculate_index_dependent_matrix(self, label: str, dataset_model: DatasetModel):
        self._matrices[label] = []
        self._reduced_matrices[label] = []
        for i, index in enumerate(dataset_model.get_global_axis()):
            matrix = calculate_matrix(
                dataset_model,
                {dataset_model.get_global_dimension(): i},
            )
            self._matrices[label].append(matrix)
            reduced_matrix = reduce_matrix(matrix, self.model, self.parameters, index)
            self._reduced_matrices[label].append(reduced_matrix)

    def _calculate_index_independent_matrix(self, label: str, dataset_model: DatasetModel):

        matrix = calculate_matrix(dataset_model, {})
        self._matrices[label] = matrix
        reduced_matrix = reduce_matrix(matrix, self.model, self.parameters, None)
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

        for label, dataset_model in self._dataset_models.items():
            self._calculate_residual(label, dataset_model)

        self._additional_penalty = (
            np.concatenate(self._additional_penalty) if len(self._additional_penalty) != 0 else []
        )
        return self._reduced_clps, self._clps, self._weighted_residuals, self._residuals

    def _calculate_residual(self, label: str, dataset_model: DatasetModel):
        self._reduced_clps[label] = []
        self._clps[label] = []
        self._weighted_residuals[label] = []
        self._residuals[label] = []

        data = dataset_model.get_data()
        global_axis = dataset_model.get_global_axis()

        for i, index in enumerate(global_axis):
            reduced_clp_labels, reduced_matrix = (
                self.reduced_matrices[label][i]
                if dataset_model.index_dependent()
                else self.reduced_matrices[label]
            )
            if not dataset_model.index_dependent():
                reduced_matrix = reduced_matrix.copy()

            if dataset_model.scale is not None:
                reduced_matrix *= dataset_model.scale

            weight = dataset_model.get_weight()
            if weight is not None:
                apply_weight(reduced_matrix, weight[:, i])

            reduced_clps, residual = self._residual_function(reduced_matrix, data[:, i])

            self._reduced_clps[label].append(reduced_clps)

            clp_labels = self._get_clp_labels(label, i)
            self._clps[label].append(
                retrieve_clps(
                    self.model,
                    self.parameters,
                    clp_labels,
                    reduced_clp_labels,
                    reduced_clps,
                    index,
                )
            )
            self._weighted_residuals[label].append(residual)
            if weight is not None:
                self._residuals[label].append(residual / weight[:, i])
            else:
                self._residuals[label].append(residual)

        clp_labels = self._get_clp_labels(label)
        additional_penalty = calculate_clp_penalties(
            self.model, self.parameters, clp_labels, self._clps[label], global_axis
        )
        if additional_penalty.size != 0:
            self._additional_penalty.append(additional_penalty)

    def _get_clp_labels(self, label: str, index: int = 0):
        return (
            self.matrices[label][index].clp_labels
            if self.dataset_models[label].index_dependent()
            else self.matrices[label].clp_labels
        )

    def create_index_dependent_result_dataset(self, label: str, dataset: xr.Dataset) -> xr.Dataset:
        """Creates a result datasets for index dependent matrices."""

        model_dimension = self.dataset_models[label].get_model_dimension()
        global_dimension = self.dataset_models[label].get_global_dimension()

        dataset.coords["clp_label"] = self._get_clp_labels(label)
        dataset["matrix"] = (
            (
                (global_dimension),
                (model_dimension),
                ("clp_label"),
            ),
            np.asarray([m.matrix for m in self.matrices[label]]),
        )

        self._add_residual_and_full_clp_to_dataset(label, dataset)

        return dataset

    def create_index_independent_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:
        """Creates a result datasets for index independent matrices."""

        matrix = self.matrices[label]
        dataset.coords["clp_label"] = matrix.clp_labels
        model_dimension = self.dataset_models[label].get_model_dimension()
        dataset["matrix"] = (
            (
                (model_dimension),
                ("clp_label"),
            ),
            matrix.matrix,
        )

        self._add_residual_and_full_clp_to_dataset(label, dataset)

        return dataset

    def _add_residual_and_full_clp_to_dataset(self, label: str, dataset: xr.Dataset):
        model_dimension = self.dataset_models[label].get_model_dimension()
        global_dimension = self.dataset_models[label].get_global_dimension()
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

    @property
    def full_penalty(self) -> np.ndarray:
        if self._full_penalty is None:
            residuals = self.weighted_residuals
            additional_penalty = self.additional_penalty
            residuals = [np.concatenate(residuals[label]) for label in residuals.keys()]

            self._full_penalty = (
                np.concatenate((np.concatenate(residuals), additional_penalty))
                if additional_penalty is not None
                else np.concatenate(residuals)
            )
        return self._full_penalty
