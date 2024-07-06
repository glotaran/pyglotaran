from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model.data_model import iterate_data_model_elements
from glotaran.model.data_model import iterate_data_model_global_elements
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.penalty import calculate_clp_penalties
from glotaran.parameter.parameter import Parameter

if TYPE_CHECKING:
    from glotaran.model.experiment_model import ExperimentModel
    from glotaran.typing.types import ArrayLike


@dataclass
class OptimizationObjectiveResult:
    data: dict[str, xr.Dataset]
    free_clp_size: int


class OptimizationObjective:
    def __init__(self, model: ExperimentModel):
        self._data = (
            LinkedOptimizationData.from_experiment_model(model)
            if len(model.datasets) > 1
            else OptimizationData(next(iter(model.datasets.values())))
        )
        self._model = model

    def calculate_matrices(self) -> list[OptimizationMatrix]:
        if isinstance(self._data, OptimizationData):
            return OptimizationMatrix.from_data(self._data).as_global_list(self._data.global_axis)
        return OptimizationMatrix.from_linked_data(self._data)  # type:ignore[arg-type]

    def calculate_reduced_matrices(
        self, matrices: list[OptimizationMatrix]
    ) -> list[OptimizationMatrix]:
        return [
            matrices[i].reduce(
                index, self._model.clp_constraints, self._model.clp_relations, copy=True
            )
            for i, index in enumerate(self._data.global_axis)
        ]

    def calculate_estimations(
        self, reduced_matrices: list[OptimizationMatrix]
    ) -> list[OptimizationEstimation]:
        return [
            OptimizationEstimation.calculate(matrix.array, data, self._model.residual_function)
            for matrix, data in zip(reduced_matrices, self._data.data_slices)
        ]

    def resolve_estimations(
        self,
        matrices: list[OptimizationMatrix],
        reduced_matrices: list[OptimizationMatrix],
        estimations: list[OptimizationEstimation],
    ) -> list[OptimizationEstimation]:
        return [
            e.resolve_clp(m.clp_axis, r.clp_axis, i, self._model.clp_relations)
            for e, m, r, i in zip(estimations, matrices, reduced_matrices, self._data.global_axis)
        ]

    def calculate_global_penalty(self) -> ArrayLike:
        _, _, matrix = OptimizationMatrix.from_global_data(self._data)  # type:ignore[arg-type]
        return OptimizationEstimation.calculate(
            matrix.array,
            self._data.flat_data,  # type:ignore[attr-defined]
            self._model.residual_function,
        ).residual

    def calculate(self) -> ArrayLike:
        if isinstance(self._data, OptimizationData) and self._data.is_global:
            return self.calculate_global_penalty()
        matrices = self.calculate_matrices()
        reduced_matrices = self.calculate_reduced_matrices(matrices)
        estimations = self.calculate_estimations(reduced_matrices)

        penalties = [e.residual for e in estimations]
        if len(self._model.clp_penalties) > 0:
            estimations = self.resolve_estimations(matrices, reduced_matrices, estimations)
            penalties.append(
                calculate_clp_penalties(
                    matrices,
                    estimations,
                    self._data.global_axis,
                    self._model.clp_penalties,
                )
            )
        return np.concatenate(penalties)

    def get_result(self) -> OptimizationObjectiveResult:
        return (
            self.create_unlinked_result()
            if isinstance(self._data, OptimizationData)
            else self.create_linked_result()
        )

    def create_result_dataset(self, label: str, data: OptimizationData) -> xr.Dataset:
        assert isinstance(data.model.data, xr.Dataset)
        dataset = data.model.data.copy()
        if dataset.data.dims != (data.model_dimension, data.global_dimension):
            dataset["data"] = dataset.data.T
        dataset.attrs["model_dimension"] = data.model_dimension
        dataset.attrs["global_dimension"] = data.global_dimension
        dataset.coords[data.model_dimension] = data.model_axis
        dataset.coords[data.global_dimension] = data.global_axis
        if isinstance(self._data, LinkedOptimizationData):
            scale = self._data.scales[label]
            dataset.attrs["scale"] = scale.value if isinstance(scale, Parameter) else scale
        return dataset

    def add_matrix_to_dataset(self, dataset: xr.Dataset, matrix: OptimizationMatrix):
        dataset.coords["clp_label"] = matrix.clp_axis
        matrix_dims = (dataset.attrs["model_dimension"], "clp_label")
        if matrix.is_index_dependent:
            matrix_dims = (  # type:ignore[assignment]
                dataset.attrs["global_dimension"],
                *matrix_dims,
            )
        dataset["matrix"] = xr.DataArray(matrix.array, dims=matrix_dims)

    def add_linked_clp_and_residual_to_dataset(
        self,
        dataset: xr.Dataset,
        label: str,
        clp_axes: list[list[str]],
        estimations: list[OptimizationEstimation],
    ):
        assert isinstance(self._data, LinkedOptimizationData)
        global_indices = [
            i
            for i, group_label in enumerate(self._data.group_labels)
            if label in self._data.group_definitions[group_label]
        ]

        clp_dims = (dataset.attrs["global_dimension"], "clp_label")
        dataset["clp"] = xr.DataArray(
            [
                [
                    estimations[i].clp[clp_axes[i].index(clp_label)]
                    for clp_label in dataset.coords["clp_label"]
                ]
                for i in global_indices
            ],
            dims=clp_dims,
        )

        offsets = []
        for i in global_indices:
            group_label = self._data._group_labels[i]
            group_index = self._data.group_definitions[group_label].index(label)
            offsets.append(sum(self._data.group_sizes[group_label][:group_index]))
        size = dataset.coords[dataset.attrs["model_dimension"]].size
        residual_dims = (
            dataset.attrs["global_dimension"],
            dataset.attrs["model_dimension"],
        )
        dataset["residual"] = xr.DataArray(
            [
                estimations[i].residual[offset : offset + size]
                for i, offset in zip(global_indices, offsets)
            ],
            dims=residual_dims,
        ).T

    def add_unlinked_clp_and_residual_to_dataset(
        self,
        dataset: xr.Dataset,
        estimations: list[OptimizationEstimation],
    ):
        clp_dims = (dataset.attrs["global_dimension"], "clp_label")
        dataset["clp"] = xr.DataArray([e.clp for e in estimations], dims=clp_dims)

        residual_dims = (
            dataset.attrs["global_dimension"],
            dataset.attrs["model_dimension"],
        )
        dataset["residual"] = xr.DataArray([e.residual for e in estimations], dims=residual_dims).T

    def add_global_clp_and_residual_to_dataset(
        self,
        dataset: xr.Dataset,
        data: OptimizationData,
        matrix: OptimizationMatrix,
    ):
        global_matrix = OptimizationMatrix.from_data(data, global_matrix=True)
        global_matrix_coords = (
            (data.global_dimension, data.global_axis),
            ("global_clp_label", matrix.clp_axis),
        )
        if global_matrix.is_index_dependent:
            global_matrix_coords = (  # type:ignore[assignment]
                (data.model_dimension, data.model_axis),
                *global_matrix_coords,
            )
        dataset["global_matrix"] = xr.DataArray(global_matrix.array, coords=global_matrix_coords)
        _, _, full_matrix = OptimizationMatrix.from_global_data(data)
        estimation = OptimizationEstimation.calculate(
            full_matrix.array,
            data.flat_data,  # type:ignore[arg-type]
            data.model.residual_function,
        )
        dataset["clp"] = xr.DataArray(
            estimation.clp.reshape((len(global_matrix.clp_axis), len(matrix.clp_axis))),
            coords={
                "global_clp_label": global_matrix.clp_axis,
                "clp_label": matrix.clp_axis,
            },
            dims=["global_clp_label", "clp_label"],
        )
        dataset["residual"] = xr.DataArray(
            estimation.residual.reshape(
                data.global_axis.size,
                data.model_axis.size,
            ),
            coords=(
                (data.global_dimension, data.global_axis),
                (data.model_dimension, data.model_axis),
            ),
        ).T

    def finalize_result_dataset(self, dataset: xr.Dataset, data: OptimizationData, add_svd=True):
        # Calculate RMS
        size = dataset.residual.shape[0] * dataset.residual.shape[1]
        dataset.attrs["root_mean_square_error"] = np.sqrt((dataset.residual**2).sum() / size).data
        dataset["fitted_data"] = dataset.data - dataset.residual

        if data.weight is not None:
            weight = data.weight
            if data.is_global:
                dataset["global_weighted_matrix"] = dataset["global_matrix"]
                dataset["global_matrix"] = dataset["global_matrix"] / weight[..., np.newaxis]
            if "weight" not in dataset:
                dataset["weight"] = xr.DataArray(data.weight, coords=dataset.data.coords)
            dataset["weighted_residual"] = dataset["residual"]
            dataset["residual"] = dataset["residual"] / weight
            dataset["weighted_matrix"] = dataset["matrix"]
            dataset["matrix"] = dataset["matrix"] / weight.T[..., np.newaxis]
            dataset.attrs["weighted_root_mean_square_error"] = dataset.attrs[
                "root_mean_square_error"
            ]
            dataset.attrs["root_mean_square_error"] = np.sqrt(
                (dataset.residual**2).sum() / size
            ).data

        if add_svd:
            for name in ["data", "residual"]:
                if f"{name}_singular_values" in dataset:
                    continue
                lsv, sv, rsv = np.linalg.svd(dataset[name], full_matrices=False)
                dataset[f"{name}_left_singular_vectors"] = (
                    (data.model_dimension, "left_singular_value_index"),
                    lsv,
                )
                dataset[f"{name}_singular_values"] = (("singular_value_index"), sv)
                dataset[f"{name}_right_singular_vectors"] = (
                    (data.global_dimension, "right_singular_value_index"),
                    rsv.T,
                )
        for _, model in iterate_data_model_elements(data.model):
            model.add_to_result_data(  # type:ignore[union-attr]
                data.model, dataset, False
            )
        for _, model in iterate_data_model_global_elements(data.model):
            model.add_to_result_data(  # type:ignore[union-attr]
                data.model, dataset, True
            )

    def create_linked_result(self) -> OptimizationObjectiveResult:
        assert isinstance(self._data, LinkedOptimizationData)
        matrices = {
            label: OptimizationMatrix.from_data(data) for label, data in self._data.data.items()
        }
        linked_matrices = OptimizationMatrix.from_linked_data(self._data, matrices)
        clp_axes = [matrix.clp_axis for matrix in linked_matrices]
        reduced_matrices = self.calculate_reduced_matrices(linked_matrices)
        free_clp_size = sum(len(matrix.clp_axis) for matrix in reduced_matrices)
        estimations = self.resolve_estimations(
            linked_matrices,
            reduced_matrices,
            self.calculate_estimations(reduced_matrices),
        )

        results = {}
        for label, matrix in matrices.items():
            data = self._data.data[label]
            results[label] = self.create_result_dataset(label, data)
            self.add_matrix_to_dataset(results[label], matrix)
            self.add_linked_clp_and_residual_to_dataset(
                results[label], label, clp_axes, estimations
            )
            self.finalize_result_dataset(results[label], data)
        return OptimizationObjectiveResult(results, free_clp_size)

    def create_unlinked_result(self) -> OptimizationObjectiveResult:
        assert isinstance(self._data, OptimizationData)

        label = next(iter(self._model.datasets.keys()))
        result = self.create_result_dataset(label, self._data)

        matrix = OptimizationMatrix.from_data(self._data)
        self.add_matrix_to_dataset(result, matrix)
        if self._data.is_global:
            self.add_global_clp_and_residual_to_dataset(result, self._data, matrix)
            free_clp_size = len(matrix.clp_axis)

        else:
            reduced_matrices = self.calculate_reduced_matrices(
                matrix.as_global_list(self._data.global_axis)
            )
            free_clp_size = sum(len(matrix.clp_axis) for matrix in reduced_matrices)
            estimations = self.resolve_estimations(
                matrix.as_global_list(self._data.global_axis),
                reduced_matrices,
                self.calculate_estimations(reduced_matrices),
            )
            self.add_unlinked_clp_and_residual_to_dataset(result, estimations)

        self.finalize_result_dataset(result, self._data)
        return OptimizationObjectiveResult({label: result}, free_clp_size)
