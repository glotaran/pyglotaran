from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model.data_model import DataModel
from glotaran.model.data_model import iterate_data_model_elements
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.penalty import calculate_clp_penalties
from glotaran.parameter.parameter import Parameter

if TYPE_CHECKING:
    from glotaran.model.element import ElementResult
    from glotaran.model.experiment_model import ExperimentModel
    from glotaran.typing.types import ArrayLike


def add_svd_to_result_dataset(dataset: xr.Dataset, global_dim: str, model_dim: str):
    for name in ["data", "residual"]:
        if f"{name}_singular_values" in dataset:
            continue
        lsv, sv, rsv = np.linalg.svd(dataset[name], full_matrices=False)
        dataset[f"{name}_left_singular_vectors"] = (
            (model_dim, "left_singular_value_index"),
            lsv,
        )
        dataset[f"{name}_singular_values"] = (("singular_value_index"), sv)
        dataset[f"{name}_right_singular_vectors"] = (
            (global_dim, "right_singular_value_index"),
            rsv.T,
        )


@dataclass
class OptimizationObjectiveResult:
    data: dict[str, xr.Dataset]
    additional_penalty: float
    clp_size: int


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

    def get_global_indices(self, label: str) -> list[str]:
        return [
            i
            for i, group_label in enumerate(self._data.group_labels)
            if label in self._data.group_definitions[group_label]
        ]

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

    def add_global_clp_and_residual_to_dataset(
        self,
        dataset: xr.Dataset,
        data: OptimizationData,
        matrix: OptimizationMatrix,
    ):
        dataset["matrix"] = matrix.to_data_array(
            data.global_dimension, data.global_axis, data.model_dimension, data.model_axis
        )
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
        dataset.attrs["root_mean_square_error"] = np.sqrt(
            (dataset.residual.to_numpy() ** 2).sum() / sum(dataset.residual.shape)
        )

    def create_single_dataset_result(self) -> OptimizationObjectiveResult:
        assert isinstance(self._data, OptimizationData)

        label = next(iter(self._model.datasets.keys()))
        result_dataset = self.create_result_dataset(label, self._data)

        global_dim = result_dataset.attrs["global_dimension"]
        global_axis = result_dataset.coords[global_dim]
        model_dim = result_dataset.attrs["model_dimension"]
        model_axis = result_dataset.coords[model_dim]

        concentrations = OptimizationMatrix.from_data(self._data)
        additional_penalty = 0
        if self._data.is_global:
            self.add_global_clp_and_residual_to_dataset(result_dataset, self._data, concentrations)
            clp_size = len(concentrations.clp_axis)

        else:
            clp_concentration = self.calculate_reduced_matrices(
                concentrations.as_global_list(self._data.global_axis)
            )
            clp_size = sum(len(c.clp_axis) for c in clp_concentration)
            estimations = self.resolve_estimations(
                concentrations.as_global_list(self._data.global_axis),
                clp_concentration,
                self.calculate_estimations(clp_concentration),
            )
            amplitude_coords = {
                global_dim: global_axis,
                "amplitude_label": concentrations.clp_axis,
            }
            amplitude = xr.DataArray(
                [e.clp for e in estimations], dims=amplitude_coords.keys(), coords=amplitude_coords
            )
            concentration = concentrations.to_data_array(
                global_dim, global_axis, model_dim, model_axis
            )

            residual_dims = (global_dim, model_dim)
            result_dataset["residual"] = xr.DataArray(
                [e.residual for e in estimations], dims=residual_dims
            ).T
            result_dataset.attrs["root_mean_square_error"] = np.sqrt(
                (result_dataset.residual.to_numpy() ** 2).sum()
                / sum(result_dataset.residual.shape)
            )
            additional_penalty = sum(
                calculate_clp_penalties(
                    [concentrations],
                    estimations,
                    global_axis,
                    self._model.clp_penalties,
                )
            )
            self.add_element_results(
                result_dataset, label, global_dim, model_dim, amplitude, concentration
            )
            self.add_data_model_results(
                label, result_dataset, global_dim, model_dim, amplitude, concentration
            )

        self._data.unweight_result_dataset(result_dataset)
        result_dataset["fit"] = result_dataset.data - result_dataset.residual
        add_svd_to_result_dataset(result_dataset, global_dim, model_dim)
        return OptimizationObjectiveResult({label: result_dataset}, clp_size, additional_penalty)

    def create_multi_dataset_result(self) -> dict[str, xr.Dataset]:
        assert isinstance(self._data, LinkedOptimizationData)
        dataset_concentrations = {
            label: OptimizationMatrix.from_data(data) for label, data in self._data.data.items()
        }
        full_concentration = OptimizationMatrix.from_linked_data(
            self._data, dataset_concentrations
        )
        estimated_amplitude_axes = [concentration.clp_axis for concentration in full_concentration]
        clp_concentration = self.calculate_reduced_matrices(full_concentration)
        clp_size = sum(len(concentration.clp_axis) for concentration in clp_concentration)

        estimations = self.resolve_estimations(
            full_concentration,
            clp_concentration,
            self.calculate_estimations(clp_concentration),
        )
        additional_penalty = sum(
            calculate_clp_penalties(
                full_concentration,
                estimations,
                self._data.global_axis,
                self._model.clp_penalties,
            )
        )

        results = {
            label: self.create_dataset_result(
                label,
                data,
                dataset_concentrations[label],
                estimated_amplitude_axes,
                estimations,
            )
            for label, data in self._data.data.items()
        }
        return OptimizationObjectiveResult(
            results,
            clp_size,
            additional_penalty,
        )

    def get_dataset_amplitudes(
        self,
        label: str,
        estimated_amplitude_axes: list[list[str]],
        estimated_amplitudes: list[OptimizationEstimation],
        amplitude_axis: ArrayLike,
        global_dim: str,
        global_axis: ArrayLike,
    ) -> xr.DataArray:
        assert isinstance(self._data, LinkedOptimizationData)

        global_indices = self.get_global_indices(label)
        coords = {
            global_dim: global_axis,
            "amplitude_label": amplitude_axis,
        }
        return xr.DataArray(
            [
                [
                    estimated_amplitudes[i].clp[estimated_amplitude_axes[i].index(amplitude_label)]
                    for amplitude_label in amplitude_axis
                ]
                for i in global_indices
            ],
            dims=coords.keys(),
            coords=coords,
        )

    def get_dataset_residual(
        self,
        label: str,
        estimations: list[OptimizationEstimation],
        model_dim: str,
        model_axis: ArrayLike,
        global_dim: str,
        global_axis: ArrayLike,
    ) -> xr.DataArray:
        assert isinstance(self._data, LinkedOptimizationData)

        global_indices = self.get_global_indices(label)
        coords = {global_dim: global_axis, model_dim: model_axis}
        offsets = []
        for i in global_indices:
            group_label = self._data._group_labels[i]
            group_index = self._data.group_definitions[group_label].index(label)
            offsets.append(sum(self._data.group_sizes[group_label][:group_index]))
        size = model_axis.size
        return xr.DataArray(
            [
                estimations[i].residual[offset : offset + size]
                for i, offset in zip(global_indices, offsets)
            ],
            dims=coords.keys(),
            coords=coords,
        ).T

    def create_element_results(
        self,
        model: DataModel,
        global_dim: str,
        model_dim: str,
        amplitudes: xr.DataArray,
        concentrations: xr.DataArray,
    ) -> dict[str, ElementResult]:
        return {
            element.label: element.create_result(
                model, global_dim, model_dim, amplitudes, concentrations
            )
            for element in model.elements
        }

    def create_dataset_result(
        self,
        label: str,
        data: OptimizationData,
        concentration: OptimizationEstimation,
        estimated_amplitude_axes: list[list[str]],
        estimations: list[OptimizationEstimation],
    ) -> xr.Dataset:
        result_dataset = self.create_result_dataset(label, data)

        global_dim = result_dataset.attrs["global_dimension"]
        global_axis = result_dataset.coords[global_dim]
        model_dim = result_dataset.attrs["model_dimension"]
        model_axis = result_dataset.coords[model_dim]

        result_dataset["residual"] = self.get_dataset_residual(
            label, estimations, model_dim, model_axis, global_dim, global_axis
        )
        result_dataset.attrs["root_mean_square_error"] = np.sqrt(
            (result_dataset.residual.to_numpy() ** 2).sum() / sum(result_dataset.residual.shape)
        )
        self._data.data[label].unweight_result_dataset(result_dataset)
        result_dataset["fit"] = result_dataset.data - result_dataset.residual
        add_svd_to_result_dataset(result_dataset, global_dim, model_dim)

        concentrations = concentration.to_data_array(
            global_dim, global_axis, model_dim, model_axis
        )
        amplitudes = self.get_dataset_amplitudes(
            label,
            estimated_amplitude_axes,
            estimations,
            concentrations.amplitude_label,
            global_dim,
            global_axis,
        )
        self.add_element_results(
            result_dataset, label, global_dim, model_dim, amplitudes, concentrations
        )
        self.add_data_model_results(
            label, result_dataset, global_dim, model_dim, amplitudes, concentrations
        )
        return result_dataset

    def add_element_results(
        self,
        result_dataset: xr.Dataset,
        label: str,
        global_dim: str,
        model_dim: str,
        amplitudes: xr.DataArray,
        concentrations: xr.DataArray,
    ):
        for element_label, element_result in self.create_element_results(
            self._model.datasets[label], global_dim, model_dim, amplitudes, concentrations
        ).items():
            for amplitude_label, amplitude in element_result.amplitudes.items():
                result_dataset[f"{amplitude_label}_associated_amplitude_{element_label}"] = (
                    amplitude.rename(
                        {
                            c: f"{c}_{element_label}"
                            for c in amplitude.coords
                            if c not in [global_dim, model_dim]
                        }
                    )
                )
            for concentration_label, concentration in element_result.concentrations.items():
                result_dataset[
                    f"{concentration_label}_associated_concentration_{element_label}"
                ] = concentration.rename(
                    {
                        c: f"{c}_{element_label}"
                        for c in concentration.coords
                        if c not in [global_dim, model_dim]
                    }
                )
            for extra_label, extra in element_result.extra.items():
                result_dataset[f"{extra_label}_{element_label}"] = extra.rename(
                    {
                        c: f"{c}_{element_label}"
                        for c in extra.coords
                        if c not in [global_dim, model_dim]
                    }
                )

    def add_data_model_results(
        self,
        label: str,
        result_dataset: xr.Dataset,
        global_dim: str,
        model_dim: str,
        amplitudes: xr.DataArray,
        concentrations: xr.DataArray,
    ):
        data_model = self._model.datasets[label]
        for data_model_cls in {
            e[1].__class__.data_model_type
            for e in iterate_data_model_elements(data_model)
            if e[1].__class__.data_model_type is not None
        }:
            result_dataset.update(
                data_model_cls.create_result(
                    data_model,
                    global_dim,
                    model_dim,
                    amplitudes,
                    concentrations,
                )
            )

    def get_result(self) -> OptimizationObjectiveResult:
        return (
            self.create_single_dataset_result()
            if isinstance(self._data, OptimizationData)
            else self.create_multi_dataset_result()
        )
