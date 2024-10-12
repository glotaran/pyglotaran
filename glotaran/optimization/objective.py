from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import numpy as np
import xarray as xr
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from glotaran.model.data_model import DataModel
from glotaran.model.data_model import iterate_data_model_elements
from glotaran.model.element import Element
from glotaran.model.element import ElementResult
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.penalty import calculate_clp_penalties
from glotaran.parameter.parameter import Parameter

if TYPE_CHECKING:
    from glotaran.model.experiment_model import ExperimentModel
    from glotaran.typing.types import ArrayLike


def add_svd_to_result_dataset(dataset: xr.Dataset, global_dim: str, model_dim: str):
    for name in ["data", "residual"]:
        if f"{name}_singular_values" in dataset:
            continue
        lsv, sv, rsv = np.linalg.svd(dataset[name].data, full_matrices=False)
        dataset[f"{name}_left_singular_vectors"] = (
            (model_dim, "left_singular_value_index"),
            lsv,
        )
        dataset[f"{name}_singular_values"] = (("singular_value_index"), sv)
        dataset[f"{name}_right_singular_vectors"] = (
            (global_dim, "right_singular_value_index"),
            rsv.T,
        )


class DatasetResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    elements: dict[str, ElementResult] = Field(default_factory=dict)
    activations: xr.Dataset = Field(default_factory=dict)
    input_data: xr.DataArray | xr.Dataset | None = None
    residuals: xr.DataArray | xr.Dataset | None = None

    @property
    def fitted_data(self) -> xr.Dataset:
        if self.input_data is None or self.residuals is None:
            raise ValueError("Data and residuals must be set to calculate fitted data.")
        return self.input_data - self.residuals


@dataclass
class OptimizationObjectiveResult:
    data: dict[str, DatasetResult]
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
            matrices[i].reduce(index, self._model.clp_relations, copy=True)
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

    def get_global_indices(self, label: str) -> list[int]:
        assert isinstance(self._data, LinkedOptimizationData)
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

    def create_global_result(self) -> OptimizationObjectiveResult:
        label = next(iter(self._model.datasets.keys()))
        result_dataset = self.create_result_dataset(label, self._data)

        global_dim = result_dataset.attrs["global_dimension"]
        global_axis = result_dataset.coords[global_dim]
        model_dim = result_dataset.attrs["model_dimension"]
        model_axis = result_dataset.coords[model_dim]

        matrix = OptimizationMatrix.from_data(self._data).to_data_array(
            global_dim, global_axis, model_dim, model_axis
        )
        global_matrix = OptimizationMatrix.from_data(self._data, global_matrix=True).to_data_array(
            model_dim, model_axis, global_dim, global_axis
        )
        _, _, full_matrix = OptimizationMatrix.from_global_data(self._data)
        estimation = OptimizationEstimation.calculate(
            full_matrix.array,
            self._data.flat_data,  # type:ignore[arg-type]
            self._data.model.residual_function,
        )
        clp = xr.DataArray(
            estimation.clp.reshape((len(global_matrix.clp_axis), len(matrix.clp_axis))),
            coords={
                "global_clp_label": global_matrix.clp_axis,
                "clp_label": matrix.clp_axis,
            },
            dims=["global_clp_label", "clp_label"],
        )
        result_dataset["residual"] = xr.DataArray(
            estimation.residual.reshape(global_axis.size, model_axis.size),
            coords=((global_dim, global_axis), (model_dim, model_axis)),
        ).T
        result_dataset.attrs["root_mean_square_error"] = np.sqrt(
            (result_dataset.residual.to_numpy() ** 2).sum() / sum(result_dataset.residual.shape)
        )
        clp_size = len(matrix.clp_axis) + len(global_matrix.clp_axis)
        self._data.unweight_result_dataset(result_dataset)
        result_dataset["fit"] = result_dataset.data - result_dataset.residual
        add_svd_to_result_dataset(result_dataset, global_dim, model_dim)
        result = DatasetResult(
            result_dataset,
            {
                label: ElementResult(
                    amplitudes={"clp": clp},
                    concentrations={"global": global_matrix, "model": matrix},
                )
            },
            {},
        )
        return OptimizationObjectiveResult(
            data={label: result}, clp_size=clp_size, additional_penalty=0
        )

    def create_single_dataset_result(self) -> OptimizationObjectiveResult:
        assert isinstance(self._data, OptimizationData)
        if self._data.is_global:
            return self.create_global_result()

        label = next(iter(self._model.datasets.keys()))
        result_dataset = self.create_result_dataset(label, self._data)

        global_dim = result_dataset.attrs["global_dimension"]
        global_axis = result_dataset.coords[global_dim]
        model_dim = result_dataset.attrs["model_dimension"]
        model_axis = result_dataset.coords[model_dim]

        concentrations = OptimizationMatrix.from_data(self._data)
        additional_penalty = 0

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
        amplitudes = xr.DataArray(
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
            (result_dataset.residual.to_numpy() ** 2).sum() / sum(result_dataset.residual.shape)
        )
        additional_penalty = sum(
            calculate_clp_penalties(
                [concentrations],
                estimations,
                global_axis,
                self._model.clp_penalties,
            )
        )
        element_results = self.create_element_results(
            self._model.datasets[label], global_dim, model_dim, amplitudes, concentration
        )
        activations = self.create_data_model_results(
            label, global_dim, model_dim, amplitudes, concentration
        )

        self._data.unweight_result_dataset(result_dataset)
        add_svd_to_result_dataset(result_dataset, global_dim, model_dim)
        result = DatasetResult(
            input_data=result_dataset.data,
            residuals=result_dataset.residual,
            elements=element_results,
            activations=activations,
        )
        return OptimizationObjectiveResult(
            data={label: result}, clp_size=clp_size, additional_penalty=additional_penalty
        )

    def create_multi_dataset_result(self) -> OptimizationObjectiveResult:
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
            data=results,
            clp_size=clp_size,
            additional_penalty=additional_penalty,
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
        assert any(isinstance(element, str) for element in model.elements) is False
        return {
            element.label: element.create_result(
                model, global_dim, model_dim, amplitudes, concentrations
            )
            for element in cast(list[Element], model.elements)
        }

    def create_dataset_result(
        self,
        label: str,
        data: OptimizationData,
        concentration: OptimizationMatrix,
        estimated_amplitude_axes: list[list[str]],
        estimations: list[OptimizationEstimation],
    ) -> xr.Dataset:
        assert isinstance(self._data, LinkedOptimizationData)
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
        element_results = self.create_element_results(
            self._model.datasets[label], global_dim, model_dim, amplitudes, concentrations
        )
        activations = self.create_data_model_results(
            label, global_dim, model_dim, amplitudes, concentration
        )

        return DatasetResult(
            input_data=result_dataset.data,
            residuals=result_dataset.residual,
            elements=element_results,
            activations=activations,
        )

    def create_data_model_results(
        self,
        label: str,
        global_dim: str,
        model_dim: str,
        amplitudes: xr.DataArray,
        concentrations: xr.DataArray,
    ) -> xr.Dataset:
        result = {}
        data_model = self._model.datasets[label]
        assert any(isinstance(e, str) for _, e in iterate_data_model_elements(data_model)) is False
        for data_model_cls in {
            e.__class__.data_model_type
            for _, e in cast(tuple[Any, Element], iterate_data_model_elements(data_model))
            if e.__class__.data_model_type is not None
        }:
            result = result | data_model_cls.create_result(
                data_model,
                global_dim,
                model_dim,
                amplitudes,
                concentrations,
            )
        return xr.Dataset(result)

    def get_result(self) -> OptimizationObjectiveResult:
        return (
            self.create_single_dataset_result()
            if isinstance(self._data, OptimizationData)
            else self.create_multi_dataset_result()
        )
