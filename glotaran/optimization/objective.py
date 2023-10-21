import numpy as np
import xarray as xr

from glotaran.model import ExperimentModel
from glotaran.model import iterate_data_model_elements
from glotaran.model import iterate_data_model_global_elements
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.penalty import calculate_clp_penalties
from glotaran.parameter.parameter import Parameter
from glotaran.typing.types import ArrayLike


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
            matrix = OptimizationMatrix.from_data(self._data)
            return [matrix.at_index(i) for i in range(self._data.global_axis.size)]
        return OptimizationMatrix.from_linked_data(self._data)  # type:ignore[arg-type]

    def calculate(self) -> ArrayLike:
        if isinstance(self._data, OptimizationData) and self._data.is_global:
            _, _, matrix = OptimizationMatrix.from_global_data(self._data)
            return OptimizationEstimation.calculate(
                matrix.array,
                self._data.flat_data,  # type:ignore[arg-type]
                self._model.residual_function,
            ).residual
        matrices = self.calculate_matrices()
        reduced_matrices = [
            matrices[i].reduce(index, self._model.clp_constraints, self._model.clp_relations)
            for i, index in enumerate(self._data.global_axis)
        ]
        estimations = [
            OptimizationEstimation.calculate(matrix.array, data, self._model.residual_function)
            for matrix, data in zip(reduced_matrices, self._data.data_slices)
        ]

        penalties = [e.residual for e in estimations]
        if len(self._model.clp_penalties) > 0:
            estimations = [
                e.resolve_clp(m.clp_axis, r.clp_axis, i, self._model.clp_relations)
                for e, m, r, i in zip(
                    estimations, matrices, reduced_matrices, self._data.global_axis
                )
            ]
            penalties.append(
                calculate_clp_penalties(
                    matrices, estimations, self._data.global_axis, self._model.clp_penalties
                )
            )
        return np.concatenate(penalties)

    def calculate_result_dataset(
        self, data: OptimizationData, matrix: OptimizationMatrix, dataset: xr.Dataset
    ):
        reduced_matrices = [
            matrix.at_index(i).reduce(
                index, self._model.clp_constraints, self._model.clp_relations
            )
            for i, index in enumerate(data.global_axis)
        ]
        estimations = [
            OptimizationEstimation.calculate(
                reduced_mat.array, data_slice, self._model.residual_function
            ).resolve_clp(
                matrix.clp_axis,
                reduced_mat.clp_axis,
                index,
                self._model.clp_relations,
            )
            for reduced_mat, data_slice, index in zip(
                reduced_matrices, data.data_slices, data.global_axis
            )
        ]
        dataset["clp"] = xr.DataArray(
            [e.clp for e in estimations],
            coords=(
                (data.global_dimension, data.global_axis),
                ("clp_label", matrix.clp_axis),
            ),
        )
        dataset["residual"] = xr.DataArray(
            [e.residual for e in estimations],
            coords=(
                (data.global_dimension, data.global_axis),
                (data.model_dimension, data.model_axis),
            ),
        ).T
        dataset["fitted_data"] = dataset.data - dataset.residual

    def calculate_result_dataset_global(
        self, data: OptimizationData, matrix: OptimizationMatrix, dataset: xr.Dataset
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
        dataset["fitted_data"] = dataset.data - dataset.residual

    def get_result_dataset(self, label: str, data: OptimizationData, add_svd=True) -> xr.Dataset:
        assert isinstance(data.model.data, xr.Dataset)
        dataset = data.model.data.copy()
        if dataset.data.dims != (data.model_dimension, data.global_dimension):
            dataset["data"] = dataset.data.T
        dataset.attrs["model_dimension"] = data.model_dimension
        dataset.attrs["global_dimension"] = data.global_dimension

        if isinstance(self._data, LinkedOptimizationData):
            scale = self._data.scales[label]
            dataset.attrs["scale"] = scale.value if isinstance(scale, Parameter) else scale

        matrix = OptimizationMatrix.from_data(data)
        matrix_coords = (
            (data.model_dimension, data.model_axis),
            ("clp_label", matrix.clp_axis),
        )
        if matrix.is_index_dependent:
            matrix_coords = (  # type:ignore[assignment]
                (data.global_dimension, data.global_axis),
                *matrix_coords,
            )
        dataset["matrix"] = xr.DataArray(matrix.array, coords=matrix_coords)
        if data.is_global:
            self.calculate_result_dataset_global(data, matrix, dataset)
        else:
            self.calculate_result_dataset(data, matrix, dataset)
        # Calculate RMS
        size = dataset.residual.shape[0] * dataset.residual.shape[1]
        dataset.attrs["root_mean_square_error"] = np.sqrt(
            (dataset.residual**2).sum() / size
        ).data

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
            model.add_to_result_data(data.model, dataset, False)  # type:ignore[union-attr]
        for _, model in iterate_data_model_global_elements(data.model):
            model.add_to_result_data(data.model, dataset, True)  # type:ignore[union-attr]

        return dataset

    def get_result(self) -> dict[str, xr.Dataset]:
        return {
            label: self.get_result_dataset(label, OptimizationData(data))
            for label, data in self._model.datasets.items()
        }
