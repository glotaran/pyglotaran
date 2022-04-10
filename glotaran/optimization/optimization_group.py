from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.io.prepare_dataset import add_svd_to_dataset
from glotaran.model import DatasetGroup
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked
from glotaran.optimization.estimation_provider import EstimationProviderLinked
from glotaran.optimization.estimation_provider import EstimationProviderUnlinked
from glotaran.optimization.matrix_provider import MatrixProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderUnlinked
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


class OptimizationGroup:
    def __init__(
        self,
        scheme: Scheme,
        dataset_group: DatasetGroup,
    ):
        """Create OptimizationGroup instance  from a scheme (:class:`.Scheme`)

        Args:
            scheme (Scheme): An instance of :class:`.Scheme`
                which defines your model, parameters, and data
        """

        self._dataset_group = dataset_group
        self._dataset_group.set_parameters(scheme.parameters)
        self._data = scheme.data
        self._add_svd = scheme.add_svd
        link_clp = dataset_group.link_clp
        if link_clp is None:
            link_clp = self.model.is_groupable(self.parameters, self.data)

        if link_clp:
            self._data_provider = DataProviderLinked(scheme, dataset_group)
            self._matrix_provider = MatrixProviderLinked(dataset_group, self._data_provider)
            self._estimation_provider = EstimationProviderLinked(
                dataset_group, self._data_provider, self._matrix_provider
            )
        else:
            self._data_provider = DataProvider(scheme, dataset_group)
            self._matrix_provider = MatrixProviderUnlinked(
                self._dataset_group, self._data_provider
            )
            self._estimation_provider = EstimationProviderUnlinked(
                dataset_group, self._data_provider, self._matrix_provider
            )

    def calculate(self, parameters: ParameterGroup):
        self._dataset_group.set_parameters(parameters)
        self._matrix_provider.calculate()
        self._estimation_provider.estimate()

    def get_full_penalty(self) -> np.typing.ArrayLike:
        return self._estimation_provider.get_full_penalty()

    def create_result_data(self, parameters: ParameterGroup) -> dict[str, xr.Dataset]:

        result_datasets = {label: data.copy() for label, data in self._data.items()}

        clp_labels, matrices = self._matrix_provider.get_result()
        clps, residuals = self._estimation_provider.get_result()

        for label, dataset_model in self.dataset_models.items():
            result_dataset = result_datasets[label]

            model_dimension = self._data_provider.get_model_dimension(label)
            model_axis = self._data_provider.get_model_axis(label)
            result_dataset.attrs["model_dimension"] = model_dimension
            global_dimension = self._data_provider.get_global_dimension(label)
            global_axis = self._data_provider.get_global_axis(label)
            result_dataset.attrs["global_dimension"] = global_dimension

            if dataset_model.is_index_dependent():
                matrix = xr.concat(
                    [
                        xr.DataArray(
                            m, coords=(("model_dimension", model_axis), ("clp_label", labels))
                        )
                        for labels, m in zip(clp_labels[label], matrices[label])
                    ],
                    dim=global_dimension,
                )
                #  matrix.coords[global_dimension] = global_axis
                result_dataset["matrix"] = matrix
                clp = xr.concat(
                    [
                        xr.DataArray(c, coords=(("clp_label", labels)))
                        for labels, c in zip(clp_labels[label], clps[label])
                    ],
                    dim=global_dimension,
                )
                result_dataset["clp"] = clp
            else:
                result_dataset["matrix"] = (
                    (
                        (model_dimension),
                        ("clp_label", clp_labels[label]),
                    ),
                    matrices[label],
                )
                result_dataset["clp"] = (
                    ((global_dimension, global_axis), ("clp_label", clp_labels[label])),
                    clps[label],
                )
                residual = xr.DataArray(
                    residuals[label],
                    coords=((global_dimension, global_axis), (model_dimension, model_axis)),
                )

                weight = self._data_provider.get_weight(label)
                if weight is not None:
                    result_dataset["weighted_residual"] = residual
                    residual = residual / weight
                result_dataset["residual"] = residual

            if self._add_svd:
                self._create_svd("residual", result_dataset, model_dimension, global_dimension)
                if "weighted_residual" in result_dataset:
                    self._create_svd(
                        "weighted_residual", result_dataset, model_dimension, global_dimension
                    )

            # Calculate RMS
            size = result_dataset.residual.shape[0] * result_dataset.residual.shape[1]
            result_dataset.attrs["root_mean_square_error"] = np.sqrt(
                (result_dataset.residual**2).sum() / size
            ).values
            if "weighted_residual" in result_dataset:
                result_dataset.attrs["weighted_root_mean_square_error"] = np.sqrt(
                    (result_dataset.weighted_residual**2).sum() / size
                ).values

            result_dataset.attrs["dataset_scale"] = (
                1 if dataset_model.scale is None else dataset_model.scale.value
            )

            # reconstruct fitted data
            result_dataset["fitted_data"] = result_dataset.data - result_dataset.residual

            dataset_model.finalize_data(result_dataset)

        return result_datasets

    def _create_svd(self, name: str, dataset: xr.Dataset, lsv_dim: str, rsv_dim: str):
        """Calculate the SVD of a data matrix in the dataset and add it to the dataset.

        Parameters
        ----------
        name : str
            Name of the data matrix.
        dataset : xr.Dataset
            Dataset containing the data, which will be updated with the SVD values.
        """
        add_svd_to_dataset(
            dataset, name=name, lsv_dim=lsv_dim, rsv_dim=rsv_dim, data_array=dataset[name]
        )
