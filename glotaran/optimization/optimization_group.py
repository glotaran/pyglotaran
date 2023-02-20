"""Module containing the optimization group class."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.io.prepare_dataset import add_svd_to_dataset
from glotaran.model import DatasetGroup
from glotaran.model.dataset_model import finalize_dataset_model
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked
from glotaran.optimization.estimation_provider import EstimationProvider
from glotaran.optimization.estimation_provider import EstimationProviderLinked
from glotaran.optimization.estimation_provider import EstimationProviderUnlinked
from glotaran.optimization.matrix_provider import MatrixProvider
from glotaran.optimization.matrix_provider import MatrixProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderUnlinked
from glotaran.parameter import Parameters
from glotaran.project import Scheme

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class OptimizationGroup:
    """A class to optimize a dataset group."""

    def __init__(
        self,
        scheme: Scheme,
        dataset_group: DatasetGroup,
    ):
        """Initialize an optimization group for a dataset group.

        Parameters
        ----------
        scheme : Scheme
            The optimization scheme.
        dataset_group : DatasetGroup
            The dataset group.
        """
        self._dataset_group = dataset_group
        self._dataset_group.set_parameters(scheme.parameters)
        self._data = scheme.data
        self._add_svd = scheme.add_svd
        link_clp = dataset_group.link_clp
        if link_clp is None:
            link_clp = dataset_group.is_linkable(scheme.parameters, scheme.data)

        if link_clp:
            data_provider = DataProviderLinked(scheme, dataset_group)
            matrix_provider = MatrixProviderLinked(dataset_group, data_provider)
            estimation_provider = EstimationProviderLinked(
                dataset_group, data_provider, matrix_provider
            )
        else:
            data_provider = DataProvider(scheme, dataset_group)  # type:ignore[assignment]
            matrix_provider = MatrixProviderUnlinked(  # type:ignore[assignment]
                self._dataset_group, data_provider
            )
            estimation_provider = EstimationProviderUnlinked(  # type:ignore[assignment]
                dataset_group, data_provider, matrix_provider  # type:ignore[arg-type]
            )

        self._data_provider: DataProvider = data_provider
        self._matrix_provider: MatrixProvider = matrix_provider
        self._estimation_provider: EstimationProvider = estimation_provider

        if self._add_svd:
            for dataset in self._data.values():
                self.add_svd_data(
                    "data",
                    dataset,
                    dataset.data.dims[0],
                    dataset.data.dims[1],
                )

    def calculate(self, parameters: Parameters):
        """Calculate the optimization group data.

        Parameters
        ----------
        parameters : Parameters
            The parameters.
        """
        self._dataset_group.set_parameters(parameters)
        self._matrix_provider.calculate()
        self._estimation_provider.estimate()

    def get_additional_penalties(self) -> list[float]:
        """Get additional penalties.

        Returns
        -------
        list[float]
            The additional penalties.
        """
        return self._estimation_provider.get_additional_penalties()

    def get_full_penalty(self) -> ArrayLike:
        """Get the full penalty.

        Returns
        -------
        ArrayLike
            The full penalty.
        """
        return self._estimation_provider.get_full_penalty()

    def add_weight_to_result_data(self, dataset_label: str, result_dataset: xr.Dataset):
        """Add weight to result dataset if dataset is weighted.

        Parameters
        ----------
        dataset_label : str
            The label of the data.
        result_dataset : xr.Dataset
            The label of the data.
        """
        weight = self._data_provider.get_weight(dataset_label)
        if weight is None:
            return
        result_dataset["weighted_residual"] = result_dataset["residual"]
        result_dataset["residual"] = result_dataset["residual"] / weight
        if "weight" not in result_dataset:
            if weight.shape != result_dataset.data.shape:
                weight = weight.T
            result_dataset["weight"] = (result_dataset.data.dims, weight)

    def create_result_data(self) -> dict[str, xr.Dataset]:
        """Create resulting datasets.

        Returns
        -------
        dict[str, xr.Dataset]
            The datasets with the results.
        """
        result_datasets = {
            label: data.copy()
            for label, data in self._data.items()
            if label in self._dataset_group.dataset_models.keys()
        }

        global_matrices, matrices = self._matrix_provider.get_result()
        clps, residuals = self._estimation_provider.get_result()

        for label, dataset_model in self._dataset_group.dataset_models.items():
            result_dataset = result_datasets[label]

            model_dimension = self._data_provider.get_model_dimension(label)
            result_dataset.attrs["model_dimension"] = model_dimension
            global_dimension = self._data_provider.get_global_dimension(label)
            result_dataset.attrs["global_dimension"] = global_dimension

            result_dataset["residual"] = residuals[label]
            self.add_weight_to_result_data(label, result_dataset)

            result_dataset["matrix"] = matrices[label]
            if label in global_matrices:
                result_dataset["global_matrix"] = global_matrices[label]
            result_dataset["clp"] = clps[label]

            if self._add_svd:
                self.add_svd_data("residual", result_dataset, model_dimension, global_dimension)
                if "weighted_residual" in result_dataset:
                    self.add_svd_data(
                        "weighted_residual", result_dataset, model_dimension, global_dimension
                    )

            # Calculate RMS
            size = result_dataset.residual.shape[0] * result_dataset.residual.shape[1]
            result_dataset.attrs["root_mean_square_error"] = np.sqrt(
                (result_dataset.residual**2).sum() / size
            ).data

            result_dataset.attrs["weighted_root_mean_square_error"] = (
                np.sqrt((result_dataset.weighted_residual**2).sum() / size).data
                if "weighted_residual" in result_dataset
                else result_dataset.attrs["root_mean_square_error"]
            )

            result_dataset.attrs["dataset_scale"] = (
                1
                if dataset_model.scale is None
                else dataset_model.scale.value  # type:ignore[union-attr]
            )

            # reconstruct fitted data
            result_dataset["fitted_data"] = result_dataset.data - result_dataset.residual

            finalize_dataset_model(dataset_model, result_dataset)

        return result_datasets

    @staticmethod
    def add_svd_data(name: str, dataset: xr.Dataset, lsv_dim: str, rsv_dim: str):
        """Add the SVD of a data matrix to a dataset.

        Parameters
        ----------
        name : str
            Name of the data matrix.
        dataset : xr.Dataset
            Dataset containing the data, which will be updated with the SVD values.
        lsv_dim : str
            The dimension name of the left singular vectors.
        rsv_dim : str
            The dimension name of the right singular vectors.
        """
        add_svd_to_dataset(
            dataset, name=name, lsv_dim=lsv_dim, rsv_dim=rsv_dim, data_array=dataset[name]
        )

    @property
    def number_of_clps(self) -> int:
        """Return number of conditionally linear parameters.

        Returns
        -------
        int
        """
        return self._matrix_provider.number_of_clps
