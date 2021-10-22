from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from typing import TypeVar

import numpy as np
import xarray as xr

from glotaran.analysis.nnls import residual_nnls
from glotaran.analysis.optimization_group_calculator import OptimizationGroupCalculator
from glotaran.analysis.optimization_group_calculator_linked import (
    OptimizationGroupCalculatorLinked,
)
from glotaran.analysis.optimization_group_calculator_unlinked import (
    OptimizationGroupCalculatorUnlinked,
)
from glotaran.analysis.util import get_min_max_from_interval
from glotaran.analysis.variable_projection import residual_variable_projection
from glotaran.io.prepare_dataset import add_svd_to_dataset
from glotaran.model import DatasetGroup
from glotaran.model import DatasetModel
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.parameter import ParameterHistory
from glotaran.project import Scheme

if TYPE_CHECKING:
    from typing import Hashable


class InitialParameterError(ValueError):
    def __init__(self):
        super().__init__("Initial parameters can not be evaluated.")


class ParameterNotInitializedError(ValueError):
    def __init__(self):
        super().__init__("Parameter not initialized")


XrDataContainer = TypeVar("XrDataContainer", xr.DataArray, xr.Dataset)

residual_functions = {
    "variable_projection": residual_variable_projection,
    "non_negative_least_squares": residual_nnls,
}


class OptimizationGroup:
    def __init__(
        self,
        scheme: Scheme,
        dataset_group: DatasetGroup,
    ):
        """Create OptimizationGroup instance  from a scheme (:class:`glotaran.analysis.scheme.Scheme`)

        Args:
            scheme (Scheme): An instance of :class:`glotaran.analysis.scheme.Scheme`
                which defines your model, parameters, and data
        """

        self._model = scheme.model
        if scheme.parameters is None:
            raise ParameterNotInitializedError
        self._parameters = scheme.parameters.copy()
        self._dataset_group_model = dataset_group.model
        self._clp_link_tolerance = scheme.clp_link_tolerance

        try:
            self._residual_function = residual_functions[dataset_group.model.residual_function]
        except KeyError:
            raise ValueError(
                f"Unknown residual function '{dataset_group.model.residual_function}', "
                f"allowed functions are: {list(residual_functions.keys())}."
            )
        self._dataset_models = dataset_group.dataset_models

        self._overwrite_index_dependent = self.model.need_index_dependent()

        self._model.validate(raise_exception=True)

        self._prepare_data(scheme, list(dataset_group.dataset_models.keys()))
        self._dataset_labels = list(self.data.keys())

        link_clp = dataset_group.model.link_clp
        if link_clp is None:
            link_clp = self.model.is_groupable(self.parameters, self.data)

        self._calculator: OptimizationGroupCalculator = (
            OptimizationGroupCalculatorLinked(self)
            if link_clp
            else OptimizationGroupCalculatorUnlinked(self)
        )

        # all of the above are always not None

        self._matrices = None
        self._reduced_matrices = None
        self._reduced_clps = None
        self._clps = None
        self._weighted_residuals = None
        self._residuals = None
        self._additional_penalty = None
        self._full_penalty = None

    @property
    def model(self) -> Model:
        """Property providing access to the used model

        The model is a subclass of :class:`glotaran.model.Model` decorated with the `@model`
        decorator :class:`glotaran.model.model_decorator.model`
        For an example implementation see e.g. :class:`glotaran.builtin.models.kinetic_spectrum`

        Returns:
            Model: A subclass of :class:`glotaran.model.Model`
                The model must be decorated with the `@model` decorator
                :class:`glotaran.model.model_decorator.model`
        """
        return self._model

    @property
    def data(self) -> dict[str, xr.Dataset]:
        return self._data

    @property
    def parameters(self) -> ParameterGroup:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: ParameterGroup):
        self._parameters = parameters
        self.reset()

    @property
    def dataset_models(self) -> dict[str, DatasetModel]:
        return self._dataset_models

    @property
    def matrices(
        self,
    ) -> dict[str, np.ndarray | list[np.ndarray]]:
        if self._matrices is None:
            self._calculator.calculate_matrices()
        return self._matrices

    @property
    def reduced_matrices(
        self,
    ) -> dict[str, np.ndarray] | dict[str, list[np.ndarray]] | list[np.ndarray]:
        if self._reduced_matrices is None:
            self._calculator.calculate_matrices()
        return self._reduced_matrices

    @property
    def reduced_clps(
        self,
    ) -> dict[str, list[np.ndarray]]:
        if self._reduced_clps is None:
            self._calculator.calculate_residual()
        return self._reduced_clps

    @property
    def clps(
        self,
    ) -> dict[str, list[np.ndarray]]:
        if self._clps is None:
            self._calculator.calculate_residual()
        return self._clps

    @property
    def weighted_residuals(
        self,
    ) -> dict[str, list[np.ndarray]]:
        if self._weighted_residuals is None:
            self._calculator.calculate_residual()
        return self._weighted_residuals

    @property
    def residuals(
        self,
    ) -> dict[str, list[np.ndarray]]:
        if self._residuals is None:
            self._calculator.calculate_residual()
        return self._residuals

    @property
    def additional_penalty(
        self,
    ) -> dict[str, list[float]]:
        if self._additional_penalty is None:
            self._calculator.calculate_residual()
        return self._additional_penalty

    @property
    def full_penalty(self) -> np.ndarray:
        if self._full_penalty is None:
            self._calculator.calculate_full_penalty()
        return self._full_penalty

    @property
    def cost(self) -> float:
        return 0.5 * np.dot(self.full_penalty, self.full_penalty)

    def reset(self):
        """Resets all results and `DatasetModels`. Use after updating parameters."""
        self._dataset_models = {
            label: dataset_model.fill(self._model, self._parameters).set_data(self.data[label])
            for label, dataset_model in self.model.dataset.items()
            if label in self._dataset_labels
        }
        if self._overwrite_index_dependent:
            for d in self._dataset_models.values():
                d.overwrite_index_dependent(self._overwrite_index_dependent)
        self._reset_results()

    def _reset_results(self):
        self._matrices = None
        self._reduced_matrices = None
        self._reduced_clps = None
        self._clps = None
        self._weighted_residuals = None
        self._residuals = None
        self._additional_penalty = None
        self._full_penalty = None

    def _prepare_data(self, scheme: Scheme, labels: list[str]):
        self._data = {}
        self._dataset_models = {}
        for label, dataset in scheme.data.items():
            if label not in labels:
                continue
            if isinstance(dataset, xr.DataArray):
                dataset = dataset.to_dataset(name="data")

            dataset_model = self._model.dataset[label]
            dataset_model = dataset_model.fill(self.model, self.parameters)
            dataset_model.set_data(dataset)
            if self._overwrite_index_dependent:
                dataset_model.overwrite_index_dependent(self._overwrite_index_dependent)
            self._dataset_models[label] = dataset_model
            global_dimension = dataset_model.get_global_dimension()
            model_dimension = dataset_model.get_model_dimension()

            dataset = self._transpose_dataset(
                dataset, ordered_dims=[model_dimension, global_dimension]
            )

            if scheme.add_svd:
                add_svd_to_dataset(dataset, lsv_dim=model_dimension, rsv_dim=global_dimension)

            self._add_weight(label, dataset)
            self._data[label] = dataset

    def _transpose_dataset(
        self, datacontainer: XrDataContainer, ordered_dims: list[Hashable]
    ) -> XrDataContainer:
        """Reorder dimension of the datacontainer with the order provided by ``ordered_dims``.

        Parameters
        ----------
        dataset: XrDataContainer
            Dataset to be reordered
        ordered_dims: list[Hashable]
            Order the dimensions should be in.

        Returns
        -------
        XrDataContainer
            Datacontainer with reordered dimensions.
        """
        ordered_dims = list(filter(lambda dim: dim in datacontainer.dims, ordered_dims))
        ordered_dims += list(filter(lambda dim: dim not in ordered_dims, datacontainer.dims))
        return datacontainer.transpose(*ordered_dims)

    def _add_weight(self, label, dataset):

        # if the user supplies a weight we ignore modeled weights
        if "weight" in dataset:
            if any(label in weight.datasets for weight in self.model.weights):
                warnings.warn(
                    f"Ignoring model weight for dataset '{label}'"
                    " because weight is already supplied by dataset."
                )
            return
        dataset_model = self.dataset_models[label]
        dataset_model.set_data(dataset)
        global_dimension = dataset_model.get_global_dimension()
        model_dimension = dataset_model.get_model_dimension()

        global_axis = dataset.coords[global_dimension]
        model_axis = dataset.coords[model_dimension]

        for weight in self.model.weights:
            if label in weight.datasets:
                if "weight" not in dataset:
                    dataset["weight"] = xr.DataArray(
                        np.ones_like(dataset.data), coords=dataset.data.coords
                    )

                idx = {}
                if weight.global_interval is not None:
                    idx[global_dimension] = get_min_max_from_interval(
                        weight.global_interval, global_axis
                    )
                if weight.model_interval is not None:
                    idx[model_dimension] = get_min_max_from_interval(
                        weight.model_interval, model_axis
                    )
                dataset.weight[idx] *= weight.value

    def create_result_data(
        self,
        parameter_history: ParameterHistory = None,
        copy: bool = True,
        success: bool = True,
        add_svd: bool = True,
    ) -> dict[str, xr.Dataset]:

        if not success:
            if parameter_history is not None and parameter_history.number_of_records > 1:
                self.parameters.set_from_history(parameter_history, -2)
            else:
                raise InitialParameterError()

        self.reset()
        self._calculator.prepare_result_creation()
        result_data = {}
        for label, dataset_model in self.dataset_models.items():
            result_data[label] = self.create_result_dataset(label, copy=copy)
            dataset_model.finalize_data(result_data[label])

        return result_data

    def create_result_dataset(
        self, label: str, copy: bool = True, add_svd: bool = True
    ) -> xr.Dataset:
        dataset = self.data[label]
        dataset_model = self.dataset_models[label]
        global_dimension = dataset_model.get_global_dimension()
        model_dimension = dataset_model.get_model_dimension()
        if copy:
            dataset = dataset.copy()
        if dataset_model.is_index_dependent():
            dataset = self._calculator.create_index_dependent_result_dataset(label, dataset)
        else:
            dataset = self._calculator.create_index_independent_result_dataset(label, dataset)

        # TODO: adapt tests to handle add_svd=False
        if add_svd:
            self._create_svd("weighted_residual", dataset, model_dimension, global_dimension)
            self._create_svd("residual", dataset, model_dimension, global_dimension)

        # Calculate RMS
        size = dataset.residual.shape[0] * dataset.residual.shape[1]
        dataset.attrs["root_mean_square_error"] = np.sqrt(
            (dataset.residual ** 2).sum() / size
        ).values
        size = dataset.weighted_residual.shape[0] * dataset.weighted_residual.shape[1]
        dataset.attrs["weighted_root_mean_square_error"] = np.sqrt(
            (dataset.weighted_residual ** 2).sum() / size
        ).values
        if dataset_model.scale is not None:
            dataset.attrs["dataset_scale"] = dataset_model.scale.value
        else:
            dataset.attrs["dataset_scale"] = 1

        # reconstruct fitted data
        dataset["fitted_data"] = dataset.data - dataset.residual
        return dataset

    def _create_svd(self, name: str, dataset: xr.Dataset, lsv_dim: str, rsv_dim: str):
        """Calculate the SVD of a data matrix in the dataset and add it to the dataset.

        Parameters
        ----------
        name : str
            Name of the data matrix.
        dataset : xr.Dataset
            Dataset containing the data, which will be updated with the SVD values.
        """
        data_array: xr.DataArray = self._transpose_dataset(
            dataset[name],
            ordered_dims=[lsv_dim, rsv_dim],
        )

        add_svd_to_dataset(
            dataset, name=name, lsv_dim=lsv_dim, rsv_dim=rsv_dim, data_array=data_array
        )
