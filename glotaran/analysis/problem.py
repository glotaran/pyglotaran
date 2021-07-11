from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from typing import Deque
from typing import Dict
from typing import NamedTuple
from typing import TypeVar

import numpy as np
import xarray as xr

from glotaran.analysis.nnls import residual_nnls
from glotaran.analysis.util import get_min_max_from_interval
from glotaran.analysis.variable_projection import residual_variable_projection
from glotaran.io.prepare_dataset import add_svd_to_dataset
from glotaran.model import DatasetModel
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme

if TYPE_CHECKING:
    from typing import Hashable


class ParameterError(ValueError):
    def __init__(self):
        super().__init__("Parameter not initialized")


class UngroupedProblemDescriptor(NamedTuple):
    dataset: DatasetModel
    data: xr.DataArray
    model_axis: np.ndarray
    global_axis: np.ndarray
    weight: xr.DataArray


class GroupedProblemDescriptor(NamedTuple):
    label: str
    indices: dict[str, int]
    axis: dict[str, np.ndarray]


class ProblemGroup(NamedTuple):
    data: np.ndarray
    weight: np.ndarray
    has_scaling: bool
    """Indicates if at least one dataset in the group needs scaling."""
    group: str
    """The concatenated labels of the involved datasets."""
    data_sizes: list[int]
    """Holds the sizes of the concatenated datasets."""
    descriptor: list[GroupedProblemDescriptor]


UngroupedBag = Dict[str, UngroupedProblemDescriptor]
GroupedBag = Deque[ProblemGroup]

XrDataContainer = TypeVar("XrDataContainer", xr.DataArray, xr.Dataset)


class Problem:
    """A Problem class"""

    def __init__(self, scheme: Scheme):
        """Initializes the Problem class from a scheme (:class:`glotaran.analysis.scheme.Scheme`)

        Args:
            scheme (Scheme): An instance of :class:`glotaran.analysis.scheme.Scheme`
                which defines your model, parameters, and data
        """

        self._scheme = scheme

        self._model = scheme.model

        self._bag = None

        self._residual_function = (
            residual_nnls if scheme.non_negative_least_squares else residual_variable_projection
        )
        self._parameters = None
        self._dataset_models = None

        self._overwrite_index_dependent = self.model.need_index_dependent()
        self._parameters = scheme.parameters.copy()
        self._parameter_history = []

        self._model.validate(raise_exception=True)

        self._prepare_data(scheme.data)

        # all of the above are always not None

        self._matrices = None
        self._reduced_matrices = None
        self._reduced_clps = None
        self._clps = None
        self._weighted_residuals = None
        self._residuals = None
        self._additional_penalty = None
        self._full_axis = None
        self._full_penalty = None

    @property
    def scheme(self) -> Scheme:
        """Property providing access to the used scheme

        Returns:
            Scheme: An instance of :class:`glotaran.analysis.scheme.Scheme`
                Provides access to data, model, parameters and optimization arguments.
        """
        return self._scheme

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
    def parameter_history(self) -> list[ParameterGroup]:
        return self._parameter_history

    @property
    def dataset_models(self) -> dict[str, DatasetModel]:
        return self._dataset_models

    @property
    def bag(self) -> UngroupedBag | GroupedBag:
        if not self._bag:
            self.init_bag()
        return self._bag

    @property
    def matrices(
        self,
    ) -> dict[str, np.ndarray | list[np.ndarray]]:
        if self._matrices is None:
            self.calculate_matrices()
        return self._matrices

    @property
    def reduced_matrices(
        self,
    ) -> dict[str, np.ndarray] | dict[str, list[np.ndarray]] | list[np.ndarray]:
        if self._reduced_matrices is None:
            self.calculate_matrices()
        return self._reduced_matrices

    @property
    def reduced_clps(
        self,
    ) -> dict[str, list[np.ndarray]]:
        if self._reduced_clps is None:
            self.calculate_residual()
        return self._reduced_clps

    @property
    def clps(
        self,
    ) -> dict[str, list[np.ndarray]]:
        if self._clps is None:
            self.calculate_residual()
        return self._clps

    @property
    def weighted_residuals(
        self,
    ) -> dict[str, list[np.ndarray]]:
        if self._weighted_residuals is None:
            self.calculate_residual()
        return self._weighted_residuals

    @property
    def residuals(
        self,
    ) -> dict[str, list[np.ndarray]]:
        if self._residuals is None:
            self.calculate_residual()
        return self._residuals

    @property
    def additional_penalty(
        self,
    ) -> dict[str, list[float]]:
        if self._additional_penalty is None:
            self.calculate_residual()
        return self._additional_penalty

    @property
    def full_penalty(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def cost(self) -> float:
        return np.sum(self._full_penalty)

    def save_parameters_for_history(self):
        self._parameter_history.append(self._parameters)

    def reset(self):
        """Resets all results and `DatasetModels`. Use after updating parameters."""
        self._dataset_models = {
            label: dataset_model.fill(self._model, self._parameters).set_data(self.data[label])
            for label, dataset_model in self._model.dataset.items()
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

    def _prepare_data(self, data: dict[str, xr.DataArray | xr.Dataset]):
        self._data = {}
        self._dataset_models = {}
        for label, dataset in data.items():
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

            if self.scheme.add_svd:
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

    #  @profile
    def create_result_data(
        self, copy: bool = True, history_index: int | None = None
    ) -> dict[str, xr.Dataset]:

        if history_index is not None and history_index != -1:
            self.parameters = self.parameter_history[history_index]

        self.prepare_result_creation()
        result_data = {}
        for label, dataset_model in self.dataset_models.items():
            result_data[label] = self.create_result_dataset(label, copy=copy)
            dataset_model.finalize_data(result_data[label])

        return result_data

    def create_result_dataset(self, label: str, copy: bool = True) -> xr.Dataset:
        dataset = self.data[label]
        dataset_model = self.dataset_models[label]
        global_dimension = dataset_model.get_global_dimension()
        model_dimension = dataset_model.get_model_dimension()
        if copy:
            dataset = dataset.copy()
        if dataset_model.index_dependent():
            dataset = self.create_index_dependent_result_dataset(label, dataset)
        else:
            dataset = self.create_index_independent_result_dataset(label, dataset)

        # TODO: adapt tests to handle add_svd=False
        if self.scheme.add_svd:
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

    def init_bag(self):
        """Initializes a problem bag."""
        raise NotImplementedError

    def create_index_dependent_result_dataset(self, label: str, dataset: xr.Dataset) -> xr.Dataset:
        """Creates a result datasets for index dependent matrices."""
        raise NotImplementedError

    def create_index_independent_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:
        """Creates a result datasets for index independent matrices."""
        raise NotImplementedError

    def calculate_matrices(self):
        raise NotImplementedError

    def calculate_residual(self):
        raise NotImplementedError

    def prepare_result_creation(self):
        pass
