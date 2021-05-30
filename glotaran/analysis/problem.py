from __future__ import annotations

import warnings
from typing import Deque
from typing import Dict
from typing import NamedTuple

import numpy as np
import xarray as xr

from glotaran.analysis.nnls import residual_nnls
from glotaran.analysis.util import get_min_max_from_interval
from glotaran.analysis.variable_projection import residual_variable_projection
from glotaran.model import DatasetDescriptor
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


class ParameterError(ValueError):
    def __init__(self):
        super().__init__("Parameter not initialized")


class UngroupedProblemDescriptor(NamedTuple):
    dataset: DatasetDescriptor
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
        self._global_dimension = scheme.model.global_dimension
        self._model_dimension = scheme.model.model_dimension
        self._prepare_data(scheme.data)

        self._index_dependent = scheme.model.index_dependent()
        self._grouped = scheme.model.grouped()
        self._bag = None
        self._groups = None

        self._residual_function = (
            residual_nnls if scheme.non_negative_least_squares else residual_variable_projection
        )
        self._parameters = None
        self._filled_dataset_descriptors = None

        self.parameters = scheme.parameters.copy()
        self._parameter_history = []

        # all of the above are always not None

        self._clp_labels = None
        self._matrices = None
        self._reduced_clp_labels = None
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
    def grouped(self) -> bool:
        return self._grouped

    @property
    def index_dependent(self) -> bool:
        return self._index_dependent

    @property
    def filled_dataset_descriptors(self) -> dict[str, DatasetDescriptor]:
        return self._filled_dataset_descriptors

    @property
    def bag(self) -> UngroupedBag | GroupedBag:
        if not self._bag:
            self.init_bag()
        return self._bag

    @property
    def groups(self) -> dict[str, list[str]]:
        if not self._groups and self._grouped:
            self.init_bag()
        return self._groups

    @property
    def clp_labels(
        self,
    ) -> dict[str, list[str] | list[list[str]]]:
        if self._clp_labels is None:
            self.calculate_matrices()
        return self._clp_labels

    @property
    def matrices(
        self,
    ) -> dict[str, np.ndarray | list[np.ndarray]]:
        if self._matrices is None:
            self.calculate_matrices()
        return self._matrices

    @property
    def reduced_clp_labels(
        self,
    ) -> dict[str, list[str] | list[list[str]]]:
        if self._reduced_clp_labels is None:
            self.calculate_matrices()
        return self._reduced_clp_labels

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
            self.calculate_additional_penalty()
        return self._additional_penalty

    @property
    def full_penalty(self) -> np.ndarray:
        if self._full_penalty is None:
            residuals = self.weighted_residuals
            additional_penalty = self.additional_penalty
            if not self.grouped:
                residuals = [np.concatenate(residuals[label]) for label in residuals.keys()]

            self._full_penalty = (
                np.concatenate((np.concatenate(residuals), additional_penalty))
                if additional_penalty is not None
                else np.concatenate(residuals)
            )
        return self._full_penalty

    @property
    def cost(self) -> float:
        return np.sum(self._full_penalty)

    def save_parameters_for_history(self):
        self._parameter_history.append(self._parameters)

    def reset(self):
        """Resets all results and `DatasetDescriptors`. Use after updating parameters."""
        self._filled_dataset_descriptors = {
            label: descriptor.fill(self._model, self._parameters)
            for label, descriptor in self._model.dataset.items()
        }
        self._reset_results()

    def _reset_results(self):
        self._clp_labels = None
        self._matrices = None
        self._reduced_clp_labels = None
        self._reduced_matrices = None
        self._reduced_clps = None
        self._clps = None
        self._weighted_residuals = None
        self._residuals = None
        self._additional_penalty = None
        self._full_penalty = None

    def _prepare_data(self, data: dict[str, xr.DataArray | xr.Dataset]):
        self._data = {}
        for label, dataset in data.items():
            if self.model.model_dimension not in dataset.dims:
                raise ValueError(
                    "Missing coordinates for dimension "
                    f"'{self.model.model_dimension}' in data for dataset "
                    f"'{label}'"
                )
            if self.model.global_dimension not in dataset.dims:
                raise ValueError(
                    "Missing coordinates for dimension "
                    f"'{self.model.global_dimension}' in data for dataset "
                    f"'{label}'"
                )
            if isinstance(dataset, xr.DataArray):
                dataset = dataset.to_dataset(name="data")

            dataset = self._transpose_dataset(dataset)
            self._add_weight(label, dataset)

            # TODO: avoid computation if not requested
            l, s, r = np.linalg.svd(dataset.data, full_matrices=False)
            dataset["data_left_singular_vectors"] = (("time", "left_singular_value_index"), l)
            dataset["data_singular_values"] = (("singular_value_index"), s)
            dataset["data_right_singular_vectors"] = (
                ("right_singular_value_index", "spectral"),
                r,
            )

            self._data[label] = dataset

    def _transpose_dataset(self, dataset):
        new_dims = [self.model.model_dimension, self.model.global_dimension]
        new_dims += [
            dim
            for dim in dataset.dims
            if dim not in [self.model.model_dimension, self.model.global_dimension]
        ]

        return dataset.transpose(*new_dims)

    def _add_weight(self, label, dataset):

        # if the user supplies a weight we ignore modeled weights
        if "weight" in dataset:
            if any(label in weight.datasets for weight in self.model.weights):
                warnings.warn(
                    f"Ignoring model weight for dataset '{label}'"
                    " because weight is already supplied by dataset."
                )
            return

        global_axis = dataset.coords[self.model.global_dimension]
        model_axis = dataset.coords[self.model.model_dimension]

        for weight in self.model.weights:
            if label in weight.datasets:
                if "weight" not in dataset:
                    dataset["weight"] = xr.DataArray(
                        np.ones_like(dataset.data), coords=dataset.data.coords
                    )

                idx = {}
                if weight.global_interval is not None:
                    idx[self.model.global_dimension] = get_min_max_from_interval(
                        weight.global_interval, global_axis
                    )
                if weight.model_interval is not None:
                    idx[self.model.model_dimension] = get_min_max_from_interval(
                        weight.model_interval, model_axis
                    )
                dataset.weight[idx] *= weight.value

    def calculate_matrices(self):
        if self._parameters is None:
            raise ParameterError
        if self.index_dependent:
            self.calculate_index_dependent_matrices()
        else:
            self.calculate_index_independent_matrices()

    def calculate_residual(self):
        if self._index_dependent:
            self.calculate_index_dependent_residual()
        else:
            self.calculate_index_independent_residual()

    def calculate_additional_penalty(self) -> np.ndarray | dict[str, np.ndarray]:
        """Calculates additional penalties by calling the model.additional_penalty function."""
        if (
            callable(self.model.has_additional_penalty_function)
            and self.model.has_additional_penalty_function()
        ):
            self._additional_penalty = self.model.additional_penalty_function(
                self.parameters,
                self.clp_labels,
                self.clps,
                self.matrices,
                self.data,
                self._scheme.group_tolerance,
            )
        else:
            self._additional_penalty = None
        return self._additional_penalty

    def create_result_data(
        self, copy: bool = True, history_index: int | None = None
    ) -> dict[str, xr.Dataset]:

        if history_index is not None and history_index != -1:
            self.parameters = self.parameter_history[history_index]
        result_data = {label: self.create_result_dataset(label, copy=copy) for label in self.data}

        if callable(self.model.finalize_data):
            self.model.finalize_data(self, result_data)

        return result_data

    def create_result_dataset(self, label: str, copy: bool = True) -> xr.Dataset:
        dataset = self.data[label]
        if copy:
            dataset = dataset.copy()
        if self.index_dependent:
            dataset = self.create_index_dependent_result_dataset(label, dataset)
        else:
            dataset = self.create_index_independent_result_dataset(label, dataset)

        self._create_svd("weighted_residual", dataset)
        self._create_svd("residual", dataset)

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

    def _create_svd(self, name: str, dataset: xr.Dataset):
        l, v, r = np.linalg.svd(dataset[name], full_matrices=False)

        dataset[f"{name}_left_singular_vectors"] = (
            (self._model_dimension, "left_singular_value_index"),
            l,
        )

        dataset[f"{name}_right_singular_vectors"] = (
            ("right_singular_value_index", self._global_dimension),
            r,
        )

        dataset[f"{name}_singular_values"] = (("singular_value_index"), v)

    def init_bag(self):
        raise NotImplementedError

    def calculate_index_dependent_matrices(
        self,
    ) -> tuple[
        dict[str, list[list[str]]],
        dict[str, list[np.ndarray]],
        dict[str, list[str]],
        dict[str, list[np.ndarray]],
    ]:
        raise NotImplementedError

    def calculate_index_independent_matrices(
        self,
    ) -> tuple[
        dict[str, list[str]],
        dict[str, np.ndarray],
        dict[str, list[str]],
        dict[str, np.ndarray],
    ]:
        raise NotImplementedError

    def calculate_index_dependent_residual(
        self,
    ) -> tuple[
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
    ]:
        raise NotImplementedError

    def calculate_index_independent_residual(
        self,
    ) -> tuple[
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
    ]:
        raise NotImplementedError

    def create_index_dependent_result_dataset(self, label: str, dataset: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError

    def create_index_independent_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:
        raise NotImplementedError
