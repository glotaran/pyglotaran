import collections
import itertools
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

import numpy as np
import xarray as xr

from glotaran.analysis.nnls import residual_nnls
from glotaran.analysis.variable_projection import residual_variable_projection
from glotaran.model import DatasetDescriptor
from glotaran.model import Model
from glotaran.parameter import ParameterGroup

from .scheme import Scheme


class ParameterError(ValueError):
    def __init__(self):
        super().__init__("Parameter not initialized")


class ProblemDescriptor(NamedTuple):
    dataset: DatasetDescriptor
    data: xr.DataArray
    model_axis: np.ndarray
    global_axis: np.ndarray
    weight: xr.DataArray


class GroupedProblemDescriptor(NamedTuple):
    label: str
    index: Any
    axis: np.ndarray


class GroupedProblem(NamedTuple):
    data: np.ndarray
    weight: np.ndarray
    has_scaling: bool
    """Indicates if at least one dataset in the group needs scaling."""
    group: str
    """The concatenated labels of the involved datasets."""
    data_sizes: List[int]
    """Holds the sizes of the concatenated datasets."""
    descriptor: GroupedProblemDescriptor


UngroupedBag = Dict[str, ProblemDescriptor]
GroupedBag = Deque[GroupedProblem]


class LabelAndMatrix(NamedTuple):
    clp_label: List[str]
    matrix: np.ndarray


class Problem:
    """A Problem class """

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
        self._data = scheme.data

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
    def data(self) -> Dict[str, xr.Dataset]:
        return self._data

    @property
    def parameters(self) -> ParameterGroup:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: ParameterGroup):
        self._parameters = parameters
        self.reset()

    @property
    def parameter_history(self) -> List[ParameterGroup]:
        return self._parameter_history

    @property
    def grouped(self) -> bool:
        return self._grouped

    @property
    def index_dependent(self) -> bool:
        return self._index_dependent

    @property
    def filled_dataset_descriptors(self) -> Dict[str, DatasetDescriptor]:
        return self._filled_dataset_descriptors

    @property
    def bag(self) -> Union[UngroupedBag, GroupedBag]:
        if not self._bag:
            self._init_bag()
        return self._bag

    @property
    def groups(self) -> Dict[str, List[str]]:
        if not self._groups and self._grouped:
            self._init_bag()
        return self._groups

    @property
    def clp_labels(
        self,
    ) -> Dict[str, Union[List[str], List[List[str]]]]:
        if self._clp_labels is None:
            self.calculate_matrices()
        return self._clp_labels

    @property
    def matrices(
        self,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        if self._matrices is None:
            self.calculate_matrices()
        return self._matrices

    @property
    def reduced_clp_labels(
        self,
    ) -> Dict[str, Union[List[str], List[List[str]]]]:
        if self._reduced_clp_labels is None:
            self.calculate_matrices()
        return self._reduced_clp_labels

    @property
    def reduced_matrices(
        self,
    ) -> Union[Dict[str, np.ndarray], Dict[str, List[np.ndarray]], List[np.ndarray],]:
        if self._reduced_matrices is None:
            self.calculate_matrices()
        return self._reduced_matrices

    @property
    def reduced_clps(
        self,
    ) -> Dict[str, List[np.ndarray]]:
        if self._reduced_clps is None:
            self.calculate_residual()
        return self._reduced_clps

    @property
    def clps(
        self,
    ) -> Dict[str, List[np.ndarray]]:
        if self._clps is None:
            self.calculate_residual()
        return self._clps

    @property
    def weighted_residuals(
        self,
    ) -> Dict[str, List[np.ndarray]]:
        if self._weighted_residuals is None:
            self.calculate_residual()
        return self._weighted_residuals

    @property
    def residuals(
        self,
    ) -> Dict[str, List[np.ndarray]]:
        if self._residuals is None:
            self.calculate_residual()
        return self._residuals

    @property
    def additional_penalty(
        self,
    ) -> Dict[str, List[float]]:
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

    def _init_bag(self):
        if self._grouped:
            self._init_grouped_bag()
        else:
            self._init_ungrouped_bag()

    def _init_ungrouped_bag(self):
        self._bag = {}
        for label in self._scheme.model.dataset:
            dataset = self._scheme.data[label]
            data = dataset.data
            weight = dataset.weight if "weight" in dataset else None
            if weight is not None:
                data = data * weight
                dataset["weighted_data"] = data
            self._bag[label] = ProblemDescriptor(
                self._scheme.model.dataset[label],
                data,
                dataset.coords[self._model_dimension].values,
                dataset.coords[self._global_dimension].values,
                weight,
            )

    def _init_grouped_bag(self):
        datasets = None
        for label in self._model.dataset:
            dataset = self._data[label]
            if "weight" in dataset:
                weight = dataset.weight
                data = dataset.data * weight
                dataset["weighted_data"] = data
            else:
                weight = xr.DataArray(np.ones_like(dataset.data), coords=dataset.data.coords)
                data = dataset.data
            global_axis = dataset.coords[self._global_dimension].values
            model_axis = dataset.coords[self._model_dimension].values
            has_scaling = self._model.dataset[label].scale is not None
            if self._bag is None:
                self._bag = collections.deque(
                    GroupedProblem(
                        data=data.isel({self._global_dimension: i}).values,
                        weight=weight.isel({self._global_dimension: i}).values,
                        has_scaling=has_scaling,
                        group=label,
                        data_sizes=[data.isel({self._global_dimension: i}).values.size],
                        descriptor=[GroupedProblemDescriptor(label, value, model_axis)],
                    )
                    for i, value in enumerate(global_axis)
                )
                datasets = collections.deque([label] for _, _ in enumerate(global_axis))
                self._full_axis = collections.deque(global_axis)
            else:
                self._append_to_grouped_bag(
                    label, datasets, global_axis, model_axis, data, weight, has_scaling
                )
        self._full_axis = np.asarray(self._full_axis)
        self._groups = {"".join(d): d for d in datasets}

    def _append_to_grouped_bag(
        self,
        label: str,
        datasets: Deque[str],
        global_axis: np.ndarray,
        model_axis: np.ndarray,
        data: xr.DataArray,
        weight: xr.DataArray,
        has_scaling: bool,
    ):
        i1, i2 = _find_overlap(self._full_axis, global_axis, atol=self._scheme.group_tolerance)

        for i, j in enumerate(i1):
            datasets[j].append(label)
            data_stripe = data.isel({self._global_dimension: i2[i]}).values
            self._bag[j] = GroupedProblem(
                data=np.concatenate(
                    [
                        self._bag[j].data,
                        data_stripe,
                    ]
                ),
                weight=np.concatenate(
                    [
                        self._bag[j].weight,
                        weight.isel({self._global_dimension: i2[i]}).values,
                    ]
                ),
                has_scaling=has_scaling or self._bag[j].has_scaling,
                group=self._bag[j].group + label,
                data_sizes=self._bag[j].data_sizes + [data_stripe.size],
                descriptor=self._bag[j].descriptor
                + [GroupedProblemDescriptor(label, global_axis[i2[i]], model_axis)],
            )

        # Add non-overlaping regions
        begin_overlap = i2[0] if len(i2) != 0 else 0
        end_overlap = i2[-1] + 1 if len(i2) != 0 else 0
        for i in itertools.chain(range(begin_overlap), range(end_overlap, len(global_axis))):
            data_stripe = data.isel({self._global_dimension: i}).values
            problem = GroupedProblem(
                data=data_stripe,
                weight=weight.isel({self._global_dimension: i}).values,
                has_scaling=has_scaling,
                group=label,
                data_sizes=[data_stripe.size],
                descriptor=[GroupedProblemDescriptor(label, global_axis[i], model_axis)],
            )
            if i < end_overlap:
                datasets.appendleft([label])
                self._full_axis.appendleft(global_axis[i])
                self._bag.appendleft(problem)
            else:
                datasets.append([label])
                self._full_axis.append(global_axis[i])
                self._bag.append(problem)

    def calculate_matrices(self):
        if self._index_dependent:
            if self._grouped:
                self.calculate_index_dependent_grouped_matrices()
            else:
                self.calculate_index_dependent_ungrouped_matrices()
        else:
            if self._grouped:
                self.calculate_index_independent_grouped_matrices()
            else:
                self.calculate_index_independent_ungrouped_matrices()

    def calculate_index_dependent_grouped_matrices(
        self,
    ) -> Tuple[
        Dict[str, List[List[str]]],
        Dict[str, List[np.ndarray]],
        List[List[str]],
        List[np.ndarray],
    ]:
        if self._parameters is None:
            raise ParameterError

        def calculate_group(
            group: GroupedProblem, descriptors: Dict[str, DatasetDescriptor]
        ) -> Tuple[List[Tuple[LabelAndMatrix, str]], float]:
            result = [
                (
                    _calculate_matrix(
                        self._model.matrix,
                        descriptors[problem.label],
                        problem.axis,
                        {},
                        index=problem.index,
                    ),
                    problem.label,
                )
                for problem in group.descriptor
            ]
            return result, group.descriptor[0].index

        def reduce_and_combine_matrices(
            results: Tuple[List[Tuple[LabelAndMatrix, str]], float],
        ) -> LabelAndMatrix:
            index_results, index = results
            constraint_labels_and_matrices = list(
                map(
                    lambda result: _reduce_matrix(
                        self._model, result[1], self.parameters, result[0], index
                    ),
                    index_results,
                )
            )
            clp, matrix = _combine_matrices(constraint_labels_and_matrices)
            return LabelAndMatrix(clp, matrix)

        results = list(
            map(lambda group: calculate_group(group, self._filled_dataset_descriptors), self._bag)
        )

        clp_labels = list(map(lambda result: [r[0].clp_label for r in result[0]], results))
        matrices = list(map(lambda result: [r[0].matrix for r in result[0]], results))

        self._clp_labels = {}
        self._matrices = {}

        for i, grouped_problem in enumerate(self._bag):
            for j, descriptor in enumerate(grouped_problem.descriptor):
                if descriptor.label not in self._clp_labels:
                    self._clp_labels[descriptor.label] = []
                    self._matrices[descriptor.label] = []
                self._clp_labels[descriptor.label].append(clp_labels[i][j])
                self._matrices[descriptor.label].append(matrices[i][j])

        reduced_results = list(map(reduce_and_combine_matrices, results))
        self._reduced_clp_labels = list(map(lambda result: result.clp_label, reduced_results))
        self._reduced_matrices = list(map(lambda result: result.matrix, reduced_results))
        return self._clp_labels, self._matrices, self._reduced_clp_labels, self._reduced_matrices

    def calculate_index_dependent_ungrouped_matrices(
        self,
    ) -> Tuple[
        Dict[str, List[List[str]]],
        Dict[str, List[np.ndarray]],
        Dict[str, List[str]],
        Dict[str, List[np.ndarray]],
    ]:
        if self._parameters is None:
            raise ParameterError

        self._clp_labels = {}
        self._matrices = {}
        self._reduced_clp_labels = {}
        self._reduced_matrices = {}

        for label, problem in self.bag.items():
            self._clp_labels[label] = []
            self._matrices[label] = []
            self._reduced_clp_labels[label] = []
            self._reduced_matrices[label] = []
            descriptor = self._filled_dataset_descriptors[label]

            for index in problem.global_axis:
                result = _calculate_matrix(
                    self._model.matrix,
                    descriptor,
                    problem.model_axis,
                    {},
                    index=index,
                )

                self._clp_labels[label].append(result.clp_label)
                self._matrices[label].append(result.matrix)
                reduced_labels_and_matrix = _reduce_matrix(
                    self._model, label, self._parameters, result, index
                )
                self._reduced_clp_labels[label].append(reduced_labels_and_matrix.clp_label)
                self._reduced_matrices[label].append(reduced_labels_and_matrix.matrix)

        return self._clp_labels, self._matrices, self._reduced_clp_labels, self._reduced_matrices

    def calculate_index_independent_grouped_matrices(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray], Dict[str, LabelAndMatrix],]:
        # We just need to create groups from the ungrouped matrices
        self.calculate_index_independent_ungrouped_matrices()
        for group_label, group in self._groups.items():
            if group_label not in self._matrices:
                reduced_labels_and_matrix = _combine_matrices(
                    [
                        LabelAndMatrix(
                            self._reduced_clp_labels[label], self._reduced_matrices[label]
                        )
                        for label in group
                    ]
                )
                self._reduced_clp_labels[group_label] = reduced_labels_and_matrix.clp_label
                self._reduced_matrices[group_label] = reduced_labels_and_matrix.matrix

        return self._clp_labels, self._matrices, self._reduced_clp_labels, self._reduced_matrices

    def calculate_index_independent_ungrouped_matrices(
        self,
    ) -> Tuple[
        Dict[str, List[str]],
        Dict[str, np.ndarray],
        Dict[str, List[str]],
        Dict[str, np.ndarray],
    ]:
        if self._parameters is None:
            raise ParameterError

        self._clp_labels = {}
        self._matrices = {}
        self._reduced_clp_labels = {}
        self._reduced_matrices = {}

        for label, descriptor in self._filled_dataset_descriptors.items():
            axis = self._data[label].coords[self._model_dimension].values
            result = _calculate_matrix(
                self._model.matrix,
                descriptor,
                axis,
                {},
            )

            self._clp_labels[label] = result.clp_label
            self._matrices[label] = result.matrix
            reduced_result = _reduce_matrix(self._model, label, self._parameters, result, None)
            self._reduced_clp_labels[label] = reduced_result.clp_label
            self._reduced_matrices[label] = reduced_result.matrix

        return self._clp_labels, self._matrices, self._reduced_clp_labels, self._reduced_matrices

    def calculate_residual(self):
        if self._index_dependent:
            if self._grouped:
                self.calculate_index_dependent_grouped_residual()
            else:
                self.calculate_index_dependent_ungrouped_residual()
        else:
            if self._grouped:
                self.calculate_index_independent_grouped_residual()
            else:
                self.calculate_index_independent_ungrouped_residual()

    def calculate_index_dependent_grouped_residual(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray],]:
        def residual_function(
            problem: GroupedProblem, matrix: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

            matrix = matrix.copy()
            for i in range(matrix.shape[1]):
                matrix[:, i] *= problem.weight
            data = problem.data
            if problem.has_scaling:
                for i, descriptor in enumerate(problem.descriptor):
                    label = descriptor.label
                    if self.filled_dataset_descriptors[label] is not None:
                        start = sum(problem.data_sizes[0:i])
                        end = start + problem.data_sizes[i]
                        matrix[start:end, :] *= self.filled_dataset_descriptors[label].scale

            clp, residual = self._residual_function(matrix, data)
            return clp, residual, residual / problem.weight

        results = list(map(residual_function, self.bag, self.reduced_matrices))

        self._weighted_residuals = list(map(lambda result: result[1], results))
        self._residuals = list(map(lambda result: result[2], results))

        reduced_clps = list(map(lambda result: result[0], results))
        self._ungroup_clps(reduced_clps)

        return self._reduced_clps, self._clps, self._weighted_residuals, self._residuals

    def calculate_index_dependent_ungrouped_residual(
        self,
    ) -> Tuple[
        Dict[str, List[np.ndarray]],
        Dict[str, List[np.ndarray]],
        Dict[str, List[np.ndarray]],
        Dict[str, List[np.ndarray]],
    ]:

        self._reduced_clps = {}
        self._weighted_residuals = {}
        self._residuals = {}

        for label, problem in self.bag.items():
            self._reduced_clps[label] = []
            self._residuals[label] = []
            self._weighted_residuals[label] = []
            data = problem.data
            for i in range(len(problem.global_axis)):
                matrix_at_index = self.reduced_matrices[label][i]

                if problem.dataset.scale is not None:
                    matrix_at_index *= self.filled_dataset_descriptors[label].scale
                if problem.weight is not None:
                    matrix_at_index = matrix_at_index.copy()
                    for j in range(matrix_at_index.shape[1]):
                        matrix_at_index[:, j] *= problem.weight.isel({self._global_dimension: i})
                clp, residual = self._residual_function(
                    matrix_at_index, data.isel({self._global_dimension: i}).values
                )

                self._reduced_clps[label].append(clp)
                self._weighted_residuals[label].append(residual)
                if problem.weight is not None:
                    self._residuals[label].append(
                        residual / problem.weight.isel({self._global_dimension: i})
                    )
                else:
                    self._residuals[label].append(residual)

        self._clps = (
            self.model.retrieve_clp_function(
                self.parameters,
                self.clp_labels,
                self.reduced_clp_labels,
                self.reduced_clps,
                self.data,
            )
            if callable(self.model.retrieve_clp_function)
            else self.reduced_clps
        )

        return self._reduced_clps, self._clps, self._weighted_residuals, self._residuals

    def calculate_index_independent_grouped_residual(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray],]:
        def residual_function(problem: GroupedProblem):
            matrix = self.reduced_matrices[problem.group].copy()
            for i in range(matrix.shape[1]):
                matrix[:, i] *= problem.weight
            data = problem.data
            if problem.has_scaling:
                for i, descriptor in enumerate(problem.descriptor):
                    label = descriptor.label
                    if self.filled_dataset_descriptors[label] is not None:
                        start = sum(problem.data_sizes[0:i])
                        end = start + problem.data_sizes[i]
                        matrix[start:end, :] *= self.filled_dataset_descriptors[label].scale
            clp, residual = self._residual_function(matrix, data)
            return clp, residual, residual / problem.weight

        results = list(map(residual_function, self.bag))

        self._weighted_residuals = list(map(lambda result: result[1], results))
        self._residuals = list(map(lambda result: result[2], results))

        reduced_clps = list(map(lambda result: result[0], results))
        self._ungroup_clps(reduced_clps)

        return self._reduced_clps, self._clps, self._weighted_residuals, self._residuals

    def calculate_index_independent_ungrouped_residual(
        self,
    ) -> Tuple[
        Dict[str, List[np.ndarray]],
        Dict[str, List[np.ndarray]],
        Dict[str, List[np.ndarray]],
        Dict[str, List[np.ndarray]],
    ]:

        self._clps = {}
        self._reduced_clps = {}
        self._weighted_residuals = {}
        self._residuals = {}
        for label, problem in self.bag.items():

            self._clps[label] = []
            self._reduced_clps[label] = []
            self._weighted_residuals[label] = []
            self._residuals[label] = []
            data = problem.data

            for i in range(len(problem.global_axis)):
                matrix = self.reduced_matrices[label].copy()  # TODO: .copy() or not
                if problem.dataset.scale is not None:
                    matrix *= self.filled_dataset_descriptors[label].scale

                if problem.weight is not None:
                    for j in range(matrix.shape[1]):
                        matrix[:, j] *= problem.weight.isel({self._global_dimension: i}).values

                clp, residual = self._residual_function(
                    matrix, data.isel({self._global_dimension: i}).values
                )
                self._reduced_clps[label].append(clp)
                self._weighted_residuals[label].append(residual)
                if problem.weight is not None:
                    self._residuals[label].append(
                        residual / problem.weight.isel({self._global_dimension: i})
                    )
                else:
                    self._residuals[label].append(residual)

        self._clps = (
            self.model.retrieve_clp_function(
                self.parameters,
                self.clp_labels,
                self.reduced_clp_labels,
                self.reduced_clps,
                self.data,
            )
            if callable(self.model.retrieve_clp_function)
            else self._reduced_clps
        )

        return self._reduced_clps, self._clps, self._weighted_residuals, self._residuals

    def _ungroup_clps(self, reduced_clps: np.ndarray):
        reduced_clp_labels = self.reduced_clp_labels
        self._reduced_clp_labels = {}
        self._reduced_clps = {}
        for label, clp_labels in self.clp_labels.items():

            # find offset in the full axis
            offset = _find_closest_index(
                self.data[label].coords[self._global_dimension][0].values, self._full_axis
            )

            self._reduced_clp_labels[label] = []
            self._reduced_clps[label] = []
            for i in range(self.data[label].coords[self._global_dimension].size):
                group_label = self.bag[i].group
                dataset_clp_labels = clp_labels[i] if self._index_dependent else clp_labels
                index_clp_labels = (
                    reduced_clp_labels[i + offset]
                    if self._index_dependent
                    else reduced_clp_labels[group_label]
                )
                self._reduced_clp_labels[label].append(
                    [
                        clp_label
                        for clp_label in dataset_clp_labels
                        if clp_label in index_clp_labels
                    ]
                )

                mask = [
                    clp_label in self._reduced_clp_labels[label][i]
                    for clp_label in index_clp_labels
                ]
                self._reduced_clps[label].append(reduced_clps[i + offset][mask])
        self._clps = (
            self.model.retrieve_clp_function(
                self.parameters,
                self.clp_labels,
                self.reduced_clp_labels,
                self.reduced_clps,
                self.data,
            )
            if callable(self.model.retrieve_clp_function)
            else self._reduced_clps
        )

    def calculate_additional_penalty(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
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
        self, copy: bool = True, history_index: int = None
    ) -> Dict[str, xr.Dataset]:

        if history_index is not None and history_index != -1:
            self.parameters = self.parameter_history[history_index]
        result_data = {label: self._create_result_dataset(label, copy=copy) for label in self.data}

        if callable(self.model.finalize_data):
            self.model.finalize_data(self, result_data)

        return result_data

    def _create_result_dataset(self, label: str, copy: bool = True) -> xr.Dataset:
        dataset = self.data[label]
        if copy:
            dataset = dataset.copy()
        if self.grouped:
            if self.index_dependent:
                dataset = self._create_index_dependent_grouped_result_dataset(label, dataset)
            else:
                dataset = self._create_index_independent_grouped_result_dataset(label, dataset)
        else:
            if self.index_dependent:
                dataset = self._create_index_dependent_ungrouped_result_dataset(label, dataset)
            else:
                dataset = self._create_index_independent_ungrouped_result_dataset(label, dataset)

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

    def _create_index_dependent_grouped_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:

        for index, grouped_problem in enumerate(self.bag):

            if label in grouped_problem.group:
                group_index = [
                    descriptor.label for descriptor in grouped_problem.descriptor
                ].index(label)
                global_index = grouped_problem.descriptor[group_index].index

                self._add_grouped_residual_to_dataset(
                    dataset, grouped_problem, index, group_index, global_index
                )

        # we assume that the labels are the same, this might not be true in
        # future models
        dataset.coords["clp_label"] = self.clp_labels[label][0]
        dataset["matrix"] = (
            (
                (self._global_dimension),
                (self._model_dimension),
                ("clp_label"),
            ),
            self.matrices[label],
        )
        dataset["clp"] = (
            (
                (self._global_dimension),
                ("clp_label"),
            ),
            self.clps[label],
        )

        return dataset

    def _create_index_independent_grouped_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:

        self._add_index_independent_matrix_to_dataset(label, dataset)

        for index, grouped_problem in enumerate(self.bag):

            if label in grouped_problem.group:
                group_index = [
                    descriptor.label for descriptor in grouped_problem.descriptor
                ].index(label)
                global_index = grouped_problem.descriptor[group_index].index

                self._add_grouped_residual_to_dataset(
                    dataset, grouped_problem, index, group_index, global_index
                )

        dataset["clp"] = (
            (
                (self._global_dimension),
                ("clp_label"),
            ),
            self.clps[label],
        )

        return dataset

    def _create_index_dependent_ungrouped_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:

        self._add_index_dependent_ungrouped_matrix_to_dataset(label, dataset)

        self._add_ungrouped_residual_and_full_clp_to_dataset(label, dataset)

        return dataset

    def _create_index_independent_ungrouped_result_dataset(
        self, label: str, dataset: xr.Dataset
    ) -> xr.Dataset:

        self._add_index_independent_matrix_to_dataset(label, dataset)

        self._add_ungrouped_residual_and_full_clp_to_dataset(label, dataset)

        return dataset

    def _add_index_dependent_ungrouped_matrix_to_dataset(self, label: str, dataset: xr.Dataset):
        # we assume that the labels are the same, this might not be true in
        # future models
        dataset.coords["clp_label"] = self.clp_labels[label][0]

        dataset["matrix"] = (
            (
                (self._global_dimension),
                (self._model_dimension),
                ("clp_label"),
            ),
            np.asarray(self.matrices[label]),
        )

    def _add_index_independent_matrix_to_dataset(self, label: str, dataset: xr.Dataset):
        dataset.coords["clp_label"] = self.clp_labels[label]
        dataset["matrix"] = (
            (
                (self._model_dimension),
                ("clp_label"),
            ),
            np.asarray(self.matrices[label]),
        )

    def _add_grouped_residual_to_dataset(
        self,
        dataset: xr.Dataset,
        grouped_problem: GroupedProblem,
        index: int,
        group_index: int,
        global_index: int,
    ):
        if "residual" not in dataset:
            dim1 = dataset.coords[self._model_dimension].size
            dim2 = dataset.coords[self._global_dimension].size
            dataset["weighted_residual"] = (
                (self._model_dimension, self._global_dimension),
                np.zeros((dim1, dim2), dtype=np.float64),
            )
            dataset["residual"] = (
                (self._model_dimension, self._global_dimension),
                np.zeros((dim1, dim2), dtype=np.float64),
            )

        start = sum(
            self.data[grouped_problem.descriptor[i].label].coords[self._model_dimension].size
            for i in range(group_index)
        )

        end = start + dataset.coords[self._model_dimension].size
        dataset.weighted_residual.loc[
            {self._global_dimension: global_index}
        ] = self.weighted_residuals[index][start:end]
        dataset.residual.loc[{self._global_dimension: global_index}] = self.residuals[index][
            start:end
        ]

    def _add_ungrouped_residual_and_full_clp_to_dataset(self, label: str, dataset: xr.Dataset):
        dataset["clp"] = (
            (
                (self._global_dimension),
                ("clp_label"),
            ),
            np.asarray(self.clps[label]),
        )
        dataset["weighted_residual"] = (
            (
                (self._model_dimension),
                (self._global_dimension),
            ),
            np.transpose(np.asarray(self.weighted_residuals[label])),
        )
        dataset["residual"] = (
            (
                (self._model_dimension),
                (self._global_dimension),
            ),
            np.transpose(np.asarray(self.residuals[label])),
        )

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


def _find_overlap(a, b, rtol=1e-05, atol=1e-08):
    ovr_a = []
    ovr_b = []
    start_b = 0
    for i, ai in enumerate(a):
        for j, bj in itertools.islice(enumerate(b), start_b, None):
            if np.isclose(ai, bj, rtol=rtol, atol=atol, equal_nan=False):
                ovr_a.append(i)
                ovr_b.append(j)
            elif bj > ai:  # (more than tolerance)
                break  # all the rest will be farther away
            else:  # bj < ai (more than tolerance)
                start_b += 1  # ignore further tests of this item
    return (ovr_a, ovr_b)


def _calculate_matrix(
    matrix_function: Callable,
    dataset_descriptor: DatasetDescriptor,
    axis: np.ndarray,
    extra: Dict,
    index: float = None,
) -> LabelAndMatrix:
    args = {
        "dataset_descriptor": dataset_descriptor,
        "axis": axis,
    }
    for k, v in extra:
        args[k] = v
    if index is not None:
        args["index"] = index
    clp_label, matrix = matrix_function(**args)
    return LabelAndMatrix(clp_label, matrix)


def _reduce_matrix(
    model: Model,
    label: str,
    parameters: ParameterGroup,
    result: LabelAndMatrix,
    index: float,
) -> LabelAndMatrix:
    clp_labels = result.clp_label.copy()
    if callable(model.has_matrix_constraints_function) and model.has_matrix_constraints_function():
        clp_label, matrix = model.constrain_matrix_function(
            label, parameters, clp_labels, result.matrix, index
        )
        return LabelAndMatrix(clp_label, matrix)
    return LabelAndMatrix(clp_labels, result.matrix)


def _combine_matrices(labels_and_matrices: List[LabelAndMatrix]) -> LabelAndMatrix:
    masks = []
    full_clp_labels = None
    sizes = []
    for label_and_matrix in labels_and_matrices:
        (clp_label, matrix) = label_and_matrix
        sizes.append(matrix.shape[0])
        if full_clp_labels is None:
            full_clp_labels = clp_label
            masks.append([i for i, _ in enumerate(clp_label)])
        else:
            mask = []
            for c in clp_label:
                if c not in full_clp_labels:
                    full_clp_labels.append(c)
                mask.append(full_clp_labels.index(c))
            masks.append(mask)
    dim1 = np.sum(sizes)
    dim2 = len(full_clp_labels)
    full_matrix = np.zeros((dim1, dim2), dtype=np.float64)
    start = 0
    for i, m in enumerate(labels_and_matrices):
        end = start + sizes[i]
        full_matrix[start:end, masks[i]] = m[1]
        start = end

    return LabelAndMatrix(full_clp_labels, full_matrix)


def _find_closest_index(index: float, axis: np.ndarray):
    return np.abs(axis - index).argmin()
