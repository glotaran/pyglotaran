import collections
import itertools
import typing

import numpy as np
import xarray as xr

from glotaran.analysis.nnls import residual_nnls
from glotaran.analysis.variable_projection import residual_variable_projection
from glotaran.model import DatasetDescriptor
from glotaran.model import Model
from glotaran.parameter import ParameterGroup

from .scheme import Scheme

ParameterError = ValueError("Parameter not initialized")

ProblemDescriptor = collections.namedtuple(
    "ProblemDescriptor", "dataset data model_axis global_axis weight"
)
GroupedProblem = collections.namedtuple("GroupedProblem", "data weight group descriptor")
GroupedProblemDescriptor = collections.namedtuple("ProblemDescriptor", "dataset index axis")

UngroupedBag = typing.Dict[str, ProblemDescriptor]
GroupedBag = typing.Deque[GroupedProblem]


LabelAndMatrix = collections.namedtuple("LabelAndMatrix", "clp_label matrix")
LabelAndMatrixAndData = collections.namedtuple("LabelAndMatrixAndData", "label_matrix data")


class Problem:
    def __init__(self, scheme: Scheme):

        self._scheme = scheme

        self._model = scheme.model
        self._global_dimension = scheme.model.global_dimension
        self._model_dimension = scheme.model.model_dimension
        self._data = scheme.data

        self._index_dependent = scheme.model.index_dependent()
        self._grouped = scheme.model.grouped()

        self._parameter = None
        self._filled_dataset_descriptors = None

        self._clp_labels = None
        self._matrices = None
        self._reduced_clp_labels = None
        self._reduced_matrices = None
        self._reduced_clps = None
        self._full_clps = None
        self._weighted_residuals = None
        self._residuals = None
        self._additional_penalty = None
        self._full_penalty = None

        self._bag = None
        self._groups = None

        self._residual_function = residual_nnls if scheme.nnls else residual_variable_projection

        self.parameter = scheme.parameter

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

        The model is a subclass of :class:`glotaran.model.Model` decorated with the `@model` decorator :class:`glotaran.model.model_decorator.model`
        For an example implementation see e.g. :class:`glotaran.builtin.models.kinetic_spectrum`

        Returns:
            Model: A subclass of :class:`glotaran.model.Model`
                The model must be decorated with the `@model` decorator :class:`glotaran.model.model_decorator.model`
        """
        return self._model

    @property
    def parameter(self) -> ParameterGroup:
        return self._parameter

    @property
    def grouped(self) -> bool:
        return self._grouped

    @property
    def index_dependent(self) -> bool:
        return self._index_dependent

    @property
    def filled_dataset_descriptors(self) -> typing.Dict[str, DatasetDescriptor]:
        return self._filled_dataset_descriptors

    @property
    def bag(self) -> typing.Union[UngroupedBag, GroupedBag]:
        if not self._bag:
            self._init_bag()
        return self._bag

    @property
    def groups(self) -> typing.Dict[str, typing.List[str]]:
        if not self._groups and self._grouped:
            self._init_bag()
        return self._groups

    @property
    def clp_labels(
        self,
    ) -> typing.Union[
        typing.Dict[str, typing.List[str]],
        typing.Dict[str, typing.List[typing.List[str]]],
        typing.List[typing.List[typing.List[str]]],
    ]:
        if self._clp_labels is None:
            self.calculate_matrices()
        return self._clp_labels

    @property
    def matrices(
        self,
    ) -> typing.Union[
        typing.Dict[str, np.ndarray],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.List[typing.List[np.ndarray]],
    ]:
        if self._matrices is None:
            self.calculate_matrices()
        return self._matrices

    @property
    def reduced_clp_labels(
        self,
    ) -> typing.Union[
        typing.Dict[str, typing.List[str]],
        typing.Dict[str, typing.List[typing.List[str]]],
        typing.List[typing.List[typing.List[str]]],
    ]:
        if self._reduced_clp_labels is None:
            self.calculate_matrices()
        return self._reduced_clp_labels

    @property
    def reduced_matrices(
        self,
    ) -> typing.Union[
        typing.Dict[str, np.ndarray],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.List[typing.List[np.ndarray]],
    ]:
        if self._reduced_matrices is None:
            self.calculate_matrices()
        return self._reduced_matrices

    @property
    def reduced_clps(
        self,
    ) -> typing.Union[typing.List[np.ndarray], typing.Dict[str, typing.List[np.ndarray]],]:
        if self._reduced_clps is None:
            self.calculate_residual()
        return self._reduced_clps

    @property
    def full_clps(
        self,
    ) -> typing.Union[typing.List[np.ndarray], typing.Dict[str, typing.List[np.ndarray]],]:
        if self._full_clps is None:
            self.calculate_residual()
        return self._full_clps

    @property
    def weighted_residuals(
        self,
    ) -> typing.Union[typing.List[np.ndarray], typing.Dict[str, typing.List[np.ndarray]],]:
        if self._weighted_residuals is None:
            self.calculate_residual()
        return self._weighted_residuals

    @property
    def residuals(
        self,
    ) -> typing.Union[typing.List[np.ndarray], typing.Dict[str, typing.List[np.ndarray]],]:
        if self._residuals is None:
            self.calculate_residual()
        return self._residuals

    @property
    def additional_penalty(
        self,
    ) -> typing.Union[typing.List[float], typing.Dict[str, typing.List[float]],]:
        if self._additional_penalty is None:
            self.calculate_residual()
        return self._additional_penalty

    @property
    def full_penalty(self) -> np.ndarray:
        if self._full_penalty is None:
            residuals = self._weighted_residuals
            additional_penalty = self.additional_penalty
            if not self.grouped:
                residuals = [np.concatenate(residuals[label]) for label in residuals]
                additional_penalty = [additional_penalty[label] for label in additional_penalty]

            self._full_penalty = np.concatenate(residuals + additional_penalty)
        return self._full_penalty

    @parameter.setter
    def parameter(self, parameter: ParameterGroup):
        self._parameter = parameter
        self._filled_dataset_descriptors = {
            label: descriptor.fill(self._model, self._parameter)
            for label, descriptor in self._model.dataset.items()
        }
        self._reset_results()

    def _reset_results(self):
        self._clp_labels = None
        self._matrices = None
        self._reduced_clp_labels = None
        self._reduced_matrices = None
        self._reduced_clps = None
        self._full_clps = None
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
            self._bag[label] = ProblemDescriptor(
                self._scheme.model.dataset[label],
                data,
                dataset.coords[self._model_dimension].values,
                dataset.coords[self._global_dimension].values,
                weight,
            )

    def _init_grouped_bag(self):
        datasets = None
        self._full_axis = None
        for label in self._model.dataset:
            dataset = self._data[label]
            weight = (
                dataset.weight
                if "weight" in dataset
                else xr.DataArray(np.ones_like(dataset.data), coords=dataset.data.coords)
            )
            data = dataset.data * weight
            global_axis = dataset.coords[self._global_dimension].values
            model_axis = dataset.coords[self._model_dimension].values
            if self._bag is None:
                self._bag = collections.deque(
                    GroupedProblem(
                        data.isel({self._global_dimension: i}).values,
                        weight.isel({self._global_dimension: i}).values,
                        label,
                        [GroupedProblemDescriptor(label, value, model_axis)],
                    )
                    for i, value in enumerate(global_axis)
                )
                datasets = collections.deque([label] for _, _ in enumerate(global_axis))
                self._full_axis = collections.deque(global_axis)
            else:
                self._append_to_grouped_bag(label, datasets, global_axis, model_axis, data, weight)
        self._groups = {"".join(d): d for d in datasets if len(d) > 1}

    def _append_to_grouped_bag(
        self,
        label: str,
        datasets: typing.Deque[str],
        global_axis: np.ndarray,
        model_axis: np.ndarray,
        data: xr.DataArray,
        weight: xr.DataArray,
    ):
        i1, i2 = _find_overlap(self._full_axis, global_axis, atol=self._scheme.group_tolerance)

        for i, j in enumerate(i1):
            datasets[j].append(label)
            self._bag[j] = GroupedProblem(
                np.concatenate(
                    [
                        self._bag[j].data,
                        data.isel({self._global_dimension: i2[i]}).values,
                    ]
                ),
                np.concatenate(
                    [
                        self._bag[j].weight,
                        weight.isel({self._global_dimension: i2[i]}).values,
                    ]
                ),
                self._bag[j].group + label,
                self._bag[j].descriptor
                + [GroupedProblemDescriptor(label, global_axis[i2[i]], model_axis)],
            )

        # Add non-overlaping regions
        begin_overlap = i2[0] if len(i2) != 0 else 0
        end_overlap = i2[-1] + 1 if len(i2) != 0 else 0
        for i in itertools.chain(range(begin_overlap), range(end_overlap, len(global_axis))):
            problem = GroupedProblem(
                data.isel({self._global_dimension: i}).values,
                weight.isel({self._global_dimension: i}).values,
                [GroupedProblemDescriptor(label, global_axis[i], model_axis)],
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
    ) -> typing.Tuple[
        typing.List[typing.List[typing.List[str]]],
        typing.List[typing.List[np.ndarray]],
        typing.List[typing.List[typing.List[str]]],
        typing.List[typing.List[np.ndarray]],
    ]:
        if self._parameter is None:
            raise ParameterError

        def calculate_group(
            group: GroupedProblem, descriptors: typing.Dict[str, DatasetDescriptor]
        ) -> typing.Tuple[typing.List[LabelAndMatrix], typing.Var]:
            result = [
                _calculate_matrix(
                    self._model.matrix,
                    descriptors[problem.dataset],
                    problem.axis,
                    {},
                    index=problem.index,
                )
                for problem in group.descriptor
            ]
            return result, group.descriptor[0].index

        def get_clp(
            result: typing.Tuple[typing.List[LabelAndMatrix], typing.Var]
        ) -> typing.List[typing.List[str]]:
            return [d.clp_label for d in result[0]]

        def get_matrices(
            result: typing.Tuple[typing.List[LabelAndMatrix], typing.Var]
        ) -> typing.List[np.ndarray]:
            return [d.matrix for d in result[0]]

        def reduce_and_combine_matrices(
            parameter: ParameterGroup,
            result: typing.Tuple[typing.List[LabelAndMatrix], typing.Var],
        ) -> LabelAndMatrix:
            labels_and_matrices, index = result
            constraint_labels_and_matrices = list(
                map(
                    lambda label_and_matrix: _reduce_matrix(
                        self._model, parameter, label_and_matrix, index
                    ),
                    labels_and_matrices,
                )
            )
            clp, matrix = _combine_matrices(constraint_labels_and_matrices)
            return LabelAndMatrix(clp, matrix)

        results = list(
            map(lambda group: calculate_group(group, self._filled_dataset_descriptors), self._bag)
        )
        self._clp_labels = list(map(get_clp, results))
        self._matrices = list(map(get_matrices, results))
        reduced_results = list(map(reduce_and_combine_matrices, results))
        self._reduced_clp_labels = list(map(get_clp, reduced_results))
        self._reduced_matrices = list(map(get_matrices, reduced_results))
        return self.clp_labels, self._matrices, self._reduced_clp_labels, self._reduced_matrices

    def calculate_index_dependent_ungrouped_matrices(
        self,
    ) -> typing.Tuple[
        typing.Dict[str, typing.List[typing.List[str]]],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.Dict[str, typing.List[typing.List[str]]],
        typing.Dict[str, typing.List[np.ndarray]],
    ]:
        if self._parameter is None:
            raise ParameterError

        self._clp_labels = {}
        self._matrices = {}
        self._reduced_clp_labels = {}
        self._reduced_matrices = {}

        for label, problem in self._bag.items():
            self._clp_labels[label] = []
            self._constraint_labels_and_matrices[label] = []
            self._matrices[label] = []
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
                    self._model, self._parameter, result, index
                )
                self._reduced_clp_labels[label].append(reduced_labels_and_matrix.clp_label)
                self._reduced_matrices[label].append(reduced_labels_and_matrix.matrix)

        return self._clp_labels, self._matrices, self._reduced_clp_labels, self._reduced_matrices

    def calculate_index_independent_grouped_matrices(
        self,
    ) -> typing.Tuple[
        typing.Dict[str, typing.List[str]],
        typing.Dict[str, np.ndarray],
        typing.Dict[str, LabelAndMatrix],
    ]:
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
    ) -> typing.Tuple[
        typing.Dict[str, typing.List[str]],
        typing.Dict[str, np.ndarray],
        typing.Dict[str, typing.List[str]],
        typing.Dict[str, np.ndarray],
    ]:
        if self._parameter is None:
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
            reduced_result = _reduce_matrix(self._model, self._parameter, result, None)
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
    ) -> typing.Tuple[
        typing.List[np.ndarray],
        typing.List[np.ndarray],
        typing.List[np.ndarray],
        typing.List[np.ndarray],
        typing.List[float],
    ]:
        def residual_function(
            problem: GroupedProblem, matrix: np.ndarray
        ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

            matrix = matrix.copy()
            for i in range(matrix.shape[1]):
                matrix[:, i] *= problem.weight
            clp, residual = self._residual_function(matrix, problem.data)
            return clp, residual, residual / problem.weight

        results = list(map(residual_function, self.bag, self.reduced_matrices))

        self._reduced_clps = list(map(lambda result: result[0], results))
        self._weighted_residuals = list(map(lambda result: result[1], results))
        self._residuals = list(map(lambda result: result[2], results))
        self._full_clps = self.model.retrieve_clp_function(
            self.parameter,
            self.clp_labels,
            self.reduced_clp_labels,
            self.reduced_clps,
            self.full_axis,
        )

        self._calculate_additional_grouped_penalty()

        return (
            self._reduced_clps,
            self._full_clps,
            self._weighted_residuals,
            self._residuals,
            self._additional_penalty,
        )

    def calculate_index_dependent_ungrouped_residual(
        self,
    ) -> typing.Tuple[
        typing.Dict[str, typing.List[np.ndarray]],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.Dict[str, typing.List[float]],
    ]:
        self._reduced_clps = {}
        self._weighted_residuals = {}
        self._residuals = {}
        self._additional_penalty = {}
        for label, problem in self.bag.items():
            self._reduced_clps[label] = []
            self._residuals[label] = []
            self._weighted_residuals[label] = []
            for i, index in enumerate(problem.global_axis):
                matrix_at_index = self.reduced_matrices[label][i]
                if problem.weight is not None:
                    matrix_at_index = matrix_at_index.copy()
                    for j in range(matrix_at_index.shape[1]):
                        matrix_at_index[:, j] *= problem.weight.isel({self._global_dimension: i})
                clp, residual = self._residual_function(
                    matrix_at_index, problem.data.isel({self._global_dimension: i}).values
                )
                self._reduced_clps[label].append(clp)
                self._weighted_residuals[label].append(residual)
                if problem.weight is not None:
                    self._residuals[label].append(
                        residual / problem.weight.isel({self._global_dimension: i})
                    )
                else:
                    self._residuals[label].append(residual)

            self._full_clps[label] = self.model.retrieve_clp_function(
                self.parameter,
                self.clp_labels[label],
                self.reduced_clp_labels[label],
                self.reduced_clps[label],
                problem.global_axis,
            )
            self._calculate_additional_ungrouped_penalty(label, problem.global_axis)
        return (
            self._reduced_clps,
            self._full_clps,
            self._weighted_residuals,
            self._residuals,
            self._additional_penalty,
        )

    def calculate_index_independent_grouped_residual(
        self,
    ) -> typing.Tuple[
        typing.List[np.ndarray],
        typing.List[np.ndarray],
        typing.List[np.ndarray],
        typing.List[np.ndarray],
        typing.List[float],
    ]:
        def residual_function(problem: GroupedProblem):
            matrix = self.matrices[problem.group].copy()
            for i in range(matrix.shape[1]):
                matrix[:, i] *= problem.weight
            clp, residual = self._residual_function(matrix, problem.data)
            return clp, residual, residual / problem.weight

        results = list(map(residual_function, self.bag))

        self._reduced_clps = list(map(lambda result: result[0]), results)
        self._weighted_residuals = list(map(lambda result: result[1]), results)
        self._residuals = list(map(lambda result: result[2], results))
        self._full_clps = self.model.retrieve_clp_function(
            self.parameter,
            self.clp_labels,
            self.reduced_clp_labels,
            self.reduced_clps,
            self.full_axis,
        )

        self._calculate_additional_grouped_penalty()

        return (
            self._reduced_clps,
            self._full_clps,
            self._weighted_residuals,
            self._residuals,
            self._additional_penalty,
        )

    def calculate_index_independent_ungrouped_residual(
        self,
    ) -> typing.Tuple[
        typing.Dict[str, typing.List[np.ndarray]],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.Dict[str, typing.List[float]],
    ]:

        self._full_clps = {}
        self._reduced_clps = {}
        self._weighted_residuals = {}
        self._residuals = {}
        self._additional_penalty = {}
        for label, problem in self.bag.items():

            self._full_clps[label] = []
            self._reduced_clps[label] = []
            self._weighted_residuals[label] = []
            self._residuals[label] = []

            for i, index in enumerate(problem.global_axis):
                data = problem.data.isel({self._global_dimension: i}).values
                matrix = self.reduced_matrices[label]

                if problem.weight is not None:
                    matrix = matrix.copy()
                    for j in range(matrix.shape[1]):
                        matrix[:, j] *= problem.weight.isel({problem.global_dimension: i}).values

                clp, residual = self._residual_function(matrix, data)
                self._reduced_clps[label].append(clp)
                self._weighted_residuals[label].append(residual)
                if problem.weight is not None:
                    self._residuals[label].append(
                        residual / problem.weight.isel({self._global_dimension: i})
                    )
                else:
                    self._residuals[label].append(residual)

            self._full_clps[label] = self.model.retrieve_clp_function(
                self.parameter,
                self.clp_labels[label],
                self.reduced_clp_labels[label],
                self.reduced_clps[label],
                problem.global_axis,
            )
            self._calculate_additional_ungrouped_penalty(label, problem.global_axis)
        return (
            self._reduced_clps,
            self._full_clps,
            self._weighted_residuals,
            self._residuals,
            self._additional_penalty,
        )

    def _calculate_additional_grouped_penalty(self):
        if (
            callable(self.model.has_additional_penalty_function)
            and self.model.has_additional_penalty_function()
        ):
            self._additional_penalty = self.model.additional_penalty_function(
                self.parameter, self.full_clp_labels, self.full_clps, self.full_axis
            )
        else:
            self._additional_penalty = []

    def _calculate_additional_ungrouped_penalty(self, label: str, global_axis: np.ndarray):
        if (
            callable(self.model.has_additional_penalty_function)
            and self.model.has_additional_penalty_function()
        ):
            self._additional_penalty[label] = self.model.additional_penalty_function(
                self.parameter, self._clp_labels[label], self._full_clps, global_axis
            )


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
    matrix_function: typing.Callable,
    dataset_descriptor: DatasetDescriptor,
    axis: np.ndarray,
    extra: typing.Dict,
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
    if dataset_descriptor.scale is not None:
        matrix *= dataset_descriptor.scale
    return LabelAndMatrix(clp_label, matrix)


def _reduce_matrix(
    model: Model,
    parameter: ParameterGroup,
    result: LabelAndMatrix,
    index: float,
) -> LabelAndMatrix:
    if callable(model.has_matrix_constraints_function) and model.has_matrix_constraints_function():
        clp_label, matrix = model.constrain_matrix_function(
            parameter, result.clp_label, result.matrix, index
        )
        return LabelAndMatrix(clp_label, matrix)
    return result


def _combine_matrices(labels_and_matrices: typing.List[LabelAndMatrix]):
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
