import collections
import itertools
import typing

import numpy as np
import xarray as xr
from dask import array as da
from dask import bag as db

from .scheme import Scheme

ProblemDescriptor = collections.namedtuple(
    "ProblemDescriptor", "dataset data model_axis global_axis weight"
)
GroupedProblem = collections.namedtuple("GroupedProblem", "data weight descriptor")
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

        self._bag = None
        self._groups = None

    @property
    def scheme(self) -> Scheme:
        return self._scheme

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
        return self._clp_label

    @property
    def matrices(
        self,
    ) -> typing.Union[
        typing.Dict[str, np.ndarray],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.List[typing.List[np.ndarray]],
    ]:
        return self._matrices

    @property
    def constraint_labels_and_matrices(
        self,
    ) -> typing.Union[
        typing.Dict[str, LabelAndMatrix],
        typing.Dict[str, typing.List[LabelAndMatrix]],
        typing.List[LabelAndMatrix],
    ]:
        return self._constraint_labels_and_matrices

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
        full_axis = None
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
                bag = collections.deque(
                    GroupedProblem(
                        data.isel({self._global_dimension: i}).values,
                        weight.isel({self._global_dimension: i}).values,
                        [GroupedProblemDescriptor(label, value, model_axis)],
                    )
                    for i, value in enumerate(global_axis)
                )
                datasets = collections.deque([label] for _, _ in enumerate(global_axis))
                full_axis = collections.deque(global_axis)
            else:
                self._append_to_grouped_bag(
                    label, datasets, full_axis, global_axis, model_axis, data, weight
                )
        self._bag = db.from_sequence(bag)
        self._groups = {"".join(d): d for d in datasets if len(d) > 1}

    def _append_to_grouped_bag(
        self,
        label: str,
        datasets: collections.Deque[str],
        full_axis: collections.Deque[float],
        global_axis: np.ndarray,
        model_axis: np.ndarray,
        data: xr.DataArray,
        weight: xr.DataArray,
    ):
        i1, i2 = _find_overlap(full_axis, global_axis, atol=self._scheme.group_tolerance)

        for i, j in enumerate(i1):
            datasets[j].append(label)
            self._bag[j] = GroupedProblem(
                da.concatenate(
                    [
                        self._bag[j][0],
                        data.isel({self._global_dimension: i2[i]}).values,
                    ]
                ),
                da.concatenate(
                    [
                        self._bag[j][1],
                        weight.isel({self._global_dimension: i2[i]}).values,
                    ]
                ),
                self._bag[j][2]
                + [GroupedProblemDescriptor(label, global_axis[i2[i]], model_axis)],
            )

        # Add non-overlaping regions
        begin_overlap = i2[0] if len(i2) != 0 else 0
        end_overlap = i2[-1] + 1 if len(i2) != 0 else 0
        for i in itertools.chain(range(0, begin_overlap), range(end_overlap, len(global_axis))):
            problem = GroupedProblem(
                data.isel({self._global_dimension: i}).values,
                weight.isel({self._global_dimension: i}).values,
                [GroupedProblemDescriptor(label, global_axis[i], model_axis)],
            )
            if i < end_overlap:
                datasets.appendleft([label])
                full_axis.appendleft(global_axis[i])
                self._bag.appendleft(problem)
            else:
                datasets.append([label])
                full_axis.append(global_axis[i])
                self._bag.append(problem)

    def _calculate_index_independent_ungrouped_matrices(
        self, parameter: ParameterGroup
    ) -> typing.Tuple[
        typing.Dict[str, typing.List[str]],
        typing.Dict[str, np.ndarray],
        typing.Dict[str, LabelAndMatrix],
    ]:

        clp_labels = {}
        matrices = {}
        constraint_labels_and_matrices = {}

        descriptors = self._get_filled_descriptors(parameter)

        for label, descriptor in descriptors.items():
            axis = self._data[label].coords[self._model_dimension].values
            result = _calculate_matrix(
                self._model.matrix,
                descriptor,
                axis,
                {},
            )

            clp_labels[label] = result.clp_label
            matrices[label] = result.matrix
            constraint_labels_and_matrices[label] = _constrain_matrix(
                self._model, parameter, result, None
            )

        return clp_labels, matrices, constraint_labels_and_matrices

    def _calculate_index_independent_grouped_matrices(
        self, parameter: ParameterGroup
    ) -> typing.Tuple[
        typing.Dict[str, typing.List[str]],
        typing.Dict[str, np.ndarray],
        typing.Dict[str, LabelAndMatrix],
    ]:

        # We just need to create groups from the ungrouped matrices
        (
            clp_labels,
            matrices,
            constraint_labels_and_matrices,
        ) = self._calculate_index_independent_ungrouped_matrices(parameter)
        for label, group in self._groups.items():
            if label not in matrices:
                constraint_labels_and_matrices[label] = _combine_matrices(
                    [constraint_labels_and_matrices[label] for label in group]
                )

        return clp_labels, matrices, constraint_labels_and_matrices

    def _calculate_index_dependent_ungrouped_matrix_jobs(
        self, parameter: ParameterGroup
    ) -> typing.Tuple[
        typing.Dict[str, typing.List[typing.List[str]]],
        typing.Dict[str, typing.List[np.ndarray]],
        typing.Dict[str, typing.List[LabelAndMatrix]],
    ]:

        clp_labels = {}
        constraint_labels_and_matrices = {}
        descriptors = self._get_filled_descriptors(parameter)
        matrices = {}

        for label, problem in self._bag.items():

            clp_labels[label] = []
            constraint_labels_and_matrices[label] = []
            descriptor = descriptors[label]
            matrices[label] = []

            for index in problem.global_axis:
                result = _calculate_matrix(
                    self._model.matrix,
                    descriptor,
                    problem.model_axis,
                    {},
                    index=index,
                )

                clp_labels[label].append(result.clp_label)
                matrices[label].append(result.matrix)
                constraint_labels_and_matrices[label].append(
                    _constrain_matrix(self._model, parameter, result, index)
                )

        return clp_labels, matrices, constraint_labels_and_matrices

    def create_index_dependent_grouped_matrix_jobs(
        self, parameter: ParameterGroup
    ) -> typing.Tuple[
        typing.List[typing.List[typing.List[str]]],
        typing.List[typing.List[np.ndarray]],
        typing.List[LabelAndMatrix],
    ]:
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

        def constrain_and_combine_matrices(
            parameter: ParameterGroup,
            result: typing.Tuple[typing.List[LabelAndMatrix], typing.Var],
        ) -> LabelAndMatrix:
            labels_and_matrices, index = result
            constraint_labels_and_matrices = map(
                lambda label_and_matrix: _constrain_matrix(
                    self._model, parameter, label_and_matrix, index
                ),
                labels_and_matrices,
            )
            clp, matrix = _combine_matrices(constraint_labels_and_matrices)
            return LabelAndMatrix(clp, matrix)

        descriptors = self._get_filled_descriptors(parameter)

        results = map(lambda group: calculate_group(group, descriptors), self._bag)
        clp = map(get_clp, results)
        matrices = map(get_matrices, results)
        constraint_labels_and_matrices = map(constrain_and_combine_matrices, results)

        return clp, matrices, constraint_labels_and_matrices

    def _get_filled_descriptors(
        self, parameter: ParameterGroup
    ) -> typing.Dict[str, DatasetDescriptor]:
        return {
            label: descriptor.fill(self._model, parameter)
            for label, descriptor in self._model.dataset.items()
        }


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


def _constrain_matrix(
    model: Model,
    parameter: ParameterGroup,
    result: LabelAndMatrix,
    index: typing.Var,
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
