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


class Problem:
    def __init__(self, scheme: Scheme):

        self._scheme = scheme

        self._model = scheme.model
        self._global_dimension = scheme.model.global_dimension
        self._model_dimension = scheme.model.model_dimension
        self._data = scheme.data

        self._bag = None
        self._groups = None

    @property
    def scheme(self) -> Scheme:
        return self._scheme

    @property
    def bag(self) -> typing.Union[UngroupedBag, GroupedBag]:
        return self._bag

    @property
    def groups(self) -> typing.Dict[str, typing.List[str]]:
        return self._groups

    def _init_ungrouped_bag(self) -> UngroupedBag:
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
        return self._bag

    def _init_grouped_bag(self) -> typing.Tuple[GroupedBag, typing.Dict[str, typing.List[str]]]:
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
        return self._bag, self._groups

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
