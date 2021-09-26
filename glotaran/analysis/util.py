from __future__ import annotations

import itertools
from typing import Any
from typing import NamedTuple

import numba as nb
import numpy as np
import xarray as xr

from glotaran.model import DatasetModel
from glotaran.model import Model
from glotaran.parameter import ParameterGroup


class CalculatedMatrix(NamedTuple):
    clp_labels: list[str]
    matrix: np.ndarray


def find_overlap(a, b, rtol=1e-05, atol=1e-08):
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


def find_closest_index(index: float, axis: np.ndarray):
    return np.abs(axis - index).argmin()


def get_min_max_from_interval(interval, axis):
    minimum = np.abs(axis.values - interval[0]).argmin() if not np.isinf(interval[0]) else 0
    maximum = (
        np.abs(axis.values - interval[1]).argmin() + 1 if not np.isinf(interval[1]) else axis.size
    )
    return slice(minimum, maximum)


def calculate_matrix(
    dataset_model: DatasetModel,
    indices: dict[str, int],
    as_global_model: bool = False,
) -> CalculatedMatrix:

    clp_labels = None
    matrix = None

    megacomplex_iterator = dataset_model.iterate_megacomplexes

    if as_global_model:
        megacomplex_iterator = dataset_model.iterate_global_megacomplexes
        dataset_model.swap_dimensions()

    for scale, megacomplex in megacomplex_iterator():
        this_clp_labels, this_matrix = megacomplex.calculate_matrix(dataset_model, indices)

        if scale is not None:
            this_matrix *= scale

        if matrix is None:
            clp_labels = this_clp_labels
            matrix = this_matrix
        else:
            clp_labels, matrix = combine_matrix(matrix, this_matrix, clp_labels, this_clp_labels)

    if as_global_model:
        dataset_model.swap_dimensions()

    return CalculatedMatrix(clp_labels, matrix)


def combine_matrix(matrix, this_matrix, clp_labels, this_clp_labels):
    tmp_clp_labels = clp_labels + [c for c in this_clp_labels if c not in clp_labels]
    tmp_matrix = np.zeros((matrix.shape[0], len(tmp_clp_labels)), dtype=np.float64)
    for idx, label in enumerate(tmp_clp_labels):
        if label in clp_labels:
            tmp_matrix[:, idx] += matrix[:, clp_labels.index(label)]
        if label in this_clp_labels:
            tmp_matrix[:, idx] += this_matrix[:, this_clp_labels.index(label)]
    return tmp_clp_labels, tmp_matrix


@nb.jit(nopython=True, parallel=True)
def apply_weight(matrix, weight):
    for i in nb.prange(matrix.shape[1]):
        matrix[:, i] *= weight


def reduce_matrix(
    matrix: CalculatedMatrix,
    model: Model,
    parameters: ParameterGroup,
    index: Any | None,
) -> CalculatedMatrix:
    matrix = apply_relations(matrix, model, parameters, index)
    matrix = apply_constraints(matrix, model, index)
    return matrix


def apply_constraints(
    matrix: CalculatedMatrix,
    model: Model,
    index: Any | None,
) -> CalculatedMatrix:

    if len(model.clp_constraints) == 0:
        return matrix

    clp_labels = matrix.clp_labels
    removed_clp_labels = [
        c.target for c in model.clp_constraints if c.target in clp_labels and c.applies(index)
    ]
    reduced_clp_labels = [c for c in clp_labels if c not in removed_clp_labels]
    mask = [label in reduced_clp_labels for label in clp_labels]
    reduced_matrix = matrix.matrix[:, mask]
    return CalculatedMatrix(reduced_clp_labels, reduced_matrix)


def apply_relations(
    matrix: CalculatedMatrix,
    model: Model,
    parameters: ParameterGroup,
    index: Any | None,
) -> CalculatedMatrix:

    if len(model.clp_relations) == 0:
        return matrix

    clp_labels = matrix.clp_labels
    relation_matrix = np.diagflat([1.0 for _ in clp_labels])

    idx_to_delete = []
    for relation in model.clp_relations:
        if relation.target in clp_labels and relation.applies(index):

            if relation.source not in clp_labels:
                continue

            relation = relation.fill(model, parameters)
            source_idx = clp_labels.index(relation.source)
            target_idx = clp_labels.index(relation.target)
            relation_matrix[target_idx, source_idx] = relation.parameter
            idx_to_delete.append(target_idx)

    reduced_clp_labels = [label for i, label in enumerate(clp_labels) if i not in idx_to_delete]
    relation_matrix = np.delete(relation_matrix, idx_to_delete, axis=1)
    reduced_matrix = matrix.matrix @ relation_matrix
    return CalculatedMatrix(reduced_clp_labels, reduced_matrix)


def retrieve_clps(
    model: Model,
    parameters: ParameterGroup,
    clp_labels: xr.DataArray,
    reduced_clp_labels: xr.DataArray,
    reduced_clps: xr.DataArray,
    index: Any | None,
) -> xr.DataArray:
    if len(model.clp_relations) == 0 and len(model.clp_constraints) == 0:
        return reduced_clps

    clps = np.zeros(len(clp_labels))

    for i, label in enumerate(reduced_clp_labels):
        idx = clp_labels.index(label)
        clps[idx] = reduced_clps[i]

    for relation in model.clp_relations:
        relation = relation.fill(model, parameters)
        if (
            relation.target in clp_labels
            and relation.applies(index)
            and relation.source in clp_labels
        ):
            source_idx = clp_labels.index(relation.source)
            target_idx = clp_labels.index(relation.target)
            clps[target_idx] = relation.parameter * clps[source_idx]
    return clps


def calculate_clp_penalties(
    model: Model,
    parameters: ParameterGroup,
    clp_labels: list[list[str]] | list[str],
    clps: list[np.ndarray],
    global_axis: np.ndarray,
    dataset_models: dict[str, DatasetModel],
) -> np.ndarray:

    # TODO: make a decision on how to handle clp_penalties per dataset
    # 1. sum up contributions per dataset on each dataset_axis (v0.4.1)
    # 2. sum up contributions on the global_axis (future?)

    penalties = []
    for penalty in model.clp_area_penalties:
        penalty = penalty.fill(model, parameters)
        source_area = np.array([])
        target_area = np.array([])
        for _, dataset_model in dataset_models.items():
            dataset_axis = dataset_model.get_global_axis()

            source_area = np.concatenate(
                [
                    source_area,
                    _get_area(
                        penalty.source,
                        clp_labels,
                        clps,
                        penalty.source_intervals,
                        global_axis,
                        dataset_axis,
                    ),
                ]
            )

            target_area = np.concatenate(
                [
                    target_area,
                    _get_area(
                        penalty.target,
                        clp_labels,
                        clps,
                        penalty.target_intervals,
                        global_axis,
                        dataset_axis,
                    ),
                ]
            )
        area_penalty = np.abs(np.sum(source_area) - penalty.parameter * np.sum(target_area))

        penalties.append(area_penalty * penalty.weight)

    return np.asarray(penalties)


def _get_area(
    clp_label: str,
    clp_labels: list[list[str]],
    clps: list[np.ndarray],
    intervals: list[tuple[float, float]],
    global_axis: np.ndarray,
    dataset_axis: np.ndarray,
) -> np.ndarray:
    area = []

    for interval in intervals:
        if interval[0] > global_axis[-1]:
            continue
        bounded_interval = (
            max(interval[0], np.min(dataset_axis)),
            min(interval[1], np.max(dataset_axis)),
        )
        start_idx, end_idx = get_idx_from_interval(bounded_interval, global_axis)
        for i in range(start_idx, end_idx + 1):
            index_clp_labels = clp_labels[i] if isinstance(clp_labels[0], list) else clp_labels
            if clp_label in index_clp_labels:
                area.append(clps[i][index_clp_labels.index(clp_label)])

    return np.asarray(area)  # TODO: normalize for distance on global axis


def get_idx_from_interval(interval: tuple[float, float], axis: np.ndarray) -> tuple[int, int]:
    """Retrieves start and end index of an interval on some axis
    Parameters
    ----------
    interval : A tuple of floats with begin and end of the interval
    axis : Array like object which can be cast to np.array
    Returns
    -------
    start, end : tuple of int
    """
    start = np.abs(axis - interval[0]).argmin() if not np.isinf(interval[0]) else 0
    end = np.abs(axis - interval[1]).argmin() if not np.isinf(interval[1]) else axis.size - 1
    return start, end
