from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import xarray as xr

from glotaran.model import DatasetModel
from glotaran.model import Model
from glotaran.parameter import ParameterGroup


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
    indices: dict[str, int] | None,
    global_model: bool = False,
) -> xr.DataArray:
    matrix = None

    megacomplex_iterator = dataset_model.iterate_megacomplexes

    if global_model:
        megacomplex_iterator = dataset_model.iterate_global_megacomplexes
        dataset_model.swap_dimensions()

    for scale, megacomplex in megacomplex_iterator():
        this_matrix = megacomplex.calculate_matrix(dataset_model, indices)

        if scale is not None:
            this_matrix *= scale

        if matrix is None:
            matrix = this_matrix
        else:
            matrix, this_matrix = xr.align(matrix, this_matrix, join="outer", copy=False)
            matrix = matrix.fillna(0)
            matrix += this_matrix.fillna(0)

    if global_model:
        dataset_model.swap_dimensions()

    return matrix


def reduce_matrix(
    matrix: xr.DataArray,
    model: Model,
    parameters: ParameterGroup,
    model_dimension: str,
    index: Any | None,
) -> xr.DataArray:
    matrix = apply_relations(matrix, model, parameters, model_dimension, index)
    matrix = apply_constraints(matrix, model, index)
    return matrix


def apply_constraints(
    matrix: xr.DataArray,
    model: Model,
    index: Any | None,
) -> xr.DataArray:

    if len(model.constraints) == 0:
        return matrix

    clp_labels = matrix.coords["clp_label"].values
    removed_clp = [
        c.target for c in model.constraints if c.target in clp_labels and c.applies(index)
    ]
    reduced_clp_label = [c for c in clp_labels if c not in removed_clp]

    return matrix.sel({"clp_label": reduced_clp_label})


def apply_relations(
    matrix: xr.DataArray,
    model: Model,
    parameters: ParameterGroup,
    model_dimension: str,
    index: Any | None,
) -> xr.DataArray:

    if len(model.relations) == 0:
        return matrix

    clp_labels = list(matrix.coords["clp_label"].values)
    relation_matrix = np.diagflat([1.0 for _ in clp_labels])

    idx_to_delete = []
    for relation in model.relations:
        if relation.target in clp_labels and relation.applies(index):

            if relation.source not in clp_labels:
                continue

            relation = relation.fill(model, parameters)
            source_idx = clp_labels.index(relation.source)
            target_idx = clp_labels.index(relation.target)
            relation_matrix[target_idx, source_idx] = relation.parameter
            idx_to_delete.append(target_idx)

    reduced_clp_label = [label for i, label in enumerate(clp_labels) if i not in idx_to_delete]
    relation_matrix = np.delete(relation_matrix, idx_to_delete, axis=1)
    return xr.DataArray(
        matrix.values @ relation_matrix,
        dims=matrix.dims,
        coords={
            "clp_label": reduced_clp_label,
            model_dimension: matrix.coords[model_dimension],
        },
    )


def retrieve_clps(
    model: Model,
    parameters: ParameterGroup,
    clp_labels: xr.DataArray,
    reduced_clp_labels: xr.DataArray,
    reduced_clps: xr.DataArray,
    index: Any | None,
) -> xr.DataArray:
    if len(model.relations) == 0 and len(model.constraints) == 0:
        return reduced_clps

    clps = np.zeros(clp_labels.size)

    for i, label in enumerate(reduced_clp_labels):
        idx = np.where(clp_labels == label)[0]
        clps[idx] = reduced_clps[i]

    for relation in model.relations:
        relation = relation.fill(model, parameters)
        if (
            relation.target in clp_labels
            and relation.applies(index)
            and relation.source in clp_labels
        ):
            source_idx = np.where(clp_labels == relation.source)[0]
            target_idx = np.where(clp_labels == relation.target)[0]
            clps[target_idx] = relation.parameter * clps[source_idx]
    return clps


def calculate_clp_penalties(
    model: Model,
    parameters: ParameterGroup,
    clp_labels: list[list[str]],
    clps: list[np.ndarray],
    global_axis: np.ndarray,
) -> np.ndarray:

    penalties = []
    for penalty in model.clp_area_penalties:
        penalty = penalty.fill(model, parameters)
        source_area = _get_area(
            penalty.source,
            clp_labels,
            clps,
            penalty.source_intervals,
            global_axis,
        )

        target_area = _get_area(
            penalty.target,
            clp_labels,
            clps,
            penalty.target_intervals,
            global_axis,
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
) -> np.ndarray:
    area = []

    for interval in intervals:
        if interval[0] > global_axis[-1]:
            continue

        print("P", interval)
        print("D", global_axis)
        start_idx, end_idx = get_idx_from_interval(interval, global_axis)
        print("Z", start_idx, end_idx)
        for i in range(start_idx, end_idx + 1):
            index_clp_labels = clp_labels[i]
            if clp_label in index_clp_labels:
                area.append(clps[i][np.where(index_clp_labels == clp_label)[0]])

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
