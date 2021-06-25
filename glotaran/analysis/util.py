from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import xarray as xr

from glotaran.model import DatasetDescriptor
from glotaran.model import Model
from glotaran.parameter import Parameter


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
    dataset_descriptor: DatasetDescriptor,
    indices: dict[str, int] | None,
) -> xr.DataArray:
    matrix = None

    for scale, megacomplex in dataset_descriptor.iterate_megacomplexes():
        this_matrix = megacomplex.calculate_matrix(dataset_descriptor, indices)

        if scale is not None:
            this_matrix *= scale

        if matrix is None:
            matrix = this_matrix
        else:
            matrix, this_matrix = xr.align(matrix, this_matrix, join="outer", copy=False)
            matrix = matrix.fillna(0)
            matrix += this_matrix.fillna(0)

    return matrix


def reduce_matrix(
    matrix: xr.DataArray,
    model: Model,
    parameters: Parameter,
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
    parameters: Parameter,
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
    parameters: Parameter,
    clp_labels: xr.DataArray,
    reduced_clps: xr.DataArray,
    index: Any | None,
) -> xr.DataArray:
    if len(model.relations) == 0 and len(model.constraints) == 0:
        return reduced_clps

    clps = xr.DataArray(np.zeros((clp_labels.size), dtype=np.float64), coords=[clp_labels])
    clps.loc[{"clp_label": reduced_clps.coords["clp_label"]}] = reduced_clps.values

    print("ret", clps)
    for relation in model.relations:
        relation = relation.fill(model, parameters)
        print("YYY", relation.target, relation.source, relation.parameter)
        if relation.target in clp_labels and relation.applies(index):
            if relation.source not in clp_labels:
                continue
            clps.loc[{"clp_label": relation.target}] = relation.parameter * clps.sel(
                clp_label=relation.source
            )

    return clps
