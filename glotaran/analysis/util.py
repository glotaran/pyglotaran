from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
import xarray as xr

from glotaran.model import DatasetDescriptor


class LabelAndMatrix(NamedTuple):
    clp_label: list[str]
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
    dataset_descriptor: DatasetDescriptor,
    indices: dict[str, int],
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
