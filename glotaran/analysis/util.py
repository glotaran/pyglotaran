from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np

from glotaran.model import DatasetDescriptor
from glotaran.model import Model
from glotaran.parameter import ParameterGroup


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
    model: Model,
    dataset_descriptor: DatasetDescriptor,
    indices: dict[str, int],
    axis: dict[str, np.ndarray],
) -> LabelAndMatrix:
    clp_labels = None
    matrix = None

    for scale, megacomplex in dataset_descriptor.iterate_megacomplexes():
        this_clp_labels, this_matrix = megacomplex.calculate_matrix(
            model, dataset_descriptor, indices, axis
        )

        if scale is not None:
            this_matrix *= scale

        if matrix is None:
            clp_labels = this_clp_labels
            matrix = this_matrix
        else:
            tmp_clp_labels = clp_labels + [c for c in this_clp_labels if c not in clp_labels]
            tmp_matrix = np.zeros((matrix.shape[0], len(tmp_clp_labels)), dtype=np.float64)
            for idx, label in enumerate(tmp_clp_labels):
                if label in clp_labels:
                    tmp_matrix[:, idx] += matrix[:, clp_labels.index(label)]
                if label in this_clp_labels:
                    tmp_matrix[:, idx] += this_matrix[:, this_clp_labels.index(label)]
            clp_labels = tmp_clp_labels
            matrix = tmp_matrix

    return LabelAndMatrix(clp_labels, matrix)


def reduce_matrix(
    model: Model,
    label: str,
    parameters: ParameterGroup,
    result: LabelAndMatrix,
    index: float | None,
) -> LabelAndMatrix:
    clp_labels = result.clp_label.copy()
    if callable(model.has_matrix_constraints_function) and model.has_matrix_constraints_function():
        clp_label, matrix = model.constrain_matrix_function(
            label, parameters, clp_labels, result.matrix, index
        )
        return LabelAndMatrix(clp_label, matrix)
    return LabelAndMatrix(clp_labels, result.matrix)


def combine_matrices(labels_and_matrices: list[LabelAndMatrix]) -> LabelAndMatrix:
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
