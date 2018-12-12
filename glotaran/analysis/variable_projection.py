"""This package contains functions for variable projection."""
from typing import Dict, List
import numpy as np
from scipy.linalg import lapack

from glotaran.model.dataset import Dataset
from glotaran.model.parameter_group import ParameterGroup

from .grouping import Group, calculate_group


def residual_variable_projection(matrix: np.array, data: np.array) -> np.array:
    """residul_variable_projection returns the variable projection residual.

    Parameters
    ----------
    matrix: np.array
    data: np.array

    Returns
    -------
    residual: np.array
    """

    # Kaufman Q2 step 3
    qr, tau, _, _ = lapack.dgeqrf(matrix)

    # Kaufman Q2 step 4
    data, _, _ = lapack.dormqr("L", "T", qr, tau, data, max(1, matrix.shape[1]),
                               overwrite_c=0)

    clp, _ = lapack.dtrtrs(qr, data)

    for i in range(matrix.shape[1]):
        data[i] = 0

    # Kaufman Q2 step 5

    data, _, _ = lapack.dormqr("L", "N", qr, tau, data, max(1, matrix.shape[1]),
                               overwrite_c=1)
    return clp[:matrix.shape[1]], data


def qr_coefficents(A, B):

    # Kaufman Q2 step 3

    qr, tau, _, _ = lapack.dgeqrf(A)

    # Kaufman Q2 step 4

    B, _, _ = lapack.dormqr("L", "T", qr, tau, B, max(1, A.shape[1]))

    # Kaufman Q2 step 6

    P, _ = lapack.dtrtrs(qr, B)
    return P


def clp_variable_projection(parameter: ParameterGroup,
                            group: Group,
                            model,  # temp doc fix : 'glotaran.model.BaseModel',
                            data: Dict[str, Dataset],
                            data_group: List[np.ndarray],
                            ) -> np.ndarray:
    """residul_variable_projection returns the variable projection residual.

    Parameters
    ----------
    parameter : ParameterGroup
    group: Dict[any, Tuple[any, DatasetDescriptor]]
    model : glotaran.model.BaseModel
    data : Dict[str, Dataset]
        A dictionary of dataset labels and Datasets.
    data_group : List[np.ndarray]
    **kwargs

    Returns
    -------
    residual: np.ndarray
    """
    res = {}
    for i, matrix, clp, datasets in calculate_group(group, model, parameter, data):
        res[i] = (datasets, clp, qr_coefficents(matrix.T,
                                                data_group[i])[:len(clp)])
    return res
