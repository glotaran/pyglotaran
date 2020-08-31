"""Functions for calculating conditionally linear parameters and residual with the variable
projection method."""

import typing

import numpy as np
from scipy.linalg import lapack


def residual_variable_projection(
    matrix: np.ndarray, data: np.ndarray
) -> typing.Tuple[typing.List[str], np.ndarray]:
    """Calculates the conditionally linear parameters and residual with the variable projection
    method.

    Parameters
    ----------
    matrix :
        The model matrix.
    data : np.ndarray
        The data to analyze.
    """
    # TODO: Reference Kaufman paper

    # Kaufman Q2 step 3
    qr, tau, _, _ = lapack.dgeqrf(matrix)

    # Kaufman Q2 step 4
    temp, _, _ = lapack.dormqr("L", "T", qr, tau, data, max(1, matrix.shape[1]), overwrite_c=0)

    clp, _ = lapack.dtrtrs(qr, temp)

    for i in range(matrix.shape[1]):
        temp[i] = 0

    # Kaufman Q2 step 5

    residual, _, _ = lapack.dormqr("L", "N", qr, tau, temp, max(1, matrix.shape[1]), overwrite_c=0)
    return clp[: matrix.shape[1]], residual
