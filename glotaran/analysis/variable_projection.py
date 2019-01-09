"""This package contains functions for variable projection."""
import numpy as np
from scipy.linalg import lapack


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
    temp, _, _ = lapack.dormqr("L", "T", qr, tau, data, max(1, matrix.shape[1]),
                               overwrite_c=0)

    clp, _ = lapack.dtrtrs(qr, temp)

    for i in range(matrix.shape[1]):
        temp[i] = 0

    # Kaufman Q2 step 5

    residual, _, _ = lapack.dormqr("L", "N", qr, tau, temp, max(1, matrix.shape[1]),
                                   overwrite_c=0)
    return clp[:matrix.shape[1]], residual
