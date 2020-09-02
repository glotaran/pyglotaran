"""Functions for calculating conditionally linear parameters and residual with the non-negative
least-squares method."""

import typing

import numpy as np
from scipy.optimize import nnls


def residual_nnls(
    matrix: np.ndarray, data: np.ndarray
) -> typing.Tuple[typing.List[str], np.ndarray]:
    """Calculate the conditionally linear parameters and residual with the nnls method.

    nnls stands for 'non-negative least-squares'.

    Parameters
    ----------
    matrix :
        The model matrix.
    data : np.ndarray
        The data to analyze.
    """
    clp, _ = nnls(matrix, data)
    residual = data - np.dot(matrix, clp)
    return clp, residual
