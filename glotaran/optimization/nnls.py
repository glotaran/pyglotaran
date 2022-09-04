"""Module for residual calculation with the non-negative least-squares method."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def residual_nnls(
    matrix: np.typing.ArrayLike, data: np.typing.ArrayLike
) -> tuple[np.typing.ArrayLike, np.typing.ArrayLike]:
    """Calculate the conditionally linear parameters and residual with the NNLS method.

    NNLS stands for 'non-negative least-squares'.

    Parameters
    ----------
    matrix : np.typing.ArrayLike
        The model matrix.
    data : np.typing.ArrayLike
        The data to analyze.

    Returns
    -------
    tuple[np.typing.ArrayLike, np.typing.ArrayLike]
        The clps and the residual.
    """
    clp, _ = nnls(matrix, data)
    residual = data - np.dot(matrix, clp)
    return clp, residual
