"""Module for residual calculation with  the variable projection method."""
from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.linalg import lapack

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


def residual_variable_projection(
    matrix: ArrayLike, data: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """Calculate conditionally linear parameters and residual with the variable projection method.

    Parameters
    ----------
    matrix : ArrayLike
        The model matrix.
    data : ArrayLike
        The data to analyze.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        The clps and the residual.
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
