from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
from numba.extending import get_cython_function_address

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike
# This is a work around to use scipy.special function with numba
_dble = ctypes.c_double

functype = ctypes.CFUNCTYPE(_dble, _dble)

erf_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1erf")
erfcx_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1erfcx")

erf = functype(erf_addr)
erfcx = functype(erfcx_addr)

SQRT2 = np.sqrt(2)


@nb.jit(nopython=True, parallel=False)
def calculate_decay_matrix_gaussian_irf_on_index(
    matrix: ArrayLike,
    rates: ArrayLike,
    times: ArrayLike,
    centers: ArrayLike,
    widths: ArrayLike,
    scales: ArrayLike,
    backsweep: bool,
    backsweep_period: float | None,
):
    """Calculates a decay matrix with a gaussian irf."""
    for n_i in nb.prange(centers.size):
        center, width, scale = centers[n_i], widths[n_i], scales[n_i]
        for n_r in nb.prange(rates.size):
            r_n = rates[n_r]
            backsweep_valid = backsweep and abs(r_n) * backsweep_period > 0.001
            alpha = (r_n * width) / SQRT2
            for n_t in nb.prange(times.size):
                t_n = times[n_t]
                beta = (t_n - center) / (width * SQRT2)
                thresh = beta - alpha
                if thresh < -1:
                    matrix[n_t, n_r] += scale * 0.5 * erfcx(-thresh) * np.exp(-beta * beta)
                else:
                    matrix[n_t, n_r] += (
                        scale * 0.5 * (1 + erf(thresh)) * np.exp(alpha * (alpha - 2 * beta))
                    )
                if backsweep and backsweep_valid:
                    x1 = np.exp(-r_n * (t_n - center + backsweep_period))
                    x2 = np.exp(-r_n * ((backsweep_period / 2) - (t_n - center)))
                    x3 = np.exp(-r_n * backsweep_period)
                    matrix[n_t, n_r] += scale * (x1 + x2) / (1 - x3)


@nb.jit(nopython=True, parallel=True)
def calculate_decay_matrix_gaussian_irf(
    matrix: ArrayLike,
    rates: ArrayLike,
    times: ArrayLike,
    all_centers: ArrayLike,
    all_widths: ArrayLike,
    scales: ArrayLike,
    backsweep: bool,
    backsweep_period: float | None,
):
    for n_w in nb.prange(all_centers.shape[0]):
        calculate_decay_matrix_gaussian_irf_on_index(
            matrix[n_w],
            rates,
            times,
            all_centers[n_w],
            all_widths[n_w],
            scales,
            backsweep,
            backsweep_period,
        )
