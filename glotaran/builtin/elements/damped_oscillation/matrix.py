from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np
from scipy.special import erf

from glotaran.builtin.items.activation import GaussianActivationParameters  # noqa: TC001

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


@nb.jit(nopython=True, parallel=True)
def calculate_damped_oscillation_matrix_instant_activation(
    matrix: ArrayLike,
    inputs: ArrayLike,
    frequencies: ArrayLike,
    rates: ArrayLike,
    axis: ArrayLike,
):
    for idx, (amplitude, frequency, rate) in enumerate(zip(inputs, frequencies, rates)):
        osc = np.exp(-rate * axis - 1j * frequency * axis)
        matrix[:, idx] = osc.real * amplitude
        matrix[:, idx + rates.size] = osc.imag * amplitude


def calculate_damped_oscillation_matrix_gaussian_activation(
    matrix: ArrayLike,
    inputs: ArrayLike,
    frequencies: ArrayLike,
    rates: ArrayLike,
    parameters: list[list[GaussianActivationParameters]],
    model_axis: ArrayLike,
):
    for i, parameter in enumerate(parameters):
        calculate_damped_oscillation_matrix_gaussian_activation_on_index(
            matrix[i], inputs, frequencies, rates, parameter, model_axis
        )


def calculate_damped_oscillation_matrix_gaussian_activation_on_index(
    matrix: ArrayLike,
    inputs: ArrayLike,
    frequencies: ArrayLike,
    rates: ArrayLike,
    parameters: list[GaussianActivationParameters],
    model_axis: ArrayLike,
):
    scales = 0.0
    for parameter in parameters:
        scales += parameter.scale
        matrix += calculate_damped_oscillation_matrix_gaussian_activation_sin_cos(
            inputs,
            frequencies,
            rates,
            model_axis,
            parameter.center,
            parameter.width,
            parameter.scale,
        )
    matrix /= scales


def calculate_damped_oscillation_matrix_gaussian_activation_sin_cos(
    inputs: ArrayLike,
    frequencies: ArrayLike,
    rates: ArrayLike,
    model_axis: ArrayLike,
    center: float,
    width: float,
    scale: float,
) -> ArrayLike:
    """Calculate the damped oscillation matrix taking into account a gaussian irf

    Parameters
    ----------
    frequencies : np.ndarray
        an array of frequencies in THz, one per oscillation
    rates : np.ndarray
        an array of rates, one per oscillation
    model_axis : np.ndarray
        the model axis (time)
    center : float
        the center of the gaussian IRF
    width : float
        the width (σ) parameter of the the IRF
    shift : float
        a shift parameter per item on the global axis
    scale : float
        the scale parameter to scale the matrix by

    Returns
    -------
    np.ndarray
        An array of the real and imaginary part of the oscillation matrix,
        the shape being (len(model_axis), 2*len(frequencies)), with the first
        half of the second dimension representing the real part,
        and the other the imagine part of the oscillation
    """
    shifted_axis = model_axis - center
    # For calculations using the negative rates we use the time axis
    # from the beginning up to 5 σ from the irf center
    left_shifted_axis_indices = np.where(shifted_axis < 5 * width)[0]
    left_shifted_axis = shifted_axis[left_shifted_axis_indices]
    neg_idx = np.where(rates < 0)[0]
    # For calculations using the positive rates axis we use the time axis
    # from 5 σ before the irf center until the end
    right_shifted_axis_indices = np.where(shifted_axis > -5 * width)[0]
    right_shifted_axis = shifted_axis[right_shifted_axis_indices]
    pos_idx = np.where(rates >= 0)[0]

    d = width**2
    k = rates + 1j * frequencies
    dk = k * d
    sqwidth = np.sqrt(2) * width

    a = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    a[np.ix_(right_shifted_axis_indices, pos_idx)] = np.exp(
        (-1 * right_shifted_axis[:, None] + 0.5 * dk[pos_idx]) * k[pos_idx]
    )

    a[np.ix_(left_shifted_axis_indices, neg_idx)] = np.exp(
        (-1 * left_shifted_axis[:, None] + 0.5 * dk[neg_idx]) * k[neg_idx]
    )

    b = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    b[np.ix_(right_shifted_axis_indices, pos_idx)] = 1 + erf(
        (right_shifted_axis[:, None] - dk[pos_idx]) / sqwidth
    )
    # For negative rates we flip the sign of the `erf` by using `-sqwidth` in lieu of `sqwidth`
    b[np.ix_(left_shifted_axis_indices, neg_idx)] = 1 + erf(
        (left_shifted_axis[:, None] - dk[neg_idx]) / -sqwidth
    )

    osc = a * b * scale

    return np.concatenate(
        (osc.real * inputs[np.newaxis, :], osc.imag * inputs[np.newaxis, :]), axis=1
    )
