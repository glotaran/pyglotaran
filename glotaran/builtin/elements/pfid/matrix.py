from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erf

if TYPE_CHECKING:
    from glotaran.builtin.items.activation import GaussianActivationParameters
    from glotaran.typing.types import ArrayLike


def calculate_pfid_matrix(
    matrix: ArrayLike,
    frequencies: ArrayLike,
    rates: ArrayLike,
    activation_parameters: list[list[GaussianActivationParameters]],
    global_axis: ArrayLike,
    model_axis: ArrayLike,
) -> None:
    """Calculate an index-dependent PFID matrix."""
    for index, parameters in enumerate(activation_parameters):
        for parameter in parameters:
            matrix[index] += calculate_pfid_matrix_gaussian_activation(
                frequencies,
                rates,
                model_axis,
                parameter.center,
                parameter.width,
                parameter.scale,
                global_axis[index],
            )
        matrix[index] /= sum(parameter.scale for parameter in parameters)


def calculate_pfid_matrix_gaussian_activation(
    frequencies: ArrayLike,
    rates: ArrayLike,
    model_axis: ArrayLike,
    center: float,
    width: float,
    scale: float,
    global_axis_value: float,
) -> ArrayLike:
    """Calculate the Gaussian-convolved PFID matrix at one global-axis index."""
    shifted_axis = model_axis - center
    left_shifted_axis_indices = np.where(shifted_axis < 5 * width)[0]
    left_shifted_axis = shifted_axis[left_shifted_axis_indices]
    negative_rate_indices = np.where(rates < 0)[0]

    frequency_difference = (global_axis_value - frequencies) * 0.03 * 2 * np.pi
    width_squared = width**2
    exponent = rates + 1j * frequency_difference
    exponent_width = exponent * width_squared
    sqrt_two_width = np.sqrt(2) * width

    exponential = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    exponential[np.ix_(left_shifted_axis_indices, negative_rate_indices)] = np.exp(
        (-left_shifted_axis[:, None] + 0.5 * exponent_width[negative_rate_indices])
        * exponent[negative_rate_indices]
    )

    convolution = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    convolution[np.ix_(left_shifted_axis_indices, negative_rate_indices)] = 1 + erf(
        (left_shifted_axis[:, None] - exponent_width[negative_rate_indices]) / -sqrt_two_width
    )

    oscillation = -(exponential * convolution) * scale
    return np.concatenate((oscillation.real, oscillation.imag), axis=1)
