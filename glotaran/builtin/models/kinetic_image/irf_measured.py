"""This package contains irf items."""

from typing import List
import numba as nb
import numpy as np

from glotaran.model import model_attribute, model_attribute_typed
from glotaran.parameter import Parameter


@model_attribute(
    properties={
        'method': str,
    },
    has_type=True,
)
class IrfMeasured:

    @staticmethod
    def get_implementation():
        if self.method == 'conv1':
            return sel

@nb.jit(nopython=True, parallel=True)
def irf_conv_1(matrix, measured_irf, rates, time):

    time_delta = time[:-1] - time[1:]
    time_delta.append(time_delta[-1])

    for n_r in nb.prange(rates.size):
        r_n = rates[n_r]
        # forward
        for n_t in range(time.size):
            delta_n = time_delta[n_t]
            matrix[n_t, n_r] += 1/r_n * (
                np.exp(-n_t * delta_n * r_n) - np.exp(-(n_t + 1) * delta_n * r_n)
            )
        # backward
        for n_t in reversed(range(time.size)):
            delta_n = time_delta[n_t]
            matrix[n_t, n_r] = 0.5 * (
                measured_irf[0] * matrix[n_t, n_r] + measured_irf[n_t] * matrix[0, n_r]
            ) + 0.25 * matrix[n_t, n_r] * measured_irf[0]

            for i in range(1, n_t):
                matrix[n_t, n_r] += matrix[i, n_r] * measured_irf[n_t-i]
            matrix[n_t, n_r] *= delta_n


@nb.jit(nopython=True, parallel=True)
def irf_conv2(matrix, measured_irf, rates, time):
    time_delta = time[:-1] - time[1:]
    time_delta.append(time_delta[-1])

    for n_r in nb.prange(rates.size):
        r_n = rates[n_r]
        for n_t in range(1, time.size):
            delta_n = time_delta[n_t]
            e = np.exp(-r_n * time_delta[n_t])
            matrix[n_t, n_r] += \
                (matrix[n_t-1, n_r] + 0.5 * delta_n * measured_irf[n_t]) * e + \
                0.5 * delta_n * measured_irf[n_t]

    matrix /= matrix.max()
