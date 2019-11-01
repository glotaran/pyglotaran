"""Glotaran DOAS Matrix"""

import numba as nb
import numpy as np
from scipy.special import erf

from glotaran.builtin.models.kinetic_image.kinetic_image_matrix \
    import kinetic_image_matrix
from glotaran.builtin.models.kinetic_image.irf import IrfMultiGaussian


def calculate_doas_matrix(dataset_descriptor=None, axis=None, index=None, irf=None):

    oscillations = _collect_oscillations(dataset_descriptor)
    matrix = np.zeros((axis.size, len(2 * oscillations)), dtype=np.float64)
    labels = [o.label for o in oscillations]
    clp = [
        lbl for o in zip([f'{l}_cos' for l in labels], [f'{l}_sin' for l in labels]) for lbl in o
    ]

    delta = np.abs(axis[1:] - axis[:-1])
    delta_min = delta[np.argmin(delta)]
    frequency_max = 1 / (2 * 0.03 * delta_min)
    frequencies = np.asarray([o.frequency * 0.03 * 2 * np.pi for o in oscillations])
    frequencies[frequencies >= frequency_max] = \
        np.mod(frequencies[frequencies >= frequency_max], frequency_max)
    rates = np.asarray([o.rate for o in oscillations])

    if dataset_descriptor.irf is None:
        calculate_doas_matrix_no_irf(matrix, frequencies, rates, axis)
    elif isinstance(dataset_descriptor.irf, IrfMultiGaussian):
        centers, widths, scales, _, _ = dataset_descriptor.irf.parameter(index)
        calculate_doas_matrix_gaussian_irf(
            matrix, frequencies, rates, axis, centers, widths, scales)

    kinetic_clp, kinetic_matrix = kinetic_image_matrix(dataset_descriptor, axis, index, irf)
    if kinetic_matrix is not None:
        clp = clp + kinetic_clp
        matrix = np.concatenate((matrix, kinetic_matrix), axis=1)
    return (clp, matrix)


@nb.jit(nopython=True, parallel=True)
def calculate_doas_matrix_no_irf(matrix, frequencies, rates, axis):

    idx = 0
    for frequency, rate in zip(frequencies, rates):
        osc = np.exp(-rate * axis - 1j * frequency * axis)
        matrix[:, idx] = osc.real
        matrix[:, idx + 1] = osc.imag
        idx += 2


def calculate_doas_matrix_gaussian_irf(matrix, frequencies, rates, axis, centers, widths, scales):

    idx = 0
    for frequency, rate in zip(frequencies, rates):
        osc = np.zeros_like(axis, dtype=np.complex64)
        for i in range(len(centers)):
            shifted_axis = axis - centers[i]
            d = widths[i]**2
            k = (rate + 1j * frequency)

            a = (-1 * shifted_axis + 0.5 * d * k) * k
            a = np.minimum(a, 709)
            a = np.exp(a)
            b = 1 + erf((shifted_axis - d * k) / (np.sqrt(2) * widths[i]))
            if not np.all(np.isfinite(a)):
                raise Exception("Non numeric values in term 'a' for oscillation with frequency.")
            if not np.all(np.isfinite(b)):
                idx = np.where(np.logical_not(np.isfinite(b)))[0]
                raise Exception("Non numeric values in term 'b' for oscillation.")
            osc = a * b * scales[i]
        osc /= np.sum(scales)
        matrix[:, idx] = osc.real
        matrix[:, idx + 1] = osc.imag
        idx += 2


def _collect_oscillations(dataset):
    return [osc for cmplx in dataset.megacomplex for osc in cmplx.oscillation]
