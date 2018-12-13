"""Glotaran DOAS Matrix"""

import numpy as np
from scipy.special import erf
#
from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix
from glotaran.models.spectral_temporal.irf import IrfGaussian


def calculate_doas_matrix(dataset, index, axis):

    oscillations = _collect_oscillations(dataset)
    matrix = np.zeros((axis.size, len(2 * oscillations)), dtype=np.float64)
    clp = []
    frequencies = []
    rates = []
    scale = dataset.scale if dataset.scale is not None else 1.0
    idx = 0

    for osc in oscillations:
        clp.append(f'{osc.label}_sin')
        clp.append(f'{osc.label}_cos')

        # convert from cm^-1 to ps^-1
        frequency = osc.frequency * 0.03 * 2 * np.pi
        frequencies.append(osc.frequency * 0.03 * 2 * np.pi)
        rates.append(osc.rate)

        if dataset.irf is None:
            osc = scale * np.exp(-osc.rate * axis - 1j * frequency * axis)
        elif isinstance(dataset.irf, IrfGaussian):
            centers, width, irf_scale, backsweep, backsweep_period = \
                    dataset.irf.parameter(index)
            axis = axis - centers
            d = width * width
            k = (osc.rate + 1j * frequency)

            a = (-1 * axis + 0.5 * d * k) * k
            a = np.minimum(a, 709)
            a = np.exp(a)
            b = 1 + erf((axis - d * k) / (np.sqrt(2) * width))
            osc = a * b
        matrix[:, idx] = osc.real
        matrix[:, idx + 1] = osc.imag
        idx += 2

    kinetic_clp, kinetic_matrix = calculate_kinetic_matrix(dataset, index, axis)
    if kinetic_matrix is not None:
        clp = clp + kinetic_clp
        matrix = np.concatenate((matrix, kinetic_matrix), axis=1)
    return (clp, matrix)


def _collect_oscillations(dataset):
    return [osc for cmplx in dataset.megacomplex for osc in cmplx.oscillation]
