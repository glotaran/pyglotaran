"""Glotaran DOAS Matrix"""

import numpy as np
#  from scipy.special import erf
#
from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix


def calculate_doas_matrix(dataset, index, axis):

    oscillations = _collect_oscillations(dataset)
    matrix = np.zeros((len(2 * oscillations), axis.size), dtype=np.float64)
    clp = []
    scale = dataset.scale if dataset.scale is not None else 1.0

    idx = 0

    for osc in oscillations:
        clp.append(f'{osc.label}_sin')
        clp.append(f'{osc.label}_cos')
        if dataset.irf is None:
            osc = scale * np.exp(-osc.rate * axis - 1j * osc.frequency * axis)
    #      else:
    #          centers, width, irf_scale, backsweep, backsweep_period = \
    #                  self._calculate_irf_parameter(parameter)
    #          d = width * width
    #          k = (rate + 1j * freq)
    #
    #          a = np.exp((-1 * self.time + 0.5 * d * k) * k)
    #          b = 1 + erf((self.time - d * k) / (np.sqrt(2) * width))
    #          osc = a * b
        matrix[idx, :] = osc.real
        matrix[idx + 1, :] = osc.imag
        idx += 2

    kinetic_clp, kinetic_matrix = calculate_kinetic_matrix(dataset, index, axis)
    if kinetic_matrix is not None:
        clp = kinetic_clp + clp
        matrix = np.concatenate((matrix, kinetic_matrix))
    return (clp, matrix)


def _collect_oscillations(dataset):
    return [osc for cmplx in dataset.megacomplex for osc in cmplx.oscillation]
