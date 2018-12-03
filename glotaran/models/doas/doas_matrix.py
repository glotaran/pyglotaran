"""Glotaran DOAS Matrix"""

import numpy as np
from scipy.special import erf
#
from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix
from glotaran.models.spectral_temporal.irf import IrfGaussian


def calculate_doas_matrix(dataset, index, axis):

    oscillations = _collect_oscillations(dataset)
    matrix = np.zeros((len(2 * oscillations), axis.size), dtype=np.float64)
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

            a = np.exp((-1 * axis + 0.5 * d * k) * k)
            b = 1 + erf((axis - d * k) / (np.sqrt(2) * width))
            osc = a * b
        matrix[idx, :] = osc.real
        matrix[idx + 1, :] = osc.imag
        idx += 2

    #  if dataset.irf is None:
    #      raise Exception
    #  elif isinstance(dataset.irf, IrfGaussian):
    #      center, width, irf_scale, _, _ = dataset.irf.parameter(index)
    #      calc_doas_matrix_gaussian_irf(matrix,
    #                                    np.asarray(rates, dtype=np.float64),
    #                                    np.asarray(frequencies, dtype=np.float64),
    #                                    axis,
    #                                    center,
    #                                    width,
    #                                    scale)
    #  else:
    #      raise Exception(f'Unknown irf type "{type(dataset.irf)}"')

    kinetic_clp, kinetic_matrix = calculate_kinetic_matrix(dataset, index, axis)
    if kinetic_matrix is not None:
        clp = clp + kinetic_clp
        matrix = np.concatenate((matrix, kinetic_matrix))
    return (clp, matrix)


def _collect_oscillations(dataset):
    return [osc for cmplx in dataset.megacomplex for osc in cmplx.oscillation]
