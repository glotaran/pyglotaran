"""Glotaran DOAS Matrix"""

import numpy as np

from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix
from glotaran.models.spectral_temporal.irf import IrfGaussian

from doas_matrix_faddeva import calc_doas_matrix_faddeva


def calculate_doas_matrix(dataset_descriptor, dataset, index):

    axis = dataset.coords['time'].values

    oscillations = _collect_oscillations(dataset_descriptor)

    matrix = np.zeros((axis.size, 2 * len(oscillations)), dtype=np.float64)
    clp = []
    if not len(oscillations) == 0:
        for oscillation in oscillations:
            clp.append(f'{oscillation.label}_sin')
            clp.append(f'{oscillation.label}_cos')

        delta = np.abs(axis[1:] - axis[:-1])
        delta_min = delta[np.argmin(delta)]
        frequency_max = 1 / (2 * 0.03 * delta_min)

        if dataset_descriptor.irf is None:
            idx = 0
            for oscillation in oscillations:

                # convert from cm^-1 to ps^-1
                frequency = oscillation.frequency * 0.03 * 2 * np.pi
                if frequency >= frequency_max:
                    frequency = np.mod(frequency, frequency_max)

                osc = np.exp(-oscillation.rate * axis - 1j * frequency * axis)
                matrix[:, idx] = osc.real
                matrix[:, idx + 1] = osc.imag
                idx += 2
        elif isinstance(dataset_descriptor.irf, IrfGaussian):

            centers, widths, scales, backsweep, backsweep_period = \
                    dataset_descriptor.irf.parameter(index)

            rates = np.array([osc.rate for osc in oscillations])
            frequencies = np.array([osc.frequency * 0.03 * 2 * np.pi for osc in oscillations])

            over_max = frequencies >= frequency_max
            frequencies[over_max] = np.mod(frequencies[over_max], frequency_max)

            for i, _ in enumerate(centers):
                calc_doas_matrix_faddeva(matrix, frequencies, rates, axis,
                                         centers[i], widths[i], scales[i])
            matrix /= np.sum(scales)

    kinetic_clp, kinetic_matrix = calculate_kinetic_matrix(dataset_descriptor, dataset, index)
    if kinetic_matrix is not None:
        clp = clp + kinetic_clp
        matrix = np.concatenate((matrix, kinetic_matrix), axis=1)
    return (clp, matrix)


def _collect_oscillations(dataset):
    return [osc for cmplx in dataset.megacomplex for osc in cmplx.oscillation]
