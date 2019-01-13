""" Glotaran Kinetic Matrix """

import numpy as np

from kinetic_matrix_no_irf import calc_kinetic_matrix_no_irf
from kinetic_matrix_gaussian_irf import calc_kinetic_matrix_gaussian_irf
from .irf import IrfGaussian, IrfMeasured


def calculate_kinetic_matrix(dataset, index, axis):

    scale = dataset.scale if dataset.scale is not None else 1.0
    compartments = None
    matrix = None
    for _, k_matrix in dataset.get_k_matrices():

        if k_matrix is None:
            continue

        if dataset.initial_concentration is None:
            raise Exception(f'No initial concentration specified in dataset "{dataset.label}"')

        (this_compartments, this_matrix) = _calculate_for_k_matrix(
            dataset,
            index,
            axis,
            k_matrix,
            scale,
        )

        if matrix is None:
            compartments = this_compartments
            matrix = this_matrix
        else:
            new_compartments = \
                    compartments + [c for c in this_compartments if c not in compartments]
            new_matrix = np.zeros((axis.size, len(new_compartments)), dtype=np.float64)
            for i, comp in enumerate(new_compartments):
                if comp in compartments:
                    new_matrix[:, i] += matrix[:, compartments.index(comp)]
                if comp in this_compartments:
                    new_matrix[:, i] += this_matrix[:, this_compartments.index(comp)]
            compartments = new_compartments
            matrix = new_matrix

    if dataset.baseline is not None:
        baseline_compartment = f'{dataset.label}_baseline'
        baseline = np.zeros((axis.size, 1), dtype=np.float64)
        baseline.fill(dataset.baseline)
        if matrix is None:
            compartments = [baseline_compartment]
            matrix = baseline
        else:
            compartments.append(baseline_compartment)
            matrix = np.concatenate((matrix, baseline), axis=1)

    if isinstance(dataset.irf, IrfGaussian) and dataset.irf.coherent_artifact:
        irf_compartments, irf_matrix = dataset.irf.calculate_coherent_artifact(index, axis)
        if matrix is None:
            compartments = irf_compartments
            matrix = baseline
        else:
            compartments += irf_compartments
            matrix = np.concatenate((matrix, irf_matrix), axis=1)

    return (compartments, matrix)


def _calculate_for_k_matrix(dataset, index, axis, k_matrix, scale):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments

    # we might have more compartments in the model then in the k matrix
    compartments = [comp for comp in dataset.initial_concentration.compartments
                    if comp in k_matrix.involved_compartments()]

    # the rates are the eigenvalues of the k matrix
    rates = k_matrix.rates(compartments)

    # init the matrix
    size = (axis.size, rates.size)
    matrix = np.zeros(size, dtype=np.float64)

    # calculate the c_matrix
    if isinstance(dataset.irf, IrfGaussian):
        center, width, irf_scale, backsweep, backsweep_period = \
                dataset.irf.parameter(index)
        for i in range(len(center)):
            calc_kinetic_matrix_gaussian_irf(matrix,
                                             rates,
                                             axis,
                                             center[i],
                                             width[i],
                                             scale * irf_scale[i],
                                             backsweep,
                                             backsweep_period,
                                             )

    else:
        calc_kinetic_matrix_no_irf(matrix, rates, axis, scale)
        if isinstance(dataset.irf, IrfMeasured):
            irf = dataset.irf.irfdata
            if len(irf.shape) == 2:
                idx = (np.abs(dataset.data.get_axis("spectral") - index)).argmin()
                irf = irf[idx, :]
            for i in range(matrix.shape[1]):
                matrix[:, i] = np.convolve(matrix[:, i], irf, mode="same")

    # apply initial concentration vector
    matrix = np.matmul(
        matrix,
        k_matrix.a_matrix(dataset.initial_concentration),
    )

    # done
    return (compartments, matrix)
