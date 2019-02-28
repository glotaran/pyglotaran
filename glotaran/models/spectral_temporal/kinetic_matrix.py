""" Glotaran Kinetic Matrix """

import numpy as np

from kinetic_matrix_no_irf import calc_kinetic_matrix_no_irf
from kinetic_matrix_gaussian_irf import calc_kinetic_matrix_gaussian_irf
from .irf import IrfGaussian, IrfMeasured


def calculate_kinetic_matrix(dataset_descriptor, dataset, index):

    compartments = None
    matrix = None
    k_matrices = dataset_descriptor.get_k_matrices()

    if len(k_matrices) == 0:
        return (None, None)

    if dataset_descriptor.initial_concentration is None:
        raise Exception(
            f'No initial concentration specified in dataset "{dataset_descriptor.label}"')
    initial_concentration = dataset_descriptor.initial_concentration.normalized(dataset_descriptor)

    for k_matrix in k_matrices:

        if k_matrix is None:
            continue

        (this_compartments, this_matrix) = _calculate_for_k_matrix(
            dataset_descriptor,
            dataset,
            index,
            k_matrix,
            initial_concentration
        )

        if matrix is None:
            compartments = this_compartments
            matrix = this_matrix
        else:
            new_compartments = \
                    compartments + [c for c in this_compartments if c not in compartments]
            new_matrix = np.zeros((matrix.shape[0], len(new_compartments)), dtype=np.float64)
            for i, comp in enumerate(new_compartments):
                if comp in compartments:
                    new_matrix[:, i] += matrix[:, compartments.index(comp)]
                if comp in this_compartments:
                    new_matrix[:, i] += this_matrix[:, this_compartments.index(comp)]
            compartments = new_compartments
            matrix = new_matrix

    if dataset_descriptor.baseline:
        baseline_compartment = f'{dataset_descriptor.label}_baseline'
        baseline = np.ones((dataset.coords['time'].size, 1), dtype=np.float64)
        if matrix is None:
            compartments = [baseline_compartment]
            matrix = baseline
        else:
            compartments.append(baseline_compartment)
            matrix = np.concatenate((matrix, baseline), axis=1)

    if isinstance(dataset_descriptor.irf, IrfGaussian) \
       and dataset_descriptor.irf.coherent_artifact:
        irf_compartments, irf_matrix = \
                dataset_descriptor.irf.calculate_coherent_artifact(index, dataset.coords['time'])
        if matrix is None:
            compartments = irf_compartments
            matrix = baseline
        else:
            compartments += irf_compartments
            matrix = np.concatenate((matrix, irf_matrix), axis=1)

    return (compartments, matrix)


def _calculate_for_k_matrix(dataset_descriptor, dataset, index, k_matrix, initial_concentration):

    # we might have more compartments in the model then in the k matrix
    compartments = [comp for comp in initial_concentration.compartments
                    if comp in k_matrix.involved_compartments()]

    # the rates are the eigenvalues of the k matrix
    rates = k_matrix.rates(initial_concentration)

    # init the matrix
    axis = dataset.coords['time'].values
    size = (axis.size, rates.size)
    matrix = np.zeros(size, dtype=np.float64)

    # calculate the c_matrix
    if isinstance(dataset_descriptor.irf, IrfGaussian):

        center, width, irf_scale, backsweep, backsweep_period = \
                dataset_descriptor.irf.parameter(index)

        for i in range(len(center)):
            calc_kinetic_matrix_gaussian_irf(matrix,
                                             rates,
                                             axis,
                                             center[i],
                                             width[i],
                                             irf_scale[i],
                                             backsweep,
                                             backsweep_period,
                                             )
        matrix /= np.sum(irf_scale)

    else:
        calc_kinetic_matrix_no_irf(matrix, rates, axis)
        if isinstance(dataset_descriptor.irf, IrfMeasured):
            irf = dataset.irf
            if len(irf.shape) == 2:
                irf = irf.sel({irf.dims[1]: index}, method='nearest')
            for i in range(matrix.shape[1]):
                matrix[:, i] = np.convolve(matrix[:, i], irf, mode="same")

    # apply A matrix
    matrix = matrix @ k_matrix.a_matrix(initial_concentration)

    # done
    return (compartments, matrix)
