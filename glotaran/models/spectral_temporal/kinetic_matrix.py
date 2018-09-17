""" Glotaran Kinetic Matrix """

import numpy as np

from kinetic_matrix_no_irf import calc_kinetic_matrix_no_irf
from kinetic_matrix_gaussian_irf import calc_kinetic_matrix_gaussian_irf
from .irf import IrfGaussian, IrfMeasured


def calculate_kinetic_matrix(dataset, all_compartments, index, axis):
    """ Calculates the matrix.

    Parameters
    ----------
    matrix : np.array
        The preallocated matrix.

    compartment_order : list(str)
        A list of compartment labels to map compartments to indices in the
        matrix.

    parameter : glotaran.model.ParameterGroup

    """

    scale = dataset.scale if dataset.scale is not None else 1.0
    compartments = None
    matrix = None
    for k_matrix in _collect_k_matrices(dataset):
        (this_compartments, this_matrix) = _calculate_for_k_matrix(
            dataset,
            all_compartments,
            index,
            axis,
            k_matrix,
            scale,
        )

        if matrix is None:
            compartments = this_compartments
            matrix = this_matrix
        else:
            for comp in this_compartments:
                if comp in compartments:
                    matrix[compartments.index(comp), :] += \
                        this_matrix[this_compartments.index(comp), :]
                else:
                    matrix = np.concatenate((matrix,
                                             this_matrix[this_compartments.index(comp), :]))
    return (compartments, matrix)


def _collect_k_matrices(dataset):
    for cmplx in dataset.megacomplex:
        full_k_matrix = None
        for k_matrix in cmplx.k_matrix:
            if full_k_matrix is None:
                full_k_matrix = k_matrix
            # If multiple k matrices are present, we combine them
            else:
                full_k_matrix = full_k_matrix.combine(k_matrix)
        yield full_k_matrix


def _calculate_for_k_matrix(dataset, compartments, index, axis, k_matrix, scale):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments

    # we might have more compartments in the model then in the k matrix
    compartments = [comp for comp in compartments
                    if comp in k_matrix.involved_compartments()]

    # the rates are the eigenvalues of the k matrix
    rates, _ = k_matrix.eigen(compartments)

    # init the matrix
    size = (len(rates), axis.shape[0])
    matrix = np.zeros(size)

    # calculate the c_matrix
    if isinstance(dataset.irf, IrfGaussian):
        centers, widths, irf_scale, backsweep, backsweep_period = \
                dataset.irf.parameter(index)
        calc_kinetic_matrix_gaussian_irf(matrix,
                                         rates,
                                         axis,
                                         centers,
                                         widths,
                                         scale * irf_scale,
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
            for i in range(matrix.shape[0]):
                matrix[i, :] = np.convolve(matrix[i, :], irf, mode="same")

    # apply initial concentration vector
    matrix = np.dot(k_matrix.a_matrix(compartments,
                                      dataset.initial_concentration), matrix)

    # done
    return (compartments, matrix)
