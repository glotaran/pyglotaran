""" Glotaran Kinetic Matrix """

import numba as nb
import numpy as np

from .irf import IrfMultiGaussian

sqrt2 = np.sqrt(2)


def kinetic_image_matrix(dataset_descriptor=None, axis=None, index=None, irf=None):
    return kinetic_matrix(
        dataset_descriptor, axis, index, irf, kinetic_image_matrix_implementation
    )


def kinetic_matrix(
    dataset_descriptor=None, axis=None, index=None, irf=None, matrix_implementation=None
):

    compartments = None
    matrix = None
    k_matrices = dataset_descriptor.get_k_matrices()

    if len(k_matrices) == 0:
        return (None, None)

    if dataset_descriptor.initial_concentration is None:
        raise Exception(
            f'No initial concentration specified in dataset "{dataset_descriptor.label}"'
        )
    initial_concentration = dataset_descriptor.initial_concentration.normalized(dataset_descriptor)

    for k_matrix in k_matrices:

        if k_matrix is None:
            continue

        (this_compartments, this_matrix) = _calculate_for_k_matrix(
            dataset_descriptor,
            axis,
            index,
            k_matrix,
            initial_concentration,
            irf,
            matrix_implementation,
        )

        if matrix is None:
            compartments = this_compartments
            matrix = this_matrix
        else:
            new_compartments = compartments + [
                c for c in this_compartments if c not in compartments
            ]
            new_matrix = np.zeros((matrix.shape[0], len(new_compartments)), dtype=np.float64)
            for i, comp in enumerate(new_compartments):
                if comp in compartments:
                    new_matrix[:, i] += matrix[:, compartments.index(comp)]
                if comp in this_compartments:
                    new_matrix[:, i] += this_matrix[:, this_compartments.index(comp)]
            compartments = new_compartments
            matrix = new_matrix

    if dataset_descriptor.baseline:
        baseline_compartment = f"{dataset_descriptor.label}_baseline"
        baseline = np.ones((axis.size, 1), dtype=np.float64)
        if matrix is None:
            compartments = [baseline_compartment]
            matrix = baseline
        else:
            compartments.append(baseline_compartment)
            matrix = np.concatenate((matrix, baseline), axis=1)

    return (compartments, matrix)


def _calculate_for_k_matrix(
    dataset_descriptor, axis, index, k_matrix, initial_concentration, irf, matrix_implementation
):

    # we might have more compartments in the model then in the k matrix
    compartments = [
        comp
        for comp in initial_concentration.compartments
        if comp in k_matrix.involved_compartments()
    ]

    # the rates are the eigenvalues of the k matrix
    rates = k_matrix.rates(initial_concentration)

    # init the matrix
    size = (axis.size, rates.size)
    matrix = np.zeros(size, dtype=np.float64)

    matrix_implementation(matrix, rates, axis, index, dataset_descriptor, irf)

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"Non-finite concentrations for K-Matrix '{k_matrix.label}':\n"
            f"{k_matrix.matrix_as_markdown(fill_parameters=True)}"
        )

    # apply A matrix
    matrix = matrix @ k_matrix.a_matrix(initial_concentration)

    # done
    return (compartments, matrix)


def kinetic_image_matrix_implementation(
    matrix, rates, axis, index, dataset_descriptor, measured_irf
):

    if isinstance(dataset_descriptor.irf, IrfMultiGaussian):

        center, width, irf_scale, backsweep, backsweep_period = dataset_descriptor.irf.parameter(
            index
        )

        for i in range(len(center)):
            calculate_kinetic_matrix_gaussian_irf(
                matrix,
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
        calculate_kinetic_matrix_no_irf(matrix, rates, axis)


@nb.jit(nopython=True, parallel=True)
def calculate_kinetic_matrix_no_irf(matrix, rates, times):
    for n_r in nb.prange(rates.size):
        r_n = rates[n_r]
        for n_t in range(times.size):
            t_n = times[n_t]
            matrix[n_t, n_r] += np.exp(r_n * t_n)


@nb.jit(nopython=True, parallel=True)
def calculate_kinetic_matrix_gaussian_irf(
    matrix, rates, times, center, width, scale, backsweep, backsweep_period
):
    """Calculates a kinetic matrix with a gaussian irf."""
    for n_r in nb.prange(rates.size):
        r_n = -rates[n_r]
        alpha = (r_n * width) / sqrt2
        for n_t in nb.prange(times.size):
            t_n = times[n_t]
            beta = (t_n - center) / (width * sqrt2)
            thresh = beta - alpha
            if thresh < -1:
                matrix[n_t, n_r] += scale * 0.5 * erfcx(-thresh) * np.exp(-beta * beta)
            else:
                matrix[n_t, n_r] += (
                    scale * 0.5 * (1 + erf(thresh)) * np.exp(alpha * (alpha - 2 * beta))
                )
            if backsweep:
                x1 = np.exp(-r_n * (t_n - center + backsweep_period))
                x2 = np.exp(-r_n * ((backsweep_period / 2) - (t_n - center)))
                x3 = np.exp(-r_n * backsweep_period)
                matrix[n_t, n_r] += scale * (x1 + x2) / (1 - x3)


import ctypes  # noqa: E402

# This is a work around to use scipy.special function with numba
from numba.extending import get_cython_function_address  # noqa: E402

_dble = ctypes.c_double

functype = ctypes.CFUNCTYPE(_dble, _dble)

erf_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1erf")
erfcx_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1erfcx")

erf = functype(erf_addr)
erfcx = functype(erfcx_addr)
