"""This package contains the kinetic megacomplex item."""
from __future__ import annotations

from typing import List

import numba as nb
import numpy as np

from glotaran.builtin.models.kinetic_image.irf import IrfMultiGaussian
from glotaran.model import DatasetDescriptor
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import model_attribute


@model_attribute(
    properties={
        "k_matrix": List[str],
    }
)
class KineticDecayMegacomplex(Megacomplex):
    """A Megacomplex with one or more K-Matrices."""

    def has_k_matrix(self) -> bool:
        return len(self.k_matrix) != 0

    def full_k_matrix(self, model=None):
        full_k_matrix = None
        for k_matrix in self.k_matrix:
            if model:
                k_matrix = model.k_matrix[k_matrix]
            if full_k_matrix is None:
                full_k_matrix = k_matrix
            # If multiple k matrices are present, we combine them
            else:
                full_k_matrix = full_k_matrix.combine(k_matrix)
        return full_k_matrix

    @property
    def involved_compartments(self):
        return self.full_k_matrix().involved_compartments() if self.full_k_matrix() else []

    def calculate_matrix(
        self,
        model,
        dataset_descriptor: DatasetDescriptor,
        indices: dict[str, int],
        axis: dict[str, np.ndarray],
        **kwargs,
    ):
        if dataset_descriptor.initial_concentration is None:
            raise ModelError(
                f'No initial concentration specified in dataset "{dataset_descriptor.label}"'
            )
        initial_concentration = dataset_descriptor.initial_concentration.normalized()

        k_matrix = self.full_k_matrix()

        # we might have more compartments in the model then in the k matrix
        compartments = [
            comp
            for comp in initial_concentration.compartments
            if comp in k_matrix.involved_compartments()
        ]

        # the rates are the eigenvalues of the k matrix
        rates = k_matrix.rates(initial_concentration)

        global_index = indices.get(model.global_dimension, None)
        global_axis = axis.get(model.global_dimension, None)
        model_axis = axis[model.model_dimension]

        # init the matrix
        size = (model_axis.size, rates.size)
        matrix = np.zeros(size, dtype=np.float64)

        kinetic_image_matrix_implementation(
            matrix, rates, global_index, global_axis, model_axis, dataset_descriptor
        )

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
    matrix: np.ndarray,
    rates: np.ndarray,
    global_index: int,
    global_axis: np.ndarray,
    model_axis: np.ndarray,
    dataset_descriptor: DatasetDescriptor,
):
    if isinstance(dataset_descriptor.irf, IrfMultiGaussian):

        (
            centers,
            widths,
            irf_scales,
            shift,
            backsweep,
            backsweep_period,
        ) = dataset_descriptor.irf.parameter(global_index, global_axis)

        for center, width, irf_scale in zip(centers, widths, irf_scales):
            calculate_kinetic_matrix_gaussian_irf(
                matrix,
                rates,
                model_axis,
                center - shift,
                width,
                irf_scale,
                backsweep,
                backsweep_period,
            )
        if dataset_descriptor.irf.normalize:
            matrix /= np.sum(irf_scale)

    else:
        calculate_kinetic_matrix_no_irf(matrix, rates, model_axis)


@nb.jit(nopython=True, parallel=True)
def calculate_kinetic_matrix_no_irf(matrix, rates, times):
    for n_r in nb.prange(rates.size):
        r_n = rates[n_r]
        for n_t in range(times.size):
            t_n = times[n_t]
            matrix[n_t, n_r] += np.exp(r_n * t_n)


sqrt2 = np.sqrt(2)


@nb.jit(nopython=True, parallel=True)
def calculate_kinetic_matrix_gaussian_irf(
    matrix, rates, times, center, width, scale, backsweep, backsweep_period
):
    """Calculates a kinetic matrix with a gaussian irf."""
    for n_r in nb.prange(rates.size):
        r_n = -rates[n_r]
        backsweep_valid = abs(r_n) * backsweep_period > 0.001
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
            if backsweep and backsweep_valid:
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
