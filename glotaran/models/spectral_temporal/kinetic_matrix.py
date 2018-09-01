""" Glotaran Kinetic Matrix """

from typing import List, Tuple
import numpy as np

from glotaran.fitmodel import Matrix
from glotaran.model import Model, ParameterGroup

from kinetic_matrix_no_irf import calc_kinetic_matrix_no_irf
from .irf import IrfGaussian, IrfMeasured
from .k_matrix import KMatrix


def calculate_kinetic_matrix(dataset, compartments, index, axis):
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
    for k_matrix in _collect_k_matrices(dataset):
        self._calculate_for_k_matrix(matrix, compartment_order, k_matrix,
                                     parameter, scale)

def _collect_k_matrices(dataset):
    for cmplx in dataset.megacomplex:
        full_k_matrix = None
        for k_matrix in cmplx.k_matrix:
            if model_k_matrix is None:
                model_k_matrix = k_matrix
            # If multiple k matrices are present, we combine them
            else:
                model_k_matrix = model_k_matrix.combine(k_matrix)
        yield full_k_matrix


def _calculate_for_k_matrix(dataset, compartments, k_matrix, index, axis):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments

    # we might have more compartments in the model then in the k matrix
    compartments = [comp for comp in compartments
                    if comp in k_matrix.involved_compartments]

    # the rates are the eigenvalues of the k matrix
    rates, _ = k_matrix.eigen(compartments)

    # init the matrix
    size = (len(rates), axis.shape[0])
    matrix = np.zeros(shape)

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
            irf = dataset.irf.data
            if len(irf.shape) == 2:
                idx = (np.abs(self.dataset.data.spectral_axis - self.index)).argmin()
                irf = irf[idx, :]
            for i in range(matrix.shape[1]):
                matrix[:, i] = np.convolve(matrix[:, i], irf, mode="same")

#      if self._initial_concentration is not None:
#          self._apply_initial_concentration_vector(matrix,
#                                                   k_matrix,
#                                                   parameter,
#                                                   compartment_order)
#
#  def _apply_initial_concentration_vector(self, c_matrix, k_matrix,
#                                          parameter, compartment_order):
#      mask = [c in self.compartment_order for c in compartment_order]
#
#      temp = np.dot(np.copy(c_matrix[:, mask]),
#                    k_matrix.a_matrix(self._initial_concentration, parameter))
#
#      for i, comp in enumerate(self.compartment_order):
#          c_matrix[:, compartment_order.index(comp)] = temp[:, i]
