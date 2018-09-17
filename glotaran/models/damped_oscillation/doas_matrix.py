"""Glotaran DOAS Matrix"""

#  import numpy as np
#  from scipy.special import erf
#
#  from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix


def calculate_doas_matrix(dataset, all_compartments, index, axis):
    """ Calculates the matrix.

    Parameters
    ----------
    matrix : np.array
        The preallocated matrix.

    compartment_order : list(str)
        A list of compartment labels to map compartments to indices in the
        matrix.

    parameter : lmfit.Parameters
        A dictory of parameters.
    """
    #  matrix = calculate_kinetic_matrix(dataset, all_compartments, index, axis)
    #
    #  matrix = np.zeros(len(list(_collect_oscillations(dataset)), axis.size),
    #                    dtype=np.float64)
    #  for osc in _collect_oscillations(dataset):
    #      scale = dataset.scale
    #      osc = None
    #      if dataset.irf is None:
    #          osc = scale * np.exp(-osc.rate * axis - 1j * osc.frequency * axis)
    #      else:
    #          centers, width, irf_scale, backsweep, backsweep_period = \
    #                  self._calculate_irf_parameter(parameter)
    #          d = width * width
    #          k = (rate + 1j * freq)
    #
    #          a = np.exp((-1 * self.time + 0.5 * d * k) * k)
    #          b = 1 + erf((self.time - d * k) / (np.sqrt(2) * width))
    #          osc = a * b
    #      matrix[:, idx] = osc.real
    #      matrix[:, idx + 1] = osc.imag


def _collect_oscillations(dataset):
    for cmplx in dataset.megacomplex:
        for osc in cmplx.oscilattion:
            yield osc
