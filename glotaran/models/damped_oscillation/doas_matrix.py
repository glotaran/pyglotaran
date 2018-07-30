"""Glotaran DOAS Matrix"""

from typing import List
import numpy as np
from scipy.special import erf

from glotaran.model import Model, ParameterGroup
from glotaran.models.spectral_temporal import KineticMatrix


class DOASMatrix(KineticMatrix):
    """Extension of glotaran.models.spectral_temporal.KineticMatrix for a DOAS model."""
    def __init__(self, x: float, dataset: str, model: Model):
        """

        Parameters
        ----------
        x : float
            Point on the estimated axis the matrix calculated for

        dataset : str
            Dataset label of the dataset the matrix is calculated for

        model : glotaran.Model
            The model the matrix is calculated for


        """
        super(DOASMatrix, self).__init__(x, dataset, model)
        self._oscillations = []
        self._collect_oscillations(model)

    @property
    def compartment_order(self):
        """Sets the compartment order to map compartment labels to indices in
        the matrix"""
        compartment_order = super(DOASMatrix, self).compartment_order
        for osc in self._oscillations:
            compartment_order.append(osc.sin_compartment)
            compartment_order.append(osc.cos_compartment)
        return compartment_order

    def _collect_oscillations(self, model):
        for cmplx in [model.megacomplexes[mc] for mc in self.dataset.megacomplexes]:
            for label in cmplx.oscillations:
                self._oscillations.append(model.oscillations[label])

    def calculate(self,
                  matrix: np.array,
                  compartment_order: List[str],
                  parameter: ParameterGroup):
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


        Returns
        ^
        -------

        """
        super(DOASMatrix, self).calculate(matrix, compartment_order, parameter)

        idx = []
        freq = []
        rate = []
        for osc in self._oscillations:
            idx = compartment_order.index(osc.sin_compartment)
            freq = parameter.get(osc.frequency).value
            rate = parameter.get(osc.rate).value
            scale = self._dataset_scaling(parameter)
            osc = None
            if self._irf is None:
                osc = scale * np.exp(-rate * self.time - 1j * freq * self.time)
            else:
                centers, width, irf_scale, backsweep, backsweep_period = \
                        self._calculate_irf_parameter(parameter)
                d = width * width
                k = (rate + 1j * freq)

                a = np.exp((-1 * self.time + 0.5 * d * k) * k)
                b = 1 + erf((self.time - d * k) / (np.sqrt(2) * width))
                osc = a * b
            matrix[:, idx] = osc.real
            matrix[:, idx + 1] = osc.imag
