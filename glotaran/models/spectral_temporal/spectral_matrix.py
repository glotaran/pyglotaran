"""Glotaran Spectral Matrix"""

from typing import List
import numpy as np

from glotaran.fitmodel import Matrix
from glotaran.model import Model, ParameterGroup


class SpectralMatrix(Matrix):
    """Implementation of glotaran.fitmodel.Matrix for a spectral model."""
    def __init__(self, index: float, dataset: str, model: Model):
        """

        Parameters
        ----------
        index : float
            Point on the estimated axis the matrix calculated for

        dataset : str
            Dataset label of the dataset the matrix is calculated for

        model : glotaran.Model
            The model the matrix is calculated for


        """
        super(SpectralMatrix, self).__init__(index, dataset, model)

        self.shapes = {}
        self.collect_shapes()

    def collect_shapes(self):
        """Collects all shapes for the matrix from dataset."""
        for comp, shapes in self.dataset.shapes.items():
            self.shapes[comp] = [self.model.shapes[s] for s in shapes]

    @property
    def compartment_order(self):
        """A list with compartment labels. The index of label indicates the
        index of the compartment in the matrix.
        """
        cmplxs = [self.model.megacomplexes[c] for c in self.dataset.megacomplexes]
        kmats = [self.model.k_matrices[k] for cmplx in cmplxs
                 for k in cmplx.k_matrices]
        return list(set([c for kmat in kmats for c in kmat.compartment_map]))

    @property
    def shape(self):
        """The matrix dimensions as tuple(M, N)."""
        axis = self.dataset.dataset.spectral_axis
        return (axis.shape[0], len(self.compartment_order))

    def calculate(self,
                  matrix: np.ndarray,
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

        parameter : glotaran.model.ParameterGroup

        """
        # We need the spectral shapes and axis to perform the calculations
        axis = self.dataset.dataset.spectral_axis

        for (i, comp) in enumerate(compartment_order):
            if comp in self.shapes:
                for shape in self.shapes[comp]:
                    matrix[:, i] += shape.calculate(axis, parameter)
            else:
                # we use ones, so that if no shape is defined for the
                # compartment, the  amplitude is 1.0 by convention.
                matrix[:, i].fill(1.0)
