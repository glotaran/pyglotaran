"""Glotaran Spectral Matrix"""

from typing import List
import numpy as np

from glotaran.fitmodel import Matrix
from glotaran.model import Model, ParameterGroup


def calculate_spectral_matrix(self,
                              dataset,
                              index,
                              axis):
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

    compartments = []
    matrix = np.zeros((len(dataset.shapes), axis.shape[0]))
    for i, shape in enumerate(dataset.shapes):
        compartments.append(shape.compartment)
        matrix[:, i] += shape.calculate(axis)
