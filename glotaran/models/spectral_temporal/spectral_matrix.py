"""Glotaran Spectral Matrix"""

import numpy as np


def calculate_spectral_matrix(dataset, compartments, axis):
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

    shape_compartments = [s for s in dataset.shapes]
    compartments = [c for c in compartments if c in shape_compartments]
    matrix = np.zeros((axis.size, len(compartments)))
    print('eshape', matrix.shape)
    for i, comp in enumerate(compartments):
        shapes = dataset.shapes[comp]
        if not isinstance(shapes, list):
            shapes = [shapes]
        for shape in shapes:
            matrix[:, i] += shape.calculate(axis)
    return matrix
