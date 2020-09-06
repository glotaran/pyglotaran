"""Glotaran Spectral Matrix"""

import numpy as np


def spectral_matrix(dataset, axis):
    """Calculates the matrix.

    Parameters
    ----------
    matrix : np.array
        The preallocated matrix.

    compartment_order : list(str)
        A list of compartment labels to map compartments to indices in the
        matrix.

    parameter : glotaran.model.ParameterGroup

    """
    if dataset.initial_concentration is None:
        return None, None
    shape_compartments = [s for s in dataset.shape]
    compartments = [
        c for c in dataset.initial_concentration.compartments if c in shape_compartments
    ]
    matrix = np.zeros((axis.size, len(compartments)))
    for i, comp in enumerate(compartments):
        shapes = dataset.shape[comp]
        if not isinstance(shapes, list):
            shapes = [shapes]
        for shape in shapes:
            matrix[:, i] += shape.calculate(axis)
    return compartments, matrix
