"""Glotaran DOAS Model Spectral Matrix"""

import numpy as np

from glotaran.models.spectral_temporal.spectral_matrix import calculate_spectral_matrix

from .doas_matrix import _collect_oscillations


def calculate_doas_spectral_matrix(model, dataset, axis):

    oscillations = _collect_oscillations(dataset)
    all_oscillations = []
    for osc in oscillations:
        all_oscillations.append(osc)
        all_oscillations.append(osc)
    matrix = np.ones((len(all_oscillations), axis.size), dtype=np.float64)
    for i, osc in enumerate(all_oscillations):
        if osc.label not in dataset.shapes:
            raise Exception(f'No shape for oscillation "{osc.label}"')
        shapes = dataset.shapes[osc.label]
        if not isinstance(shapes, list):
            shapes = [shapes]
        for shape in shapes:
            matrix[i, :] *= shape.calculate(axis)

    spectral_matrix = calculate_spectral_matrix(dataset, axis)

    if spectral_matrix is not None:
        matrix = np.concatenate((matrix, spectral_matrix), axis=0)

    return matrix
