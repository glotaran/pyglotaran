"""Glotaran DOAS Model Spectral Matrix"""

import numpy as np

from glotaran.builtin.models.kinetic_spectrum.spectral_matrix import spectral_matrix

from .doas_matrix import _collect_oscillations


def calculate_doas_spectral_matrix(dataset, axis):

    oscillations = _collect_oscillations(dataset)
    all_oscillations = []
    clp = []
    for osc in oscillations:
        all_oscillations.append(osc)
        all_oscillations.append(osc)
        clp.append(f'{osc.label}_sin')
        clp.append(f'{osc.label}_cos')
    matrix = np.ones((axis.size, len(all_oscillations)), dtype=np.float64)
    for i, osc in enumerate(all_oscillations):
        if osc.label not in dataset.shape:
            raise Exception(f'No shape for oscillation "{osc.label}"')
        shapes = dataset.shape[osc.label]
        if not isinstance(shapes, list):
            shapes = [shapes]
        for shape in shapes:
            matrix[:, i] *= shape.calculate(axis)

    kinetic_spectral_clp, kinetic_spectral_matrix = spectral_matrix(dataset, axis)

    if kinetic_spectral_matrix is not None:
        matrix = np.concatenate((matrix, kinetic_spectral_matrix), axis=1)
        clp += kinetic_spectral_clp

    return clp, matrix
