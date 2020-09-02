"""Glotaran kinetic spectrum Matrix"""

import numpy as np

from glotaran.builtin.models.kinetic_image.kinetic_image_matrix import kinetic_image_matrix

from .spectral_irf import IrfGaussianCoherentArtifact


def kinetic_spectrum_matrix(dataset_descriptor=None, axis=None, index=None, irf=None):

    clp_label, matrix = kinetic_image_matrix(
        dataset_descriptor, axis, index, dataset_descriptor.irf
    )

    if isinstance(dataset_descriptor.irf, IrfGaussianCoherentArtifact):
        irf_clp_label, irf_matrix = dataset_descriptor.irf.calculate_coherent_artifact(axis)
        if matrix is None:
            clp_label = irf_clp_label
            matrix = irf_matrix
        else:
            clp_label += irf_clp_label
            matrix = np.concatenate((matrix, irf_matrix), axis=1)

    return (clp_label, matrix)
