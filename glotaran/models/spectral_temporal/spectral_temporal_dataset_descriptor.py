"""Spectral Temporal Dataset Descriptor"""

from typing import Dict
from glotaran.model import DatasetDescriptor, model_item


@model_item(attributes={
    'initial_concentration': {'type': str, 'default': None},
    'irf': {'type': str, 'default': None},
    'baseline': {'type': str, 'default': None},
    'shapes': {'type': Dict[str, str], 'target': (None, 'shape'), 'default': None},
})
class SpectralTemporalDatasetDescriptor(DatasetDescriptor):
    """SpectralTemporalDatasetDescriptor is an implementation of
    model.DatasetDescriptor for spectral or temporal models.

    A SpectralTemporalDatasetDescriptor additionally contains an
    instrument response functions(IRF) and one or more spectral shapes.
    """

    def get_k_matrices(self):
        for cmplx in self.megacomplex:
            full_k_matrix = None
            for k_matrix in cmplx.k_matrix:
                if full_k_matrix is None:
                    full_k_matrix = k_matrix
                # If multiple k matrices are present, we combine them
                else:
                    full_k_matrix = full_k_matrix.combine(k_matrix)
            yield (cmplx.label, full_k_matrix)
