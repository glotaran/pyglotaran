"""Spectral Temporal Dataset Descriptor"""

from typing import Dict

from glotaran.model import DatasetDescriptor, model_item


@model_item(properties={
    'initial_concentration': {'type': str, 'allow_none': True},
    'irf': {'type': str, 'allow_none': True},
    'baseline': {'type': bool, 'allow_none': True},
    'shape': {'type': Dict[str, str], 'allow_none': True},
})
class SpectralTemporalDatasetDescriptor(DatasetDescriptor):
    """SpectralTemporalDatasetDescriptor is an implementation of
    model.DatasetDescriptor for spectral or temporal models.

    A SpectralTemporalDatasetDescriptor additionally contains an
    instrument response functions(IRF) and one or more spectral shapes.
    """

    def get_k_matrices(self):
        return [mat for mat in [cmplx.full_k_matrix() for cmplx in self.megacomplex] if mat]
