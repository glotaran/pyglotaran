"""Spectral Temporal Dataset Descriptor"""

from typing import Dict, List

from glotaran.model import DatasetDescriptor, model_item

from .spectral_constraints import SpectralConstraint

@model_item(attributes={
    'initial_concentration': {'type': str, 'default': None},
    'irf': {'type': str, 'default': None},
    'baseline': {'type': str, 'default': None},
    'shapes': {'type': Dict[str, str], 'target': (None, 'shape'), 'default': None},
    'spectral_constraints': {
        'type': List[SpectralConstraint],
        'default': None},
})
class SpectralTemporalDatasetDescriptor(DatasetDescriptor):
    """SpectralTemporalDatasetDescriptor is an implementation of
    model.DatasetDescriptor for spectral or temporal models.

    A SpectralTemporalDatasetDescriptor additionally contains an
    instrument response functions(IRF) and one or more spectral shapes.
    """

    def get_k_matrices(self):
        for cmplx in self.megacomplex:
            yield (cmplx.label, cmplx.get_k_matrix())
