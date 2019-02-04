"""Spectral Temporal Dataset Descriptor"""

from typing import Dict

from glotaran.model import DatasetDescriptor, model_item


@model_item(properties={
    'initial_concentration': {'type': str, 'default': None},
    'irf': {'type': str, 'default': None},
    'baseline': {'type': True, 'default': None},
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
            scale = cmplx.scale if cmplx.scale is not None else 1.0
            yield (scale, cmplx.full_k_matrix())
