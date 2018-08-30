"""Spectral Temporal Dataset Descriptor"""

from typing import Dict, List
from glotaran.model import DatasetDescriptor, glotaran_model_item


@glotaran_model_item(attributes={
    'irf': {'type': str, 'default': None},
    'shape': {'type': Dict[str, List[str]], 'default': None}},
)
class SpectralTemporalDatasetDescriptor(DatasetDescriptor):
    """SpectralTemporalDatasetDescriptor is an implementation of
    model.DatasetDescriptor for spectral or temporal models.

    A SpectralTemporalDatasetDescriptor additionally contains an
    instrument response functions(IRF) and one or more spectral shapes.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=attribute-defined-outside-init
    # Datasets are complex.
