from typing import Dict

from glotaran.builtin.models.kinetic_image.kinetic_image_dataset_descriptor import (
    KineticImageDatasetDescriptor,
)
from glotaran.model import model_attribute

class KineticSpectrumDatasetDescriptor(KineticImageDatasetDescriptor):
    @property
    def shape(self) -> Dict[str, str]:
        ...
