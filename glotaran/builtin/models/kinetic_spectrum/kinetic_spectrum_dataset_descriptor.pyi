from __future__ import annotations

from glotaran.builtin.models.kinetic_image.kinetic_image_dataset_descriptor import (
    KineticImageDatasetDescriptor,
)
from glotaran.model import model_attribute  # noqa: F401

class KineticSpectrumDatasetDescriptor(KineticImageDatasetDescriptor):
    @property
    def shape(self) -> dict[str, str]: ...
