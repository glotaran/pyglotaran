import typing

from glotaran.model import model_attribute
from glotaran.builtin.models.kinetic_image.kinetic_image_dataset_descriptor \
    import KineticImageDatasetDescriptor


@model_attribute(properties={
    'shape': {'type': typing.Dict[str, str], 'allow_none': True},
})
class KineticSpectrumDatasetDescriptor(KineticImageDatasetDescriptor):
    pass
