""" Kinetic Image Dataset Descriptor"""

from glotaran.model import DatasetDescriptor
from glotaran.model import model_attribute


@model_attribute(
    properties={
        "initial_concentration": {"type": str, "allow_none": True},
        "irf": {"type": str, "allow_none": True},
    }
)
class KineticImageDatasetDescriptor(DatasetDescriptor):
    pass
