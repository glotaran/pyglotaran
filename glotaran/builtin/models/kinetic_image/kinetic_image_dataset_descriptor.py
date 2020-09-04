""" Kinetic Image Dataset Descriptor"""

from glotaran.model import DatasetDescriptor
from glotaran.model import model_attribute


@model_attribute(
    properties={
        "initial_concentration": {"type": str, "allow_none": True},
        "irf": {"type": str, "allow_none": True},
        "baseline": {"type": bool, "allow_none": True},
    }
)
class KineticImageDatasetDescriptor(DatasetDescriptor):
    def get_k_matrices(self):
        return [mat for mat in [cmplx.full_k_matrix() for cmplx in self.megacomplex] if mat]

    def compartments(self):
        compartments = []
        for k in self.get_k_matrices():
            compartments += k.involved_compartments()
        return list(set(compartments))
