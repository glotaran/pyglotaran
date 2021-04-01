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
    def get_megacomplex_k_matrices(self):
        scales = (
            [
                self.megacomplex_scale[i]
                for i, megacomplex in enumerate(self.megacomplex)
                if megacomplex.has_k_matrix()
            ]
            if self.megacomplex_scale is not None
            else None
        )

        matrices = [
            megacomplex.full_k_matrix()
            for megacomplex in self.megacomplex
            if megacomplex.has_k_matrix()
        ]

        return scales, matrices

    def has_k_matrix(self) -> bool:
        _, k_matrices = self.get_megacomplex_k_matrices()
        return len(k_matrices) != 0

    def compartments(self):
        compartments = []
        _, k_matrices = self.get_megacomplex_k_matrices()
        for k in k_matrices:
            compartments += k.involved_compartments()
        return list(set(compartments))
