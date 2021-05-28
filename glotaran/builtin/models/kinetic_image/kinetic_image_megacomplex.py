"""This package contains the kinetic megacomplex item."""

from typing import List

from glotaran.model import Megacomplex
from glotaran.model import model_attribute


@model_attribute(
    properties={
        "k_matrix": {"type": List[str], "default": []},
    }
)
class KineticImageMegacomplex(Megacomplex):
    """A Megacomplex with one or more K-Matrices."""

    def has_k_matrix(self) -> bool:
        return len(self.k_matrix) != 0

    def full_k_matrix(self, model=None):
        full_k_matrix = None
        for k_matrix in self.k_matrix:
            if model:
                k_matrix = model.k_matrix[k_matrix]
            if full_k_matrix is None:
                full_k_matrix = k_matrix
            # If multiple k matrices are present, we combine them
            else:
                full_k_matrix = full_k_matrix.combine(k_matrix)
        return full_k_matrix

    @property
    def involved_compartments(self):
        return self.full_k_matrix().involved_compartments() if self.full_k_matrix() else []
