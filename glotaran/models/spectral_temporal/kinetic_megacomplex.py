"""This package contains the kinetic megacomplex item."""

from typing import List

from glotaran.model import model_item


@model_item(attributes={'k_matrix': {'type': List[str], 'default': []}})
class KineticMegacomplex:
    """A Megacomplex with one or more K-Matrices."""

    def get_k_matrix(self):
        full_k_matrix = None
        for k_matrix in self.k_matrix:
            if full_k_matrix is None:
                full_k_matrix = k_matrix
            # If multiple k matrices are present, we combine them
            else:
                full_k_matrix = full_k_matrix.combine(k_matrix)
        return full_k_matrix
