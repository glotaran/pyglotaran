"""This package contains the kinetic megacomplex item."""

from typing import List

from glotaran.model import model_item


@model_item(attributes={'k_matrix': List[str]})
class KineticMegacomplex:
    """A Megacomplex with one or more K-Matrices."""
