""" Glotaran Kinetic Megacomplex """

from typing import List, Union

from glotaran.model import glotaran_model_item


@glotaran_model_item(attributes={'k_matrix': List[str]})
class KineticMegacomplex:
    """A Megacomplex with one or more K-Matrices."""
    pass
