""" Glotaran Kinetic Megacomplex """

from typing import List, Union

#  from glotaran.model.decorators import glotaran_model_item
from glotaran.model.megacomplex import Megacomplex


#  @glotaran_model_item(attributes={'k_matrices': List[str]})
class KineticMegacomplex:
    """A Megacomplex with one or more K-Matrices."""

    def __str__(self):
        string = super(KineticMegacomplex, self).__str__()
        string += f"* _K-Matrices_: {self.k_matrices}\n"
        return string
