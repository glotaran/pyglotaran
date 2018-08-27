""" Glotaran Kinetic Megacomplex """

from typing import List, Union

from glotaran.model.megacomplex import Megacomplex


class KineticMegacomplex(Megacomplex):
    """A Megacomplex with one or more K-Matrices."""
    def __init__(self, label: str, k_matrices: Union[str, List[str]]):
        """

        Parameters
        ----------
        label: str
            The label of the megacomplex.

        k_matrices: list(str) :
            A list of K-Matrix labels.

        Returns
        -------

        """
        if not isinstance(k_matrices, list):
            k_matrices = [k_matrices]
        self._k_matrices = k_matrices
        super(KineticMegacomplex, self).__init__(label)

    @property
    def k_matrices(self):
        """ K-Matrices associated with the megacomplex."""
        return self._k_matrices

    @k_matrices.setter
    def k_matrices(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(m, str) for m in value):
            raise TypeError
        self._k_matrices = value

    def __str__(self):
        string = super(KineticMegacomplex, self).__str__()
        string += f"* _K-Matrices_: {self.k_matrices}\n"
        return string
