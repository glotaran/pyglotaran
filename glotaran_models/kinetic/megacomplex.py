from glotaran_core.model.megacomplex import Megacomplex


class KineticMegacomplex(Megacomplex):
    """
    A KineticMegacomplex is a Megacomplex with one or more K-Matrix labels.
    """
    def __init__(self, label, k_matrices):
        self.k_matrices = k_matrices
        super(KineticMegacomplex, self).__init__(label)

    @property
    def k_matrices(self):
        return self._k_matrices

    @k_matrices.setter
    def k_matrices(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(m, str) for m in value):
            raise TypeError
        self._k_matrices = value

    def __str__(self):
        return "{}\nK-Matrices: {}".format(super(KineticMegacomplex,
                                           self).__str__(), self.k_matrices)
