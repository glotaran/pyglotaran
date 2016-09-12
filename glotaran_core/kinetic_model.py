from .model import Model
from .megacomplex import Megacomplex
from scipy.sparse import spmatrix


class Irf(object):
    """
    Represents an abstract IRF.
    """
    def __init__(self, label):
        if not isinstance(label, str):
            raise TypeError
        self._label = label

    def label(self):
        return self._label

    def get_irf_function(self):
        raise NotImplementedError


class GaussianIrf(Irf):
    """
    Represents a gaussian IRF.

    One width and one center is a single gauss.

    One center and multiple widths is a multiple gaussian.

    Multiple center and multiple widths is Double-, Triple- , etc. Gaussian.

    Parameter
    ---------

    label: label of the irf
    center: one or more center of the irf as parameter indices
    width: one or more widths of the gaussian as parameter index
    center_dispersion: polynomial coefficients for the dispersion of the
        center as list of parameter indices. None for no dispersion.
    width_dispersion: polynomial coefficients for the dispersion of the
        width as parameter indices. None for no dispersion.

    """
    def __init__(self, label, center, width, center_dispersion=None,
                 width_dispersion=None):
        self.center = center
        self.center_dispersion = center_dispersion
        self.width = width
        self.width_dispersion = width_dispersion
        super(GaussianIrf).__init__(label)


class KineticMegacomplex(Megacomplex):
    """
    A KineticMegacomplex is a Megacomplex with one or more K-Matrix labels.
    """
    def __init__(self, label, k_matrices):
        if not isinstance(k_matrices, list):
            k_matrices = [k_matrices]
        if any(not isinstance(m, str) for m in k_matrices):
            raise TypeError
        self.k_matrices = k_matrices
        super(KineticMegacomplex, self).__init__(label)


class KMatrix(object):
    """
    A KMatrix has an label and a scipy.sparse matrix
    """
    def __init__(self, label, matrix):
        if not isinstance(label, str) or \
          not issubclass(type(matrix), spmatrix):
            raise TypeError
        self._label = label
        self._matrix = matrix

    def label(self):
        return self._label

    def matrix(self):
        return self._matrix

    def __str__(self):
        return "Label: {}\nMatrix:\n{}".format(self.label(),
                                               self.matrix().toarray())


class KineticModel(Model):
    """
    A kinetic model is a model with one ore more Irfs, K-Matrices and
    KineticMegacomplexes.
    """
    def __init__(self):
        self.k_matrices = {}
        self.irfs = {}
        super(KineticModel, self).__init__()

    def add_megakomplex(self, megacomplex):
        if not isinstance(megacomplex, KineticMegacomplex):
            raise TypeError
        super(KineticModel).add_megakomplex(megacomplex)

    def add_k_matrix(self, k_matrix):
        if not isinstance(k_matrix, KMatrix):
            raise TypeError
        if k_matrix.label() in self.k_matrices:
            raise Exception("K-Matrix labels must be unique.")
        self.k_matrices[k_matrix.label()] = k_matrix

    def add_irf(self, irf):
        if not issubclass(type(irf), Irf):
            raise TypeError
        if irf.label() in self.k_matrices:
            raise Exception("Irf labels must be unique.")
        self.k_matrices[irf.label()] = irf

    def __str__(self):
        s = "{}\n\nK-Matrices\n----------\n\n".format(super(KineticModel,
                                                            self).__str__())
        for k in self.k_matrices:
            s += "{}\n\n".format(self.k_matrices[k])
        return s
