from .model import Model
from .megacomplex import Megacomplex
from .dataset import Dataset
from scipy.sparse import dok_matrix
import numpy as np


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

    def type_string(self):
        raise NotImplementedError

    def __str__(self):
        return "Label: {} Type: {}".format(self.label(), self.type_string())


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
    def __init__(self, label, center, width, center_dispersion=[],
                 width_dispersion=[]):
        if not isinstance(center, list):
            center = [center]
        if not isinstance(width, list):
            width = [width]
        if not isinstance(center_dispersion, list):
            center_dispersion = [center_dispersion]
        if not isinstance(width_dispersion, list):
            width_dispersion = [width_dispersion]
        if any(not isinstance(c, int) for c in center) or\
           any(not isinstance(w, int) for w in width) or \
           any(not isinstance(wd, int) for wd in width_dispersion) or\
           any(not isinstance(cd, int) for cd in center_dispersion):
            raise TypeError

        self.center = center
        self.center_dispersion = center_dispersion
        self.width = width
        self.width_dispersion = width_dispersion
        super(GaussianIrf, self).__init__(label)

    def type_string(self):
        t = "Gaussian"
        if len(self.center) != len(self.width):
            if len(self.width) is 2:
                t = "'Double Gaussian'"
            elif len(self.width) is 3:
                t = "'Triple Gaussian'"
            elif len(self.width) > 3:
                t = "'{} Gaussian'".format(len(self.width))
        elif len(self.center) is not 1:
            t = "'Multiple Gaussian'"
        return t

    def __str__(self):
        s = "{} Center: {} Width: {} Center Dispersion: Width Dispersion {}\
        "
        return s.format(super(GaussianIrf, self).__str__(), self.center,
                        self.width, self.center_dispersion,
                        self.width_dispersion)


class KineticDataset(Dataset):
    """
    KineticDataset is a Dataset with an Irf
    """
    def __init__(self, label, channels, channel_labels, observations):
        self._irf = None
        super(KineticDataset, self).__init__(label, channels, channel_labels,
                                             observations)

    def set_irf(self, irf):
        self._irf = irf

    def irf(self):
        return self._irf

    def __str__(self):
        return "{}\n\t\tIRF: {}".format(super(KineticDataset, self).__str__(),
                                        self.irf())


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

    def __str__(self):
        return "{}\nK-Matrices: {}".format(super(KineticMegacomplex,
                                          self).__str__(), self.k_matrices)


class KMatrix(object):
    """
    A KMatrix has an label and a scipy.sparse matrix
    """
    def __init__(self, label, matrix):
        if not isinstance(label, str) or \
           not isinstance(matrix, dict):
            raise TypeError
        size = 0
        for index in matrix:
            s = max(index)
            if s > size:
                size = s
        m = dok_matrix((size, size), dtype=np.int32)
        for index in matrix:
            m[index[0]-1, index[1]-1] = matrix[index]
        self._label = label
        self._matrix = m

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

    def type_string(self):
        return "Kinetic"

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
        if irf.label() in self.irfs:
            raise Exception("Irf labels must be unique.")
        self.irfs[irf.label()] = irf

    def __str__(self):
        s = "{}\n\nK-Matrices\n----------\n\n".format(super(KineticModel,
                                                            self).__str__())
        for k in self.k_matrices:
            s += "{}\n\n".format(self.k_matrices[k])

        s += "\n\nIRFs\n----\n\n"
        for irf in self.irfs:
            s += "\n{}".format(self.irfs[irf])
        return s
