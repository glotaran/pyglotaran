from glotaran.model import Model
from glotaran.fitmodel import FitModel

from .dataset import SpectralTemporalDataset
from .irf import Irf
from .k_matrix import KMatrix
from .kinetic_c_matrix import KineticCMatrix
from .megacomplex import KineticMegacomplex
from .spectral_c_matrix import SpectralCMatrix
from .spectral_shape import SpectralShape


class KineticModel(Model):
    """
    A kinetic model is a model with one ore more Irfs, K-Matrices and
    KineticMegacomplexes.
    """
    def __init__(self):
        self.k_matrices = {}
        self.irfs = {}
        self.shapes = {}
        self.dispersion_center = None
        super(KineticModel, self).__init__()

    def type_string(self):
        return "Kinetic"

    def calculated_matrix(self):
        return KineticCMatrix

    def estimated_matrix(self):
        return SpectralCMatrix

    def add_megakomplex(self, megacomplex):
        if not isinstance(megacomplex, KineticMegacomplex):
            raise TypeError
        super(KineticModel).add_megakomplex(megacomplex)

    @property
    def dispersion_center(self):
        if self._dispersion_center is None:
            for d in self.data():
                return d.spectral_axis[0]
        else:
            return self._dispersion_center

    @dispersion_center.setter
    def dispersion_center(self, value):
        self._dispersion_center = value

    @property
    def k_matrices(self):
        return self._k_matrices

    @k_matrices.setter
    def k_matrices(self, value):
        if not isinstance(value, dict):
            raise TypeError("K-Matrices must be dict.")
        if any(not isinstance(type(val), KMatrix) for val in value):
            raise TypeError("K-Matrices must be subclass of 'KMatrix'")
        self._k_matrices = value

    def add_k_matrix(self, k_matrix):
        if not issubclass(type(k_matrix), KMatrix):
            raise TypeError("K-Matrix must be subclass of 'KMatrix'")
        if self.k_matrices is None:
            self.k_matrices = {k_matrix.label: k_matrix}
        else:
            if k_matrix.label in self.k_matrices:
                raise Exception("K-Matrix labels must be unique")
            self.k_matrices[k_matrix.label] = k_matrix

    @property
    def irfs(self):
        return self._irfs

    @irfs.setter
    def irfs(self, value):
        if not isinstance(value, dict):
            raise TypeError("Irfs must be dict.")
        if any(not isinstance(type(val), Irf) for val in value):
            raise TypeError("K-Matrices must be subclass of 'KMatrix'")
        self._irfs = value

    def add_irf(self, irf):
        if not issubclass(type(irf), Irf):
            raise TypeError("Irfs must be subclass of 'Irf'")
        if self.irfs is None:
            self.irfs = {irf.label: irf}
        else:
            if irf.label in self.irfs:
                raise Exception("Irfs labels must be unique")
            self.irfs[irf.label] = irf

    @property
    def shapes(self):
        return self._shapes

    @shapes.setter
    def shapes(self, value):
        if not isinstance(value, dict):
            raise TypeError("Shapes must be dict.")
        if any(not isinstance(type(val), Irf) for val in value):
            raise TypeError("Values must be subclass of 'Shape'")
        self._shapes = value

    def add_shape(self, shape):
        if not issubclass(type(shape), SpectralShape):
            raise TypeError("Shape must be subclass of 'Shape'")
        if self.shapes is None:
            self.shapes = {shape.label: shape}
        else:
            if shape.label in self.shapes:
                raise Exception("Shape labels must be unique")
            self.shapes[shape.label] = shape

    def __str__(self):
        s = "{}\n\nK-Matrices\n----------\n\n".format(super(KineticModel,
                                                            self).__str__())
        for k in self.k_matrices:
            s += "{}\n".format(self.k_matrices[k])

        s += "\n\nIRFs\n----\n\n"
        for irf in self.irfs:
            s += "{}\n".format(self.irfs[irf])

        s += "\nShapes\n----\n\n"
        for _, shape in self.shapes.items():
            s += "{}\n".format(shape)
        return s

    def simulate(self, dataset, axes, parameter=None):
        data = SpectralTemporalDataset(dataset)
        sim_parameter = self.parameter.as_parameters_dict().copy()
        if parameter is not None:
            for k, v in parameter.items():
                k = "p_" + k.replace(".", "_")
                sim_parameter[k].value = v
        for label, val in axes.items():
            data.set_axis(label, val)
        self.set_data(dataset, data)
        fitmodel = FitModel(self)

        kwargs = {}
        kwargs['dataset'] = dataset
        data = fitmodel.eval(sim_parameter, **kwargs)
        self.datasets[dataset].data.set(data)

    def c_matrix(self, parameter=None):
        if parameter is None:
            parameter = self.parameter.as_parameters_dict()
        return self.fit_model().c_matrix(parameter)

    def e_matrix(self, parameter=None):
        if parameter is None:
            parameter = self.parameter.as_parameters_dict()
        return self.fit_model().e_matrix(parameter)
