from glotaran.model import Model
from glotaran.model import Dataset
from .megacomplex import KineticMegacomplex
from .k_matrix import KMatrix
from .irf import Irf
from .separable_model import KineticSeparableModel


class KineticModel(Model):
    """
    A kinetic model is a model with one ore more Irfs, K-Matrices and
    KineticMegacomplexes.
    """
    def __init__(self):
        self.k_matrices = {}
        self.irfs = {}
        self.dispersion_center = None
        super(KineticModel, self).__init__()

    def type_string(self):
        return "Kinetic"

    def add_megakomplex(self, megacomplex):
        if not isinstance(megacomplex, KineticMegacomplex):
            raise TypeError
        super(KineticModel).add_megakomplex(megacomplex)

    @property
    def dispersion_center(self):
        if self._dispersion_center is None:
            for d in self.data():
                return d.get_axis("spec")[0]
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

    def __str__(self):
        s = "{}\n\nK-Matrices\n----------\n\n".format(super(KineticModel,
                                                            self).__str__())
        for k in self.k_matrices:
            s += "{}\n".format(self.k_matrices[k])

        s += "\n\nIRFs\n----\n\n"
        for irf in self.irfs:
            s += "{}\n".format(self.irfs[irf])
        return s

    def eval(self, parameter, dataset, axes, **kwargs):
        data = Dataset(dataset)
        for label, val in axes.items():
            data.set_axis(label, val)
        self.set_data(dataset, data)
        fitmodel = KineticSeparableModel(self)

        kwargs['dataset'] = dataset
        data = fitmodel.eval(parameter, **kwargs)
        self.datasets[dataset].data.data = data
