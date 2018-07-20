"""Glotaran Kinetic Model"""

from typing import Type, Dict
from glotaran.model import Model
# unused import
# from glotaran.model import DatasetDescriptor
from glotaran.fitmodel import FitModel, Matrix

from .dataset import SpectralTemporalDataset
# unused import
# from .dataset_descriptor import SpectralTemporalDatasetDescriptor
from .irf import Irf
from .fitmodel import KineticFitModel
from .k_matrix import KMatrix
from .kinetic_matrix import KineticMatrix
from .megacomplex import KineticMegacomplex
from .spectral_matrix import SpectralMatrix
from .spectral_shape import SpectralShape


class KineticModel(Model):
    """A kinetic model is an implementation for model.Model. It is used describe
    time dependend datasets.

    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=attribute-defined-outside-init
    # Models are complex.
    def __init__(self):
        """ """
        self.k_matrices = {}
        self.irfs = {}
        self.shapes = {}
        super(KineticModel, self).__init__()

    def type_string(self) -> str:
        """Returns a human readable string identifying the type of the model.

        Returns
        -------

        type : str
            Type of the Model

        """
        return "Kinetic"

    def fit_model_class(self) -> Type[FitModel]:
        """Returns a kinetic fitmodel.

        Returns
        -------

        fitmodel : type(KineticFitModel)
            Implementation of fitmodel.FitModel
        """
        return KineticFitModel

    def calculated_matrix(self) -> Type[Matrix]:
        """Returns Kinetic C the calculated matrix.

        Returns
        -------

        matrix : type(fitmodel.Matrix)
            Calculated Matrix
        """
        return KineticMatrix

    def estimated_matrix(self) -> Type[Matrix]:
        """Returns the estimated matrix.

        Returns
        -------

        matrix : np.array
            Estimated Matrix
        """
        return SpectralMatrix

    def dataset_class(self) -> Type[SpectralTemporalDataset]:
        """Returns an implementation for model.Dataset

        Returns
        -------

        descriptor : type(model.Dataset)
            Implementation of model.Dataset
        """
        return SpectralTemporalDataset

    def add_megacomplex(self, megacomplex: KineticMegacomplex):
        """Adds a kinetic megacomplex to the model.

        Parameters
        ----------
        megacomplex : KineticMegacomplex


        """
        if not isinstance(megacomplex, KineticMegacomplex):
            raise TypeError
        super(KineticModel, self).add_megacomplex(megacomplex)

    @property
    def k_matrices(self) -> Dict[str, KMatrix]:
        """A dictonary of the models K-matrices"""
        return self._k_matrices

    @k_matrices.setter
    def k_matrices(self, value):
        if not isinstance(value, dict):
            raise TypeError("K-Matrices must be dict.")
        if any(not isinstance(type(val), KMatrix) for val in value):
            raise TypeError("K-Matrices must be subclass of 'KMatrix'")
        self._k_matrices = value

    def add_k_matrix(self, k_matrix):
        """

        Parameters
        ----------
        k_matrix :


        Returns
        -------

        """
        if not issubclass(type(k_matrix), KMatrix):
            raise TypeError("K-Matrix must be subclass of 'KMatrix'")
        if self.k_matrices is None:
            self.k_matrices = {k_matrix.label: k_matrix}
        else:
            if k_matrix.label in self.k_matrices:
                raise Exception("K-Matrix labels must be unique")
            self.k_matrices[k_matrix.label] = k_matrix

    @property
    def irfs(self) -> Dict[str, Irf]:
        """A dictonary of the model's IRFs"""
        return self._irfs

    @irfs.setter
    def irfs(self, value):
        if not isinstance(value, dict):
            raise TypeError("Irfs must be dict.")
        if any(issubclass(type(val), Irf) for val in value):
            raise TypeError("K-Matrices must be subclass of 'KMatrix'")
        self._irfs = value

    def add_irf(self, irf: Irf):
        """Adds an IRF to model

        Parameters
        ----------
        irf : Irf


        """
        if not issubclass(type(irf), Irf):
            raise TypeError("Irfs must be subclass of 'Irf'")
        if self.irfs is None:
            self.irfs = {irf.label: irf}
        else:
            if irf.label is None or irf.label is "":
                raise Exception("Irf label empty")
            if irf.label in self.irfs:
                raise Exception("Irf label must be unique")
            self.irfs[irf.label] = irf

    @property
    def shapes(self) -> Dict[str, SpectralShape]:
        """A Dictonary of the model's spectral shapes."""
        return self._shapes

    @shapes.setter
    def shapes(self, value):
        if not isinstance(value, dict):
            raise TypeError("Shapes must be dict.")
        if any(issubclass(type(val), SpectralShape) for val in value):
            raise TypeError("Values must be subclass of 'Shape'")
        self._shapes = value

    def add_shape(self, shape: SpectralShape):
        """Adds a spectral shape to the model.

        Parameters
        ----------
        shape : SpectralShape


        """
        if not issubclass(type(shape), SpectralShape):
            raise TypeError("Shape must be subclass of 'Shape'")
        if self.shapes is None:
            self.shapes = {shape.label: shape}
        else:
            if shape.label in self.shapes:
                raise Exception("Shape labels must be unique")
            self.shapes[shape.label] = shape

    def __str__(self):
        string = "{}\n\nK-Matrices\n----------\n\n" \
                 "".format(super(KineticModel, self).__str__())
        for k in self.k_matrices:
            string += "{}\n".format(self.k_matrices[k])

        string += "\n\nIRFs\n----\n\n"
        for irf in self.irfs:
            string += "{}\n".format(self.irfs[irf])

        string += "\nShapes\n----\n\n"
        for _, shape in self.shapes.items():
            string += "{}\n".format(shape)
        return string

    #  def c_matrix(self, parameter=None):
    #      """
    #
    #      Parameters
    #      ----------
    #      parameter :
    #           (Default value = None)
    #
    #      Returns
    #      -------
    #
    #      """
    #      if parameter is None:
    #          parameter = self.parameter.as_parameters_dict()
    #      return self.fit_model().c_matrix(parameter)
    #
    #  def e_matrix(self, parameter=None):
    #      """
    #
    #      Parameters
    #      ----------
    #      parameter :
    #           (Default value = None)
    #
    #      Returns
    #      -------
    #
    #      """
    #      if parameter is None:
    #          parameter = self.parameter.as_parameters_dict()
    #      return self.fit_model().e_matrix(parameter)
