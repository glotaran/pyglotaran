"""Dataset Descriptor"""

from typing import Dict, List
# unused imports
# from typing import Tuple
# import numpy as np

from .compartment_constraints import CompartmentConstraint
from .dataset import Dataset


class DatasetDescriptor:
    """Represents a dataset for fitting"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=attribute-defined-outside-init
    # Datasets are complex.

    def __init__(self,
                 label: str,
                 initial_concentration: str,
                 megacomplexes: List[str],
                 megacomplex_scaling: Dict[str, List[str]],
                 scaling: str,
                 compartment_scaling: Dict[str, List[str]],
                 compartment_constraints: List[CompartmentConstraint]):
        """

        Parameters
        ----------
        label : str
            The label of the dataset.

        initial_concentration : str
            The label of the initial concentration

        megacomplexes : List[str]
            A list of megacomplex labels

        megacomplex_scaling : Dict[str: List[str]]
            The megacomplex scaling parameters

        scaling : str
            The scaling parameter for the dataset

        compartment_scaling: Dict[str: List[str]]
            The compartment scaling parameters

        compartment_constraints: List[CompartmentConstraint] :
            A list of compartment constraints

        """
        self.label = label
        self.initial_concentration = initial_concentration
        self.megacomplexes = megacomplexes
        self.compartment_scaling = compartment_scaling
        self.megacomplex_scaling = megacomplex_scaling
        self.scaling = scaling
        self.dataset = None
        self.compartment_constraints = compartment_constraints

    @property
    def label(self):
        """The label of the dataset"""
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def compartment_constraints(self):
        """A list of compartment constraints"""
        return self._compartment_constraints

    @compartment_constraints.setter
    def compartment_constraints(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not issubclass(type(val), CompartmentConstraint)
               for val in value):
            raise TypeError("CompartmentConstraint must be implementations of "
                            "model.CompartmentConstraint")
        self._compartment_constraints = value

    @property
    def dataset(self):
        """An implementation of model.Dataset"""
        return self._data

    @dataset.setter
    def dataset(self, data):
        if not isinstance(data, Dataset) and data is not None:
            raise TypeError
        self._data = data

    @property
    def initial_concentration(self):
        """A list of labels of initial concentrations"""
        return self._initial_concentration

    @initial_concentration.setter
    def initial_concentration(self, value):
        self._initial_concentration = value

    @property
    def scaling(self):
        """A parameter to scale the dataset"""
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        if not isinstance(scaling, int) and scaling is not None:
            raise TypeError("Parameter index must be numerical")
        self._scaling = scaling

    @property
    def compartment_scaling(self):
        """ A dictionary of compartment parameter pairs"""
        return self._compartment_scaling

    @compartment_scaling.setter
    def compartment_scaling(self, scaling):
        if not isinstance(scaling, dict):
            raise TypeError
        self._compartment_scaling = scaling

    @property
    def megacomplexes(self):
        """list of megacomplex labels"""
        return self._megacomplexes

    @megacomplexes.setter
    def megacomplexes(self, megacomplex):
        if not isinstance(megacomplex, list):
            megacomplex = [megacomplex]
        if any(not isinstance(m, str) for m in megacomplex):
            raise TypeError("Megacomplex labels must be string.")
        self._megacomplexes = megacomplex

    @property
    def megacomplex_scaling(self):
        """A dictinary of megacomplex scaling parameters"""
        return self._megacomplex_scaling

    @megacomplex_scaling.setter
    def megacomplex_scaling(self, scaling):
        if not isinstance(scaling, dict):
            raise TypeError("Megacomplex Scaling must by dict, got"
                            "{}".format(type(scaling)))
        self._megacomplex_scaling = scaling

    def __str__(self):
        """ """
        string = "### _{}_\n\n".format(self.label)

        string += "* _Dataset Scaling_: {}\n".format(self.scaling)

        string += "* _Initial Concentration_: {}\n"\
            .format(self.initial_concentration)

        string += "* _Megacomplexes_: {}\n".format(self.megacomplexes)

        string += "* _Megacomplex Scalings_:"
        if self._megacomplex_scaling:
            string += "\n"
            for cmplx, scale in self._megacomplex_scaling.items():
                string += "  * {}:{}\n".format(cmplx, scale)
        else:
            string += " None\n"

        return string
