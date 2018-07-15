import numpy as np

from .compartment_constraints import CompartmentConstraint
from .dataset import Dataset


class DatasetDescriptor(object):
    """Class representing a dataset for fitting."""

    def __init__(self, label, initial_concentration, megacomplexes,
                 megacomplex_scaling, dataset_scaling, compartment_scaling,
                 compartement_constraints):
        self.label = label
        self.initial_concentration = initial_concentration
        self.megacomplexes = megacomplexes
        self.compartment_scaling = compartment_scaling
        self.megacomplex_scaling = megacomplex_scaling
        self.scaling = dataset_scaling
        self.data = None
        self.compartment_constraints = compartement_constraints

    @property
    def label(self):
        """label of the dataset"""
        return self._label

    @label.setter
    def label(self, value):
        """

        Parameters
        ----------
        value : label of the dataset
        """
        self._label = value

    @property
    def compartment_constraints(self):
        """a list of compartment constraints"""
        return self._compartment_constraints

    @compartment_constraints.setter
    def compartment_constraints(self, value):
        """

        Parameters
        ----------
        value : a list of compartment constraints
        """
        if not isinstance(value, list):
            value = [value]
        if any(not issubclass(type(val), CompartmentConstraint)
               for val in value):
            raise TypeError("CompartmentConstraint must be implementations of "
                            "model.CompartmentConstraint")
        self._compartment_constraints = value

    @property
    def data(self):
        """implementation of model.Dataset"""
        return self._data

    @data.setter
    def data(self, data):
        """

        Parameters
        ----------
        data : implementation of model.Dataset
        """
        if not isinstance(data, Dataset) and data is not None:
            raise TypeError
        self._data = data

    @property
    def initial_concentration(self):
        """list of labels of initial concentrations"""
        return self._initial_concentration

    @initial_concentration.setter
    def initial_concentration(self, value):
        """

        Parameters
        ----------
        value : list of labels of initial concentrations
        """
        self._initial_concentration = value

    @property
    def scaling(self):
        """parameter to scale the datasets c matrix"""
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        """

        Parameters
        ----------
        scaling : parameter to scale the datasets c matrix
        """
        if not isinstance(scaling, int) and scaling is not None:
            raise TypeError("Parameter index must be numerical")
        self._scaling = scaling

    @property
    def compartment_scaling(self):
        """dict of compartment parameter pairs"""
        return self._compartment_scaling

    @compartment_scaling.setter
    def compartment_scaling(self, scaling):
        """

        Parameters
        ----------
        scaling : dict of compartment parameter pairs
        """
        if not isinstance(scaling, dict):
            raise TypeError
        self._compartment_scaling = scaling

    @property
    def megacomplexes(self):
        """list of megacomplex labels"""
        return self._megacomplexes

    @megacomplexes.setter
    def megacomplexes(self, megacomplex):
        """

        Parameters
        ----------
        megacomplex : list of megacomplex labels
        """
        if not isinstance(megacomplex, list):
            megacomplex = [megacomplex]
        if any(not isinstance(m, str) for m in megacomplex):
            raise TypeError("Megacomplex labels must be string.")
        self._megacomplexes = megacomplex

    @property
    def megacomplex_scaling(self):
        """dict of megacomplex paramater pairs"""
        return self._megacomplex_scaling

    @megacomplex_scaling.setter
    def megacomplex_scaling(self, scaling):
        """

        Parameters
        ----------
        scaling : dict of megacomplex paramater pairs
        """
        if not isinstance(scaling, dict):
            raise TypeError("Megacomplex Scaling must by dict, got"
                            "{}".format(type(scaling)))
        self._megacomplex_scaling = scaling

    def __str__(self):
        s = "Dataset '{}'\n\n".format(self.label)

        s += "\tDataset Scaling: {}\n".format(self.scaling)

        s += "\tInitial Concentration: {}\n"\
            .format(self.initial_concentration)

        s += "\tMegacomplexes: {}\n".format(self.megacomplexes)

        s += "\tMega scalings:\n"
        for cmplx, scale in self._megacomplex_scaling.items():
            s += "\t\t- {}:{}\n".format(cmplx, scale)

        return s

    def svd(self):
        lsvd, svals, rsvd = np.linalg.svd(self.data.get())
        return lsvd, svals, rsvd
