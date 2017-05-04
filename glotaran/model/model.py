from collections import OrderedDict

from glotaran.fitmodel import FitModel

from .compartment_constraints import CompartmentConstraint
from .dataset_descriptor import DatasetDescriptor
from .initial_concentration import InitialConcentration
from .megacomplex import Megacomplex
from .parameter_leaf import ParameterLeaf


ROOT_BLOCK_LABEL = "p"


class Model(object):
    """
    Model represents a global analysis model.

    Consists of parameters, megacomplexes, relations and constraints.
    """

    def __init__(self):

        self._compartment_constraints = None
        self._compartments = None
        self._datasets = OrderedDict()
        self._initial_concentrations = None
        self._megacomplexes = {}
        self._parameter = None

    def type_string(self):
        raise NotImplementedError

    def eval(dataset, axies, parameter=None):
        raise NotImplementedError

    def calculated_matrix(self):
        raise NotImplementedError

    def estimated_matrix(self):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        if any([dset.data is None for _, dset in self.datasets.items()]):
            raise Exception("Model datasets not initialized")
        return self.fit_model().fit(*args, **kwargs)

    def fit_model(self):
        return FitModel(self)

    @property
    def compartments(self):
        return self._compartments

    @compartments.setter
    def compartments(self, value):
        self._compartments = value

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, val):
        if not isinstance(val, ParameterLeaf):
            raise TypeError
        self._parameter = val

    @property
    def megacomplexes(self):
        return self._megacomplexes

    @megacomplexes.setter
    def megacomplexes(self, value):
        if not isinstance(value, dict):
            raise TypeError("Megacomplexes must be dict.")
        if any(not issubclass(type(value[val]), Megacomplex) for val in value):
            raise TypeError("Megacomplexes must be subclass of 'Megacomplex'")
        self._megacomplexes = value

    def add_megacomplex(self, megacomplex):
        if not issubclass(type(megacomplex), Megacomplex):
            raise TypeError("Megacomplexes must be subclass of 'Megacomplex'")
        if self.megacomplexes is not None:
            if megacomplex.label in self.megacomplexes:
                raise Exception("Megacomplex labels must be unique")
            self.megacomplexes[megacomplex.label] = megacomplex
        else:
            self.megacomplexes = {megacomplex.label: megacomplex}

    @property
    def compartment_constraints(self):
        return self._compartment_constraints

    @compartment_constraints.setter
    def compartment_constraints(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(val, CompartmentConstraint) for val in value):
            raise TypeError("CompartmentConstraint must be instance of class"
                            " 'CompartmentConstraint'")
        self._compartment_constraints = value

    def add_compartment_constraint(self, constraint):
        if not issubclass(type(constraint), CompartmentConstraint):
            raise TypeError("CompartmentConstraint must be instance of class"
                            " 'CompartmentConstraint'")
        if self.compartment_constraints is not None:
            self.compartment_constraints.append(constraint)
        else:
            self.compartment_constraints = constraint

    def data(self):
        for _, d in self.datasets.items():
            yield d.data

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, value):
        if not isinstance(value, OrderedDict):
            raise TypeError("Datasets must be Ordered.")
        if any(not issubclass(type(value[val]),
                              DatasetDescriptor) for val in value):
            raise TypeError("Dataset must be subclass of 'DatasetDescriptor'")
        self._datasets = value

    def add_dataset(self, dataset):
        if not issubclass(type(dataset), DatasetDescriptor):
            raise TypeError("Dataset must be subclass of 'DatasetDescriptor'")
        self.datasets[dataset.label] = dataset

    def set_data(self, label, data):
        self.datasets[label].data = data

    @property
    def initial_concentrations(self):
        return self._initial_concentrations

    @initial_concentrations.setter
    def initial_concentrations(self, value):
        if not isinstance(value, dict):
            raise TypeError("Initial concentrations must be dict.")
        if any(not isinstance(value[val],
                              InitialConcentration) for val in value):
            raise TypeError("Initial concentrations must be instance of"
                            " 'InitialConcentration'")
        self._initial_concentrations = value

    def add_initial_concentration(self, initial_concentration):
        if not isinstance(initial_concentration, InitialConcentration):
            raise TypeError("Initial concentrations must be instance of"
                            " 'InitialConcentration'")
        if self.initial_concentrations is not None:
            self.initial_concentrations[initial_concentration.label] =\
                initial_concentration
        else:
            self.initial_concentrations = {initial_concentration.label:
                                           initial_concentration}

    def __str__(self):
        s = "Modeltype: {}\n\n".format(self.type_string())

        s += "Parameter\n---------\n{}\n".format(self.parameter)

        if self.compartments is not None:
            s += "\nCompartments\n-------------------------\n\n"

            s += "{}\n".format(self.compartments)

        s += "\nMegacomplexes\n-------------\n\n"

        for m in self._megacomplexes:
            s += "{}\n".format(self._megacomplexes[m])

        if self.compartment_constraints is not None:
            s += "\nCompartment Constraints\n------------------------\n\n"

            for c in self._compartment_constraints:
                s += "{}\n".format(c)

        if self.initial_concentrations is not None:
            s += "\nInitital Concentrations\n-----------------------\n\n"

            for i in self._initial_concentrations:
                s += "{}\n".format(self._initial_concentrations[i])

        if self.datasets is not None:
            s += "\nDatasets\n--------\n\n"

            for d in self._datasets:
                s += "{}\n".format(self._datasets[d])

        return s
