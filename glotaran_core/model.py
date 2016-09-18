from .parameter import Parameter
from .megacomplex import Megacomplex
from .constraints import ParameterConstraint, CompartmentConstraint, Relation
from .dataset import Dataset


class Model(object):
    """
    Model represents a global analysis model.

    Consists of parameters, megacomplexes, relations and constraints.
    """
    def __init__(self):
        self._parameters = []
        self._megacomplexes = {}
        self._relations = []
        self._parameter_constraints = []
        self._compartment_constraints = []
        self._datasets = {}

    def type_string(self):
        raise NotImplementedError

    def add_parameter(self, parameter):
        if not isinstance(parameter, list):
            parameter = [parameter]
        for p in parameter:
            if not isinstance(p, Parameter):
                raise TypeError
            p.set_index(len(self._parameters)+1)
            self._parameters.append(p)

    def parameters(self):
        return self._parameters

    def add_megacomplex(self, megacomplex):
        if not issubclass(type(megacomplex), Megacomplex):
            raise TypeError
        if megacomplex.label() in self._megacomplexes:
            raise Exception("Megacomplex labels must be unique")
        self._megacomplexes[megacomplex.label()] = megacomplex

    def add_relation(self, relation):
        if not isinstance(relation, Relation):
            raise TypeError
        self._relations.append(relation)

    def add_parameter_constraint(self, constraint):
        if not issubclass(type(constraint), ParameterConstraint):
            raise TypeError
        self._parameter_constraints.append(constraint)

    def add_compartment_constraint(self, constraint):
        if not issubclass(type(constraint), CompartmentConstraint):
            raise TypeError
        self._compartment_constraints.append(constraint)

    def add_dataset(self, dataset):
        if not issubclass(type(dataset), Dataset):
            raise TypeError
        self._datasets[dataset.label()] = dataset

    def __str__(self):
        s = "Modeltype: {}\n\n".format(self.type_string())

        s += "Parameter\n---------\n\n"

        for p in self.parameters():
            s += "{}\n".format(p)

        s += "\nParameter Constraints\n--------------------\n\n"

        for p in self._parameter_constraints:
            s += "{}\n".format(p)

        s += "\nParameter Relations\n------------------\n\n"

        for p in self._relations:
            s += "{}\n".format(p)

        s += "\nMegacomplexes\n-------------\n\n"

        for m in self._megacomplexes:
            s += "{}\n".format(self._megacomplexes[m])

        s += "\nCompartment Constraints\n------------------------\n\n"

        for c in self._compartment_constraints:
            s += "{}\n".format(c)

        s += "\nDatasets\n--------\n\n"

        for d in self._datasets:
            s += "{}\n".format(self._datasets[d])

        return s
