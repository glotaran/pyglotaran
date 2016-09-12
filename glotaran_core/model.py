from .parameter import Parameter
from .megacomplex import Megacomplex
from .constraints import ParameterConstraint, CompartmentConstraint, Relation


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
        self._compartement_constraints = []

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

    def add_megakomplex(self, megacomplex):
        if not issubclass(type(megacomplex), Megacomplex):
            raise TypeError
        if megacomplex.label() in self.megacomplexes:
            raise Exception("Megacomplex labels must be unique")
        self.megacomplexes[megacomplex.label()] = megacomplex

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

    def __str__(self):
        s = "Parameter\n---------\n\n"

        for p in self.parameters():
            s += "{}\n".format(p)

        return s
