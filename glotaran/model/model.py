from .parameter import Parameter
from .parameter_constraints import ParameterConstraint
from .compartment_constraints import CompartmentConstraint
from .relation import Relation
from .dataset_descriptor import DatasetDescriptor
from .initial_concentration import InitialConcentration
from .megacomplex import Megacomplex


class Model(object):
    """
    Model represents a global analysis model.

    Consists of parameters, megacomplexes, relations and constraints.
    """

    def __init__(self):
        self._compartments = None
        self._parameter = []
        self._megacomplexes = {}
        self._relations = None
        self._compartment_constraints = None
        self._parameter_constraints = None
        self._datasets = None
        self._initial_concentrations = None

    def type_string(self):
        raise NotImplementedError

    def eval(dataset, axies):
        raise NotImplementedError

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
    def parameter(self, parameter):
        if not isinstance(parameter, list):
            parameter = [parameter]

        for p in parameter:
            if not isinstance(p, Parameter):
                raise TypeError
            p.index = len(self._parameter)+1
            self._parameter.append(p)

    def add_parameter(self, parameter):
        if not isinstance(parameter, Parameter):
            raise TypeError
        parameter.index = len(self._parameter)+1
        self._parameter.append(parameter)

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
    def relations(self):
        return self._relations

    @relations.setter
    def relations(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(val, Relation) for val in value):
            raise TypeError("Relations must be instance of class 'Relation'")
        self._relations = value

    def add_relation(self, relation):
        if not isinstance(relation, Relation):
            raise TypeError("Relations must be instance of class 'Relation'")
        if self.relations is not None:
            self.relations.append(relation)
        else:
            self.relations = relation

    @property
    def parameter_constraints(self):
        return self._parameter_constraints

    @parameter_constraints.setter
    def parameter_constraints(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(val, ParameterConstraint) for val in value):
            raise TypeError("ParameterConstraint must be instance of class"
                            " 'ParameterConstraint'")
        self._parameter_constraints = value

    def add_parameter_constraint(self, constraint):
        if not isinstance(constraint, ParameterConstraint):
            raise TypeError("ParameterConstraint must be instance of class"
                            " 'ParameterConstraint'")
        if self.parameter_constraints is not None:
            self.parameter_constraints.append(constraint)
        else:
            self.parameter_constraints = constraint

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

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, value):
        if not isinstance(value, dict):
            raise TypeError("Datasets must be dict.")
        if any(not issubclass(type(value[val]),
                              DatasetDescriptor) for val in value):
            raise TypeError("Dataset must be subclass of 'DatasetDescriptor'")
        self._datasets = value

    def add_dataset(self, dataset):
        if not issubclass(type(dataset), DatasetDescriptor):
            raise TypeError("Dataset must be subclass of 'DatasetDescriptor'")
        if self.datasets is not None:
            self.datasets[dataset.label] = dataset
        else:
            self.datasets = {dataset.label: dataset}

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

        s += "Parameter\n---------\n\n"

        for p in self.parameter:
            s += "{}\n".format(p)

        if self.parameter_constraints is not None:
            s += "\nParameter Constraints\n--------------------\n\n"

            for p in self._parameter_constraints:
                s += "{}\n".format(p)

        if self.relations is not None:
            s += "\nParameter Relations\n------------------\n\n"

            for p in self._relations:
                s += "{}\n".format(p)

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
