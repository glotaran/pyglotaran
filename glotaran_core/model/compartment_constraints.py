from enum import Enum


class CompartmentConstraintType(Enum):
    zero = 0
    equal = 1
    equal_area = 2


class CompartmentConstraint(object):
    """
    A CompartmentConstraint has a compartment, one or many intervals and a
    type.
    """
    def __init__(self, compartment, intervals):
        self.compartment = compartment
        self.intervals = intervals

    @property
    def compartement(self):
        return self._compartement

    @compartement.setter
    def compartment(self, value):
        if not isinstance(value, int):
            raise TypeError("Compartment must be an integer")
        self._compartement = value

    @property
    def intervals(self):
        return self._intervals

    @intervals.setter
    def intervals(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(val, tuple) for val in value):
            raise TypeError("Intervals must be tuples or list of tuples.")
        if any(len(val) is not 2 for val in value):
            raise ValueError("Intervals must be of length 2")
        self._intervals = value

    def type(self):
        raise NotImplementedError

    def type_string(self):
        raise NotImplementedError

    def __str__(self):
        return "Type: '{}' Intervals: {}".format(self.type_string(),
                                                 self.intervals)


class ZeroConstraint(CompartmentConstraint):
    def type(self):
        return CompartmentConstraintType.zero

    def type_string(self):
        return "Zero"


class EqualConstraint(CompartmentConstraint):
    """
    An equal constraint is a CompartmentConstraint with a target compartment
    and a parameter.
    """
    def __init__(self, compartment, intervals, target, parameter):
        self.target = target
        self.parameter = parameter
        super(EqualConstraint, self).__init__(compartment, intervals)

    def type(self):
        CompartmentConstraintType.equal

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        if not isinstance(value, int):
            raise TypeError("Target must be integer.")
        self._target = value

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        if not isinstance(value, int):
            raise TypeError("Parameter must be integer.")
        self._parameter = value

    def type_string(self):
        return "Equal"

    def __str__(self):
        return "{} Target: {} Parameter: {}".format(super(EqualConstraint,
                                                          self).__str__(),
                                                    self.target,
                                                    self.parameter)


class EqualAreaConstraint(EqualConstraint):
    """
    An equal area constraint is a CompartmentConstraint with a target
    compartment, a parameter and a weigth.
    """
    def __init__(self, compartment, intervals, target, parameter, weight):
        self.weight = weight
        super(EqualAreaConstraint, self).__init__(compartment, intervals,
                                                  target, parameter)

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        if not isinstance(value, float):
            raise TypeError("Weight must be float.")
        self._weight = value

    def type(self):
        return CompartmentConstraintType.equal_area

    def type_string(self):
        return "Equal Area"

    def __str__(self):
        return "{} Weight: {}".format(super(EqualAreaConstraint,
                                            self).__str__(), self.weight)
