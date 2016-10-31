from enum import Enum


class ParameterConstraintType(Enum):
    fix = 0
    bound = 1


class ParameterConstraint(object):
    """
    A parameter constraint concerns one more parameters and has a type.
    """
    def __init__(self, parameter):
        self._range = None
        if isinstance(parameter, tuple):
            self.range = parameter
        else:
            self.parameter = parameter

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, val):
        if not isinstance(val, tuple):
            raise TypeError("Range must be tuple")
        if not len(val) == 2:
            raise ValueError("Range must be (FROM, TO)")
        self.parameter = range(val[0], val[1])
        self._range = val

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(isinstance(v, int) for v in value):
            raise TypeError("Parameter mus be an integer or a list of integer")

    def type(self):
        raise NotImplementedError

    def type_string(self):
        raise NotImplementedError

    def __str__(self):
        if self.range is None:
            parameter = self.parameter
        else:
            parameter = self.range
        return "Type: '{}' Parameters: {}".format(self.type_string(),
                                                  parameter)


class FixedConstraint(ParameterConstraint):
    def type(self):
        return ParameterConstraintType.fix

    def type_string(self):
        return "Fixed"


class BoundConstraint(ParameterConstraint):
    """
    Represents a boundary constraint.
    """
    def __init__(self, parameters, lower=float('nan'), upper=float('nan')):
        self.lower = lower
        self.upper = upper
        super(BoundConstraint, self).__init__(parameters)

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Lower bound must be a number.")
        self._lower = value

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Upper bound must be a number.")
        self._upper = value

    def type(self):
        return ParameterConstraintType.bound

    def type_string(self):
        return "Bound"

    def __str__(self):
        return "{} Lower: {} Upper: {}".format(super(BoundConstraint,
                                                     self).__str__(),
                                               self.lower, self.upper)
