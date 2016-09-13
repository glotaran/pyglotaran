from enum import Enum


class ParameterConstraintType(Enum):
    fix = 0
    bound = 1


class ParameterConstraint(object):
    """
    A parameter constraint concerns one more parameters and has a type.
    """
    def __init__(self, parameters):
        if isinstance(parameters, list):
            self.range = None
            self.parameters = parameters
        elif isinstance(parameters, tuple):
            self.range = parameters
            if len(parameters) is not 2:
                raise Exception("Size of parameter range must be 2")
            self.parameters = range(parameters[0], parameters[1])
        else:
            raise TypeError

    def type(self):
        raise NotImplementedError

    def type_string(self):
        raise NotImplementedError

    def __str__(self):
        if self.range is None:
            parameter = self.parameters
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

    def type(self):
        return ParameterConstraintType.bound

    def type_string(self):
        return "Bound"

    def __str__(self):
        return "{} Lower: {} Upper: {}".format(super(BoundConstraint,
                                                     self).__str__(),
                                               self.lower, self.upper)


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
        if not isinstance(compartment, int):
            raise TypeError
        if isinstance(intervals, tuple):
            intervals = [intervals]
        if isinstance(intervals, list):
            if any(not isinstance(interval, tuple) for interval in intervals):
                raise TypeError
            self.intervals = intervals
        else:
            raise TypeError

        if any(len(interval) is not 2 for interval in intervals):
            raise Exception("Size of interval must be 2")

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
        if not isinstance(target, int) or not isinstance(parameter, int):
            raise TypeError
        self.target = target
        self.parameter = parameter
        super(EqualConstraint, self).__init__(compartment, intervals)

    def type(self):
        CompartmentConstraintType.equal

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
        if not isinstance(weight, float):
            raise TypeError

        self.weight = weight

        super(EqualAreaConstraint, self).__init__(compartment, intervals,
                                                  target, parameter)

    def type(self):
        return CompartmentConstraintType.equal_area

    def type_string(self):
        return "Equal Area"

    def __str__(self):
        return "{} Weight: {}".format(super(EqualAreaConstraint,
                                            self).__str__(), self.weight)


class Relation(object):
    """
    Relation relates a parameter to constant and/or one or many parameters.

    Parameter
    ---------

    parameter = The index of the realted parameter.
    to = A dictionary where the keys are parameter indices or the string
        "const" and values are numbers to multiply the related parameters.

    Example usage
    -------------

    Relation(86, {'const': 1, 85: -1})

    represents the relation

    Parameter86 = 1-1*Parameter85
    """
    def __init__(self, parameter, to):
        if not isinstance(parameter, int) or not isinstance(to, dict):
            raise TypeError
        if any(not isinstance(key, int) and key is not 'const' for key in to):
            raise Exception("""Keys in 'to' must either be of type int or
                            'const'""")
        if any(not isinstance(to[key], int) and not isinstance(to[key], float)
               for key in to):
            raise TypeError

        self.parameter = parameter
        self.to = to

    def __str__(self):
        s = "Parameter {} =".format(self.parameter)

        for t in self.to:
            if t is "const":
                if self.to[t] < 0:
                    s += "{}".format(self.to[t])
                else:
                    s += "+{}".format(self.to[t])
            else:
                if self.to[t] < 0:
                    s += "{}*P{}".format(self.to[t], t)
                else:
                    s += "+{}*P{}".format(self.to[t], t)
        return s
