from enum import Enum


class CompartmentConstraintType(Enum):
    """ """
    zero = 0
    equal = 1
    equal_area = 2


class CompartmentConstraint(object):
    """A compartment constraint is applied on one compartment on one or many
    intervals on the estimated axies type.

    Parameters
    ----------
    compartment: label of the compartment
    intervals: list of tuples representing intervals on the estimated axies
    """
    def __init__(self, compartment, intervals):
        self.compartment = compartment
        self.intervals = intervals

    @property
    def compartement(self):
        """label of the compartment"""
        return self._compartement

    @compartement.setter
    def compartment(self, value):
        """

        Parameters
        ----------
        value : label of the compartment

        """
        if not isinstance(value, int):
            raise TypeError("Compartment must be an integer")
        self._compartement = value

    @property
    def intervals(self):
        """ """
        return self._intervals

    @intervals.setter
    def intervals(self, value):
        """

        Parameters
        ----------
        value : list of tuples representing intervals on the estimated axies
        """
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(val, tuple) for val in value):
            raise TypeError("Intervals must be tuples or list of tuples.")
        if any(len(val) is not 2 for val in value):
            raise ValueError("Intervals must be of length 2")
        self._intervals = value

    def applies(self, x):
        """
        Returns true if x is in one of the intervals.
        Parameters
        ----------
        x : point on the estimated axies

        """
        return any(interval[0] <= x <= interval[1] for interval in intervals)

    def type(self):
        """ """
        raise NotImplementedError

    def type_string(self):
        """ """
        raise NotImplementedError

    def __str__(self):
        return "Compartment: {}\tType: '{}'\tIntervals: {}"\
                .format(self.compartment,
                        self.type_string(),
                        self.intervals)


class ZeroConstraint(CompartmentConstraint):
    """A zero constraint sets the c matrix of a compartment to 0.

    Parameters
    ----------
    compartment: label of the compartment
    intervals: list of tuples representing intervals on the estimated axies
    """
    def type(self):
        """ """
        return CompartmentConstraintType.zero

    def type_string(self):
        """ """
        return "Zero"


class EqualConstraint(CompartmentConstraint):
    """An equal constraint The compartments c matrix will be replaced with a sum
    of target compartments, each scaled by a parameter.

    C = p1 * C_t1 + p2 * C_t1 + ...

    Parameters
    ----------
    compartment: label of the compartment
    intervals: list of tuples representing intervals on the estimated axies
    targets: list of target compartments
    parameters: list of scaling parameter for the targets
    """
    def __init__(self, compartment, intervals, targets, parameters:
        self.targets = targets
        self.parameters = parameters
        super(EqualConstraint, self).__init__(compartment, intervals)

    def type(self):
        """ """
        CompartmentConstraintType.equal

    @property
    def target(self):
        """ list of target compartments
"""
        return self._target

    @target.setter
    def target(self, value):
        """

        Parameters
        ----------
        value :list of target compartments
        """
        self._targets = value

    @property
    def parameters(self):
        """ list of scaling parameter for the targets
"""
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        """

        Parameters
        ----------
        value : list of scaling parameter for the targets
        """
        if not len(value) == len(self._compartments):
            raise ValueError("number of parameters != nr targets")
        self._parameters = value

    def parameter_and_targets(self):
        """genrates traget and parameter pairs """
        for i in range(len(self.parameters)):
            yield self.parameters[i], self.targets[i]

    def type_string(self):
        """ """
        return "Equal"

    def __str__(self):
        return "{} Target: {} Parameter: {}".format(super(EqualConstraint,
                                                          self).__str__(),
                                                    self.target,
                                                    self.parameter)


class EqualAreaConstraint(EqualConstraint):
    """An equal area constraint adds a the differenc of the sum of a
    compartements in the e matrix in one ore more intervals to the scaled sum of the
    e matrix of one or more target compartmants to resiudal. The additional
    residual is scaled with the weight.

    Parameters
    ----------
    compartment: label of the compartment
    intervals: list of tuples representing intervals on the estimated axies
    targets: list of target compartments
    parameters: list of scaling parameter for the targets
    weight: scaling factor for the residual
    """
    def __init__(self, compartment, intervals, target, parameter, weight):
        self.weight = weight
        super(EqualAreaConstraint, self).__init__(compartment, intervals,
                                                  target, parameter)

    @property
    def weight(self):
        """scaling factor for the residual"""
        return self._weight

    @weight.setter
    def weight(self, value):
        """

        Parameters
        ----------
        value : scaling factor for the residual
        """
        if not isinstance(value, float):
            raise TypeError("Weight must be float.")
        self._weight = value

    def type(self):
        """ """
        return CompartmentConstraintType.equal_area

    def type_string(self):
        """ """
        return "Equal Area"

    def __str__(self):
        return "{} Weight: {}".format(super(EqualAreaConstraint,
                                            self).__str__(), self.weight)
