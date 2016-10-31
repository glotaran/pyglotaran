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
        self.parameter = parameter
        self.to = to

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        if not isinstance(value, int):
            raise TypeError("Parameter must be integer.")
        self._parameter = value

    @property
    def to(self):
        return self._to

    @to.setter
    def to(self, value):
        if isinstance(value, int):
            value = {value: 1}
        if not isinstance(value, dict):
            raise TypeError("To must be dict or int.")
        if any(not isinstance(key, int) and key != 'const' for key in value):
            raise ValueError("""Keys in 'to' must either be of type int or
                             'const'""")
        if any(not isinstance(value[key], (int, float)) for key in value):
            raise TypeError("Values in 'to' must be numbers.")
        self._to = value

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
