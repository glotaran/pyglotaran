class Parameter(object):
    """
    A parameter has an initial value and an optional label.
    """
    def __init__(self, initial, label=None):
        self.value = initial
        self.label = label

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, i):
        self._index = i

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if not isinstance(val, (int, float)):
            raise Exception("Parameter value must be numeric.")

        if isinstance(val, int):
            val = float(val)

        self._value = val

    def __str__(self):
        return 'Index: {} Initial Value: {} Label: {}'.format(self._index,
                                                              self.value,
                                                              self.label)


def create_parameter_list(parameter):
    if not isinstance(parameter, list):
        raise TypeError
    parameterlist = []
    for p in parameter:
        if isinstance(p, (float, int)):
            parameterlist.append(Parameter(p))
        elif isinstance(p, list):
            parameterlist.append(Parameter(p[1], label=p[0]))
        else:
            raise TypeError
    return parameterlist
