from lmfit import Parameter as LmParameter


class Parameter(object):
    """
    A parameter has an initial value and an optional label.
    """
    def __init__(self, initial, label=None):
        self.value = initial
        self.label = label
        self._index = -1

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, i):
        self._index = i

    @property
    def label(self):
        if self._label is None:
            return self.index
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):

        if not isinstance(val, (int, float)) and val != 'nan':
            if isinstance(val, str):
                try:
                    val = float(val)
                except:
                    raise Exception("Parameter Error: value must be numeric:"
                                    "{} Type: {}".format(val, type(val)))

        if isinstance(val, int):
            val = float(val)

        self._value = val

    def as_lmfit_parameter(self, root):
        name = "{}.{}".format(root, self.label)
        return LmParameter(name=name, value=self.value)

    def __str__(self):
        return 'Index: {} Initial Value: {} Label: {}'.format(self._index,
                                                              self.value,
                                                              self.label)


def create_parameter_list(parameter):
    if not isinstance(parameter, list):  # TODO: consider allowing None
        raise TypeError
    parameterlist = []
    for p in parameter:
        if isinstance(p, (float, int, str)):
            parameterlist.append(Parameter(p))
        elif isinstance(p, list):
            parameterlist.append(Parameter(p[1], label=p[0]))
        else:
            raise TypeError
    return parameterlist
