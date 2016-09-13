class Parameter(object):
    """
    A parameter has an initial value and an optional label.
    """
    def __init__(self, initial, label=None):
        if not isinstance(initial, int) and not isinstance(initial, float):
            raise Exception("Parameter values bust be numeric.")

        if isinstance(initial, int):
            initial = float(initial)

        self.initial = initial
        self.label = label
        self._index = None

    def set_index(self, i):
        self._index = i

    def get_index(self):
        return self.index

    def __str__(self):
        return 'Index: {} Initial Value: {} Label: {}'.format(self._index,
                                                              self.initial,
                                                              self.label)


def create_parameter_list(parameter):
    if not isinstance(parameter, list):
        raise TypeError
    parameterlist = []
    for p in parameter:
        if isinstance(p, (float, int)):
            parameterlist.append(Parameter(p))
        elif isinstance(p, list):
            if not isinstance(p[1], (float, int)):
                raise TypeError
            parameterlist.append(Parameter(p[1], label=p[0]))
        else:
            raise TypeError
    return parameterlist
