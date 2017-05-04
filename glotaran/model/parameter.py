from math import isnan

from lmfit import Parameter as LmParameter


class Parameter(LmParameter):
    """
    A parameter has an initial value and an optional label.
    """
    def __init__(self):
        self._index = -1
        super(Parameter, self).__init__()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, i):
        self._index = i

    @property
    def label(self):
        return self.name

    @label.setter
    def label(self, label):
        self.name = label

    @LmParameter.value.setter
    def value(self, val):

        if not isinstance(val, (int, float)):
                try:
                    val = float(val)
                except:
                    raise Exception("Parameter Error: value must be numeric:"
                                    "{} Type: {}".format(val, type(val)))

        if isinstance(val, int):
            val = float(val)

        if isnan(val):
            self.vary = False

        LmParameter.value.fset(self, val)

    def _str__(self):
        return 'Label: {}\tInitial Value: {}\t Fix: {}\tMin: {} Max: {}'\
                .format(self.label,
                        self.value,
                        not self.vary,
                        self.min,
                        self.max)
