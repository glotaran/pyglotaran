class InitialConcentration(object):
    """
    An InitialConcentration constration has label and parameters.
    """
    def __init__(self, label, parameter):
        self.label = label
        self.parameter = parameter

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        if not isinstance(value, list):
            value = [value]
        self._parameter = value

    def __str__(self):
        return f"* __{self.label}__: {self.parameter}"
