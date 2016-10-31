class Irf(object):
    """
    Represents an abstract IRF.
    """
    def __init__(self, label):
        self.label = label

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        if not isinstance(value, str):
            raise TypeError("Labels must be strings.")
        self._label = value

    def get_irf_function(self):
        raise NotImplementedError

    def type_string(self):
        raise NotImplementedError

    def __str__(self):
        return "Label: {} Type: {}".format(self.label(), self.type_string())
