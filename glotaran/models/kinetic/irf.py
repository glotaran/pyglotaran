class Irf(object):
    """
    Represents an abstract IRF.
    """
    _label = None

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

    @property
    def time_axis(self):
        return self.get_axis("time")

    @time_axis.setter
    def time_axis(self, value):
        self.set_axis("time", value)

    @property
    def spectral_axis(self):
        return self.get_axis("spectral")

    @spectral_axis.setter
    def spectral_axis(self, value):
        self.set_axis("spectral", value)

    def get_estimated_axis(self):
        return self.time_axis

    def type_string(self):
        raise NotImplementedError

    def __str__(self):
        return "Label: {} Type: {}".format(self.label, self.type_string())
