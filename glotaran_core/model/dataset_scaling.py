class DatasetScaling(object):
    def __init__(self, parameter):
        self.parameter = parameter

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, parameter):
        if not isinstance(parameter, int):
            raise TypeError("Parameter index must be numerical")
        self._parameter = parameter
