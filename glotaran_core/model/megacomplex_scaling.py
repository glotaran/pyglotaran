class MegacomplexScaling(object):
    def __init__(self, megacomplexes, compartments, parameter):
        self.megacomplexes = megacomplexes
        self.compartments = compartments
        self.parameter = parameter

    @property
    def megacomplexes(self):
        return self._megacomplexes

    @megacomplexes.setter
    def megacomplexes(self, megacomplexes):
        if not isinstance(megacomplexes, list):
            megacomplexes = [megacomplexes]
        if any(not isinstance(val, str) for val in megacomplexes):
            raise ValueError("Megacomplexes labels must be string.")
        self._megacomplexes = megacomplexes

    @property
    def compartments(self):
        return self._compartments

    @compartments.setter
    def compartments(self, compartments):
        if not isinstance(compartments, list):
            compartments = [compartments]
        if any(not isinstance(val, int) for val in compartments):
            raise ValueError("Compartment indices must be integer.")
        self._compartments = compartments

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, parameter):
        if not isinstance(parameter, int):
            raise TypeError("Parameter index must be numerical")
        self._parameter = parameter

    def __str__(self):
        return \
                "Megacomplexes: {}, Compartements: {}, Parameter:{}".format(
                    self.megacomplexes, self.compartments, self.parameter)
