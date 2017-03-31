from .dataset import Dataset


class DatasetDescriptor(object):
    """
    Class representing a dataset for fitting.
    """

    def __init__(self, label, initial_concentration, megacomplexes,
                 megacomplex_scaling, dataset_scaling, compartment_scaling):
        self.label = label
        self.initial_concentration = initial_concentration
        self.megacomplexes = megacomplexes
        self.compartment_scaling = compartment_scaling
        self.megacomplex_scaling = megacomplex_scaling
        self.scaling = dataset_scaling
        self.data = None

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, Dataset) and data is not None:
            raise TypeError
        self._data = data

    @property
    def initial_concentration(self):
        '''Returns the label of the initial concentration to be used to fit the
        dataset.'''
        return self._initial_concentration

    @initial_concentration.setter
    def initial_concentration(self, value):
        '''Sets the label of the initial concentration to be used to fit the
        dataset.'''
        self._initial_concentration = value

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        if not isinstance(scaling, int) and scaling is not None:
            raise TypeError("Parameter index must be numerical")
        self._scaling = scaling

    @property
    def compartment_scaling(self):
        return self._compartment_scaling

    @compartment_scaling.setter
    def compartment_scaling(self, scaling):
        if not isinstance(scaling, dict):
            raise TypeError
        self._compartment_scaling = scaling

    @property
    def megacomplexes(self):
        return self._megacomplexes

    @megacomplexes.setter
    def megacomplexes(self, megacomplex):
        if not isinstance(megacomplex, list):
            megacomplex = [megacomplex]
        if any(not isinstance(m, str) for m in megacomplex):
            raise TypeError("Megacomplex labels must be string.")
        self._megacomplexes = megacomplex

    @property
    def megacomplex_scaling(self):
        return self._megacomplex_scaling

    @megacomplex_scaling.setter
    def megacomplex_scaling(self, scaling):
        if not isinstance(scaling, dict):
            raise TypeError("Megacomplex Scaling must by dict, got"
                            "{}".format(type(scaling)))
        self._megacomplex_scaling = scaling

    def __str__(self):
        s = "Dataset '{}'\n\n".format(self.label)

        s += "\tDataset Scaling: {}\n".format(self.scaling)

        s += "\tInitial Concentration: {}\n"\
            .format(self.initial_concentration)

        s += "\tMegacomplexes: {}\n".format(self.megacomplexes)

        s += "\tMega scalings:\n"
        for cmplx, scale in self._megacomplex_scaling.items():
            s += "\t\t- {}:{}\n".format(cmplx, scale)

        return s
