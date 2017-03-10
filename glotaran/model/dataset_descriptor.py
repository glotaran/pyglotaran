from .dataset_scaling import DatasetScaling
from .megacomplex_scaling import MegacomplexScaling


class DatasetDescriptor(object):
    """
    Class representing a dataset for fitting.
    """

    def __init__(self, label, initial_concentration, megacomplexes,
                 megacomplex_scaling, dataset_scaling):
        self.label = label
        self._dataset_scaling = None
        self._initial_concentration = None
        if initial_concentration is not None:
            self.initial_concentration = initial_concentration
        self.megacomplexes = megacomplexes
        self.megacomplex_scaling = megacomplex_scaling
        if dataset_scaling is not None:
            self.dataset_scaling = dataset_scaling
        self._data = None

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
    def dataset_scaling(self):
        return self._dataset_scaling

    @dataset_scaling.setter
    def dataset_scaling(self, scaling):
        if not isinstance(scaling, DatasetScaling):
            raise TypeError
        self._dataset_scaling = scaling

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
        if not isinstance(scaling, list):
            scaling = [scaling]
        if any(not isinstance(s, MegacomplexScaling) for s in scaling):
            raise TypeError
        self._megacomplex_scaling = scaling

    def __str__(self):
        s = "Dataset '{}'\n\n".format(self.label)

        s += "\tDataset Scaling: {}\n".format(self.dataset_scaling)

        s += "\tInitial Concentration: {}\n"\
            .format(self.initial_concentration)

        s += "\tMegacomplexes: {}\n".format(self.megacomplexes)

        if len(self.megacomplex_scaling) is not 0:
            s += "\tScalings:\n"
            for sc in self.megacomplex_scaling:
                s += "\t\t- {}\n".format(sc)

        return s
