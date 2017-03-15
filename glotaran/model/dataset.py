import numpy as np


class Dataset(object):

    def __init__(self, label, independent_axies):
        self.label = label
        self.independent_axies = independent_axies

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def independent_axies(self):
        return self._indpendent_axies

    @independent_axies.setter
    def independent_axies(self, value):
        self._indpendent_axies = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a nd array")
        if len(data.shape) is not 2:
            raise ValueError("Dataset must be 2-dimensional")
        self._data = data


class IndependentAxies(object):

    def __init__(self):
        self._axies = []

    def add(self, axies):
        self._axies.append(axies)

    def get(self, nr):
        return self._axies[nr]
