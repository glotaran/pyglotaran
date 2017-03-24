import numpy as np


class Dataset(object):

    def __init__(self, label):
        self.label = label
        self._axis = {}

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    def get_axis(self, label):
        return self._axis[label]

    def set_axis(self, label, value):
        self._axis[label] = value

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
