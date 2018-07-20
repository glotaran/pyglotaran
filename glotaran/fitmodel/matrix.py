import numpy as np
from abc import ABC, abstractmethod


class Matrix(ABC):
    def __init__(self, x, dataset, model):
        self.x = x
        self.dataset = dataset
        #  self.buffer = buffer

    def calculate_standalone(self, parameter):
        matrix = np.zeros((self.shape), np.float64)
        self.calculate(matrix, self.compartment_order, parameter)
        return matrix

    @property
    def compartment_order(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @abstractmethod
    def calculate(self, matrix, compartment_order, parameter):
        raise NotImplementedError
