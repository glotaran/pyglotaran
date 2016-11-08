import numpy as np
from .minimizer import VariableProjectionMinimizer


class SeperableModel(object):

    def c_matrix(self, parameter, *args, **kwargs):
        raise NotImplementedError

    def e_matrix(self):
        raise NotImplementedError("'self.e_matrix' not defined in model.")

    def eval(self, parameter, *args, **kwargs):
        e = self.e_matrix()
        c = self.c_matrix(parameter, *args, **kwargs)
        return np.dot(c, np.transpose(e))

    def fit(self, initial_parameter, *args, **kwargs):
        minimizer = VariableProjectionMinimizer(self, initial_parameter, *args,
                                                **kwargs)
        return minimizer.minimize(method='least_squares', ftol=1e-10, gtol=1e-10, verbose=2)
