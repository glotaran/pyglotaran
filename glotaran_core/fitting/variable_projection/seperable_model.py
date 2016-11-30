import numpy as np
from .minimizer import VariableProjectionMinimizer


class SeperableModel(object):

    def c_matrix(self, parameter, *args, **kwargs):
        raise NotImplementedError

    def e_matrix(self, **kwarg):
        raise NotImplementedError("'self.e_matrix' not defined in model.")

    def eval(self, parameter, *args, **kwargs):
        e = self.e_matrix(**kwargs)
        c = self.c_matrix(parameter, *args, **kwargs)
        noise = kwargs["noise"] if "noise" in kwargs else False
        res = np.dot(c, np.transpose(e))
        if noise:
            std_dev = kwargs["noise_std_dev"] if "noise_std_dev" in kwargs \
                else 1.0
            res = np.random.normal(res, std_dev)
        return res

    def fit(self, initial_parameter, *args, **kwargs):
        minimizer = VariableProjectionMinimizer(self, initial_parameter, *args,
                                                **kwargs)
        verbose = kwargs['verbose'] if 'verbose' in kwargs else 2
        return minimizer.minimize(method='least_squares',
                                  ftol=1e-10,
                                  gtol=1e-10,
                                  verbose=verbose)
