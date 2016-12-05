from .minimizer import VariableProjectionMinimizer
import numpy as np


class SeperableModelResult(VariableProjectionMinimizer):

    def __init__(self, model, initial_parameter, *args, **kwargs):
        super(SeperableModelResult, self).__init__(model,
                                                   initial_parameter,
                                                   *args,
                                                   **kwargs)

    def fit(self, initial_parameter, *args, **kwargs):
        verbose = kwargs['verbose'] if 'verbose' in kwargs else 2
        _res = self.minimize(method='least_squares',
                             ftol=1e-10,
                             gtol=1e-10,
                             verbose=verbose)

        self.best_fit_parameter = _res.params

    def e_matrix(self, *args, **kwargs):
        return self.model.retrieve_e_matrix(self.best_fit_parameter,
                                            *args, **kwargs)

    def eval(self, *args, **kwargs):
        e = self.e_matrix(*args, **kwargs)
        c = self.model.c_matrix(self.best_fit_parameter, *args, **kwargs)
        res = np.dot(c, e)
        return res
