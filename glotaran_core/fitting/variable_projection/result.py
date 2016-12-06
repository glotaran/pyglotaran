from lmfit import Minimizer
import numpy as np

from .qr_decomposition import qr_residual


class SeperableModelResult(Minimizer):

    def __init__(self, model, initial_parameter, *args, **kwargs):
        self.model = model
        super(SeperableModelResult, self).__init__(self._residual,
                                                   initial_parameter,
                                                   fcn_args=args,
                                                   fcn_kws=kwargs)

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

    def c_matrix(self, *args, **kwargs):
        return self.model.c_matrix(self.best_fit_parameter, *args, **kwargs)

    def eval(self, *args, **kwargs):
        e = self.e_matrix(*args, **kwargs)
        c = self.c_matrix(*args, **kwargs)
        res = np.empty((c.shape[1], e.shape[1]))
        for i in range(e.shape[1]):
            res[:, i] = np.dot(c[i, :, :], e[:, i])
        return res

    def _residual(self, parameter, *args, **kwargs):

        data = self.model.data(**kwargs)[0]
        c_matrix = self.model.c_matrix(parameter.valuesdict(), *args, **kwargs)
        res = np.empty(data.shape, dtype=np.float64)

        for i in range(data.shape[1]):

            b = data[:, i]
            c = c_matrix[i, :, :]
            qr = qr_residual(c, b)
            res[:, i] = qr

        return res.flatten()
