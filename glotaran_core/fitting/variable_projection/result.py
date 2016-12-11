from lmfit import Minimizer
import numpy as np
from .qr_decomposition import qr_residual


class SeperableModelResult(Minimizer):
    _residual_buffer = None

    def __init__(self, model, initial_parameter, *args, **kwargs):
        self.model = model
        super(SeperableModelResult, self).__init__(self._residual,
                                                   initial_parameter,
                                                   fcn_args=args,
                                                   fcn_kws=kwargs)

    def fit(self, initial_parameter, ftol=1e-10, gtol=1e-10, *args, **kwargs):
        verbose = kwargs['verbose'] if 'verbose' in kwargs else 2
        ftol = 1e-10
        gtol = 1e-10
        res = self.minimize(method='least_squares',
                            ftol=ftol,
                            gtol=gtol,
                            verbose=verbose)

        self.best_fit_parameter = res.params

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

    def final_residual(self, *args, **kwargs):
        data = self.model.data(**kwargs)[0]
        reconstructed = self.eval(*args, **kwargs)
        return data-reconstructed

    def final_residual_svd(self, *args, **kwargs):
        residual = self.final_residual(*args, **kwargs)
        lsvd, svals, rsvd = np.linalg.svd(residual)
        return lsvd, svals, rsvd

    # @profile
    def _residual(self, parameter, *args, **kwargs):

        data = self.model.data(**kwargs)[0]
        c_matrix = self.model.c_matrix(parameter.valuesdict(), *args, **kwargs)
        #  res = np.empty(data.shape, dtype=np.float64)
        if self._residual_buffer is None:
            self._residual_buffer = np.empty(data.shape, dtype=np.float64)

        #  print(parameter['p1'])
        #  print(self._residual_buffer.shape)
        #  print('bef')
        #  print(c_matrix.shape)
        #  print('befor copy')
        #  print(self._residual_buffer.flatten()[:12])
        np.copyto(self._residual_buffer, data)
        #  print('after copy')
        #  print(self._residual_buffer.flatten()[:12])
        for i in range(data.shape[1]):

            #  b = self._residual_buffer[:, i]
            #  print('go')
            #  print(i)
            #  print(self._residual_buffer[:, i].shape)
            b = data[:, i]
            #  print('bevor')
            #  print(self._residual_buffer[:, i][:6])
            c = c_matrix[i, :, :]
            #  qr_residual(c, self._residual_buffer[:, i])
            qr = qr_residual(c, b)
            self._residual_buffer[:, i] = qr
            #  print('danach')
            #  print(self._residual_buffer[:, i][:6])
            #  res[:, i] = qr

        #  print('final')
        #  print(self._residual_buffer.flatten()[:12])
        return self._residual_buffer.flatten()
        #  return res.flatten()
