import numpy as np
from .result import SeperableModelResult 
from .qr_decomposition import qr_decomposition_coeff


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
        result = SeperableModelResult(self, initial_parameter, *args,
                                      **kwargs)
        result.fit(initial_parameter, *args, **kwargs)
        return result

    def retrieve_e_matrix(self, parameter, *args, **kwargs):
        data = kwargs['data']
        c_matrix = self.c_matrix(parameter, *args, **kwargs)
        res = np.empty(data.shape, dtype=np.float64)

        for i in range(data.shape[1]):

            b = data[:, i]
            qr = qr_decomposition_coeff(c_matrix, b)
            res[:, i] = qr
        return res
