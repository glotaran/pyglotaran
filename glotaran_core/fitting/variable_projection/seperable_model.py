import numpy as np
from .result import SeperableModelResult
from .qr_decomposition import qr_coefficents


class SeperableModel(object):

    def c_matrix(self, parameter, *args, **kwargs):
        raise NotImplementedError

    def e_matrix(self, **kwarg):
        raise NotImplementedError("'self.e_matrix' not defined in model.")

    def data(self, **kwargs):
        raise NotImplementedError

    def eval(self, parameter, *args, **kwargs):
        e = self.e_matrix(**kwargs)
        c = self.c_matrix(parameter, *args, **kwargs)
        noise = kwargs["noise"] if "noise" in kwargs else False
        res = np.empty((c.shape[1], e.shape[0]))
        print(res.shape)
        for i in range(e.shape[0]):
            res[:, i] = np.dot(c[i, :, :], np.transpose(e[i, :]))
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
        data = self.data(**kwargs)[0]
        c_matrix = self.c_matrix(parameter, *args, **kwargs)
        e_matrix = np.empty((c_matrix.shape[2], data.shape[1]),
                            dtype=np.float64)

        for i in range(data.shape[1]):
            b = data[:, i]
            qr = qr_coefficents(c_matrix[i, :, :], b)
            e_matrix[:, i] = qr[:c_matrix.shape[2]]
        return e_matrix
