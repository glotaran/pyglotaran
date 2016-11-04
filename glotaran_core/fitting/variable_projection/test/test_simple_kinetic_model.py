from unittest import TestCase, main
from glotaran_core.fitting.variable_projection import (
    SeperableModel,
    VariableProjectionMinimizer,
)
from lmfit import Parameters
import numpy as np

import scipy.optimize
import scipy.linalg.lapack as lapack


def qr(a, c):
    qr, tau, _, _ = lapack.dgeqrf(a, overwrite_a=1)
    c, _, _ = lapack.dormqr("L", "T", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    for i in range(a.shape[1]):
        c[i] = 0
    c, _, _ = lapack.dormqr("L", "N", qr, tau, c, max(1, a.shape[1]), overwrite_c=1)
    return c



class TestSimpleKinetic(TestCase):

    def test_one_compartment_decay(self):

        class OneComparmentDecay(SeperableModel):

            def parameter(self):
                params = Parameters()
                params.add("p1", 4e-3)
                return params

            def c_matrix(self, parameter, *times, **kwargs):
                kinpar = np.asarray([parameter["p1"]])
                return np.exp(np.outer(np.asarray(times), -kinpar))

        # E Matrix => channels X compartments
        E = np.empty((1, 1), dtype=np.float64, order="F")

        E[0, 0] = 3.4

        model = OneComparmentDecay()
        times = np.asarray(np.arange(0, 1500, 1.5))

        real_params = Parameters()
        real_params.add("p1", 4.679e-3)

        data = np.dot(model.c_matrix(real_params, times),
                      np.transpose(E))

        print("ddddd")
        print(data.shape)
        print("ddddd")

        def solve(k, PSI, times):
            res = np.empty(PSI.shape, dtype=np.float64)
            C = model.c_matrix({'p1':k[0]}, times)
            for i in range(PSI.shape[1]):
                b = PSI[:,i]
                res[:,i] = qr(C, b)
            return res.flatten()


        minimizer = VariableProjectionMinimizer(model,
                                                *times, **{"data": data})

        result = minimizer.minimize(method='least_squares')
        print(result)
        print(result.params)
        print(result.success)
        print(result.message)

        # start_kinpar = np.asarray([.001])
        # res = scipy.optimize.least_squares(solve, start_kinpar, args=(data, times), verbose=2, method='lm')
        print("ggggggggggggggggg")
        # print(res)
        self.assertTrue(False)
#
#      class TestMultiKinetics
#
#          def test_one_compartment_decay(self):
#
#              class OneComparmentDecay(SeperableModel):
#
#                  def parameter(self):
#                      params = Parameters()
#                      params.add("p1", 4e-3)
#                      return params
#
#                  def c_matrix(self, parameter, *times, **kwargs):
#                      kinpar = np.asarray([parameter["p1"]])
#                      return np.exp(np.outer(np.asarray(times), -kinpar))
#
#
#  if __name__ == '__main__':
#      main()
