
import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport exp, erf, erfc, pow, log, sqrt
from numpy.math cimport NAN

from cython.parallel import prange, parallel

def __init__():
    np.import_array()


cpdef double erfce(double x) nogil:
    return exp(x*x) * erfc(x)


@cython.boundscheck(False)
@cython.wraparound(False)
def calculateCMultiGaussian(double[:, :] C, double[:] k, double[:] T,
                            double[:] center, double[:] width, double[:] scale):
    nr_gaussian = center.shape[0]
    if nr_gaussian > 1:
        tmp = np.empty(C.shape, dtype=np.float64)
    for i in range(nr_gaussian):
        if i is 0:
            calculateCSingleGaussian(C, k ,T, center[i], width[i], scale[i])
        else:
            calculateCSingleGaussian(tmp, k ,T, center[i], width[i], scale[i])
            C += tmp

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateCSingleGaussian(double[:, :] C, double[:] k, double[:] T, double mu, double
                  delta, double scale):
    I = T.shape[0]
    J = k.shape[0]
    cdef int i, j
    cdef double t_i, k_j, thresh, alpha, beta#term_1, term_2
    cdef double delta_tilde = delta / (2 * sqrt(2 * log(2)))
    with nogil, parallel(num_threads=4):
        for i in prange(I, schedule=static):
            for j in range(J):
                t_i = T[i]
                k_j = k[j]

                if k_j == 0:
                    C[i, j] = 0
                    continue

                alpha = -k_j * delta / sqrt(2)
                beta = (t_i - mu) / (delta * sqrt(2))
                thresh = beta - alpha
                if thresh < -1 :
                    C[i, j] = scale * .5 * erfce(-thresh) * exp(-beta * beta)
                else:
                    C[i, j] = scale * .5 * (1 + erf(thresh)) * exp(alpha * (alpha - 2 * beta))
