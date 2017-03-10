
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
def calculateCMultiGaussian(double[:, :, :] C, double[:] rates, double[:] T,
                            double[:, :] centers, double[:, :] widths, double[:] scale):
    nr_gaussian = centers.shape[1]
    if nr_gaussian > 1:
        tmp = np.empty(C.shape, dtype=np.float64)
    J = centers.shape[0]
    for i in range(nr_gaussian):
        if i is 0:
            calculateCSingleGaussian(C, rates, T, centers,
                                     widths, scale[i], i)
        else:
            calculateCSingleGaussian(tmp, rates ,T, centers,
                                     widths, scale[i], i)
            C += tmp

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateCSingleGaussian(double[:, :, :] C, double[:] rates, double[:] times,
                             double[:, :] mu, double[:, :] delta, double scale, int nr_gaussian):
    I = times.shape[0]
    J = rates.shape[0]
    K = mu.shape[0]
    cdef int i, j, k
    cdef double t_i, k_j, mu_k, delta_k, thresh, alpha, beta#term_1, term_2
    #  cdef double delta_tilde = delta / (2 * sqrt(2 * log(2)))
    with nogil, parallel(num_threads=6):
        for i in prange(I, schedule=dynamic):
    #  for i in range(I):
            for j in range(J):
                for k in range(K):
                    t_i = times[i]
                    k_j = rates[j]
                    mu_k = mu[k, nr_gaussian]
                    delta_k = delta[k, nr_gaussian]

                    if k_j == 0:
                        C[k, i, j] = 0
                        continue

                    alpha = -k_j * delta_k / sqrt(2)
                    beta = (t_i - mu_k) / (delta_k * sqrt(2))
                    thresh = beta - alpha
                    if thresh < -1 :
                        C[k, i, j] = scale * .5 * erfce(-thresh) * exp(-beta * beta)
                    else:
                        C[k, i, j] = scale * .5 * (1 + erf(thresh)) * exp(alpha * (alpha - 2 * beta))
