from __future__ import print_function
import sys

import cython
cimport cython

import numpy as np
cimport numpy as np

# Not sure why but the scipy.special.cython_special.erfcx function doesnt' sseem to work.
#from scipy.special import erf, erfc, erfcx
from scipy.special.cython_special cimport erf, erfc, erfcx
#cimport scipy.special.cython_special as csc

from libc.math cimport exp,  pow, log, sqrt
from numpy.math cimport NAN

from cython.parallel import prange, parallel

def __init__():
    np.import_array()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateCMultiGaussian(double[:, :, :] C, double[:] k, double[:] T,
                            double[:, :] centers, double[:, :] widths, double[:] scale):
    nr_gaussian = centers.shape[1]
    if nr_gaussian > 1:
        tmp = np.empty(C.shape, dtype=np.float64)
    J = centers.shape[0]
    for i in range(nr_gaussian):
        if i is 0:
            for j in range(J):
                calculateCSingleGaussian(C[j, :, :], k ,T, centers[j, i],
                                         widths[j, i], scale[i])
        else:
            for j in range(J):
                calculateCSingleGaussian(tmp[j, :, :], k ,T, centers[j, i],
                                         widths[j, i], scale[i])
            C += tmp

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateCSingleGaussian(double[:, :] C, double[:] k, double[:] T, double mu, double
                  delta, double scale):
    I = T.shape[0]
    J = k.shape[0]
    cdef int i, j
    cdef double t_i, k_j, thresh, alpha, beta#term_1, term_2
    #  cdef double delta_tilde = delta / (2 * sqrt(2 * log(2)))
    #  with nogil, parallel(num_threads=num_threads):
    #      for i in prange(I, schedule=static):
    for i in range(I):
        for j in range(J):
            t_i = T[i]
            k_j = k[j]

            if k_j == 0:
                C[i, j] = 0
                continue

            alpha = -k_j * delta / sqrt(2)
            beta = (t_i - mu) / (delta * sqrt(2))
            thresh = beta - alpha
            eprint("thresh is: ", thresh)
            if thresh < -1 :
                C[i, j] = scale * .5 * erfcx(-thresh) * exp(-beta * beta)
            else:
                C[i, j] = scale * .5 * (1 + erf(thresh)) * exp(alpha * (alpha - 2 * beta))



