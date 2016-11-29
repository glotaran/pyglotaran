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
def calculateC(double[:, :] C, double[:] k, double[:] T):
    I = T.shape[0]
    J = k.shape[0]
    cdef int i, j
    cdef double t_i, k_j
    with nogil, parallel(num_threads=4):
        for i in prange(I, schedule=static):
            for j in range(J):
                t_i = T[i]
                k_j = k[j]
                C[i, j] = exp(k_j * t_i)
