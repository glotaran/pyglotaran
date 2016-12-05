
import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport exp, cos, sin
from numpy.math cimport NAN

from cython.parallel import prange, parallel

def __init__():
    np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateC(double[:, :] C, double[:] k, double[:] o, double[:] d, double[:] T):
    I = T.shape[0]
    J = k.shape[0]
    K = o.shape[0]
    cdef int i, j
    cdef double t_i, k_j
    with nogil, parallel(num_threads=4):
        for i in prange(I, schedule=static):
            t_i = T[i]
            for j in range(J):
                k_j = k[j]
                C[i, j] = exp(k_j * t_i)
    with nogil, parallel(num_threads=4):
        for i in prange(I, schedule=static):
            t_i = T[i]
            for j in range(K):
                o_j = o[j]
                d_j = d[j]
                C[i, J+j] = cos(o_j * t_i) * exp(-d_j * t_i)
                C[i, J+K+j] = sin(o_j * t_i) * exp(-d_j * t_i)
