import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport exp
from numpy.math cimport NAN

from cython.parallel import prange, parallel

def __init__():
    np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def calculateC(double[:, :] C, double[:] k, double[:] T):
    I = T.shape[0]
    J = k.shape[0]
    cdef int i, j
    cdef double t_i, k_j
    #  with nogil, parallel(num_threads=num_threads):
    #      for i in prange(I, schedule=static):
    for i in range(I):
        for j in range(J):
            t_i = T[i]
            k_j = k[j]
            C[i, j] = exp(k_j * t_i)
