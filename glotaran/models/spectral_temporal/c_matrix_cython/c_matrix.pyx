import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport exp
from numpy.math cimport NAN

def __init__():
    np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def calculateC(double[:, :] C, idxs, double[:] k, double[:] T, double scale):
    nr_times = T.shape[0]
    nr_comps = k.shape[0]
    cdef int n_c, n_t, n_k
    cdef double t_n, k_n
    for n_k in range(nr_comps):
        n_c = idxs[n_k]
        k_n = k[n_k]
        for n_t in range(nr_times):
            t_n = T[n_t]
            C[n_t, n_c] += scale * exp(-k_n * t_n)
