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
def calc_kinetic_matrix_no_irf(double[:, :] matrix,
                               double[:] rates,
                               double[:] times,
                               double scale):
    nr_times = times.shape[0]
    nr_rates = rates.shape[0]
    cdef int n_t, n_r
    cdef double t_n, r_n
    for n_r in range(nr_rates):
        r_n = rates[n_r]
        for n_t in range(nr_times):
            t_n = times[n_t]
            matrix[n_r, n_t] += scale * exp(r_n * t_n)
