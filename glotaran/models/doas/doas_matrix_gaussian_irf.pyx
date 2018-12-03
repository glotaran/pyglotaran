# cython: language_level=3
import cython
cimport cython

import numpy as np
cimport numpy as np

cdef extern from "complex.h":
    double complex exp(double complex)
    double complex erf(double complex)


def __init__():
    np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_doas_matrix_gaussian_irf(double[:, :] matrix,
                                 double[:] rates,
                                 double[:] frequencies,
                                 double[:] times,
                                 double center,
                                 double width,
                                 double scale):
    nr_times = times.shape[0]
    nr_rates = rates.shape[0]
    width2 = width * width
    sqrt2 = np.sqrt(2)
    cdef int n_t, n_r
    cdef double t_n, f_n, r_n,
    cdef double complex k, a, b, osc
    cdef int idx = 0
    for n_r in range(nr_rates):
        r_n = rates[n_r]
        f_n = frequencies[n_r]
        k = r_n + 1j * f_n
        for n_t in range(nr_times):
            t_n = times[n_t]
            a = exp((-1 * t_n + 0.5 * width2 *k) *k)
            b = 1 + erf((t_n - width2 * k)/ (sqrt2 * width))
            osc = a * b
            matrix[idx, n_t] = np.imag(osc)
            matrix[idx + 1, n_t] = np.real(osc)
        idx += 2
