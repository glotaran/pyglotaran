# cython: language_level=3

import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport exp, sqrt, erf

from scalTOMS680 import scalwofz

def __init__():
    np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_doas_matrix_faddeva(double[:, :] matrix,
                                     double[:] frequencies,
                                     double[:] rates,
                                     double[:] times,
                                     double center,
                                     double width,
                                     double scale):
    nr_times = times.shape[0]
    nr_freq = frequencies.shape[0]
    sqwidth = width * sqrt(2)

    cdef int n_t, n_f, n_c
    cdef double t_n, f_n, r_n, f_scale
    cdef complex z

    n_c = 0
    for n_f in range(nr_freq):
        f_n = frequencies[n_f]
        r_n = rates[n_f]

        for n_t in range(nr_times):

            t_n = times[n_t] - center

            z = (t_n - width**2 * (r_n + 1j * f_n))
            if r_n < 0:
                z /= (-1j * sqwidth)
            else:
                z /= (1j * sqwidth)

            f_scale = (t_n / sqwidth)**2

            re, im, overflow, scaled = scalwofz(z.real, z.imag, f_scale)

            if scaled and not overflow:
                matrix[n_t, n_c] += re * scale
                matrix[n_t, n_c + 1] += im * scale
            if not overflow:
                scale_f = scale * exp(-1 * f_scale)
                matrix[n_t, n_c] += scale_f * re
                matrix[n_t, n_c + 1] += scale_f * im

        n_c += 2
