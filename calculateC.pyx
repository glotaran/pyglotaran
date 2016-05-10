# cython: linetrace=True

import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport exp, erf, pow, log, sqrt
from numpy.math cimport NAN

from cython.parallel import prange, parallel

def __init__():
    np.import_array()


@cython.boundscheck(False)
cdef disp(par_0, double[:] par_disp, double l, double l_c):
    cdef double disp_s = par_0
    cdef I = par_disp.shape[0]
    cdef double disp_i
    cdef int i
    for i in range(I):
        disp_i = par_disp[i]
        disp_s += disp_i * pow(((l - l_c) / 100), i)
    return disp_s

@cython.boundscheck(False)
cdef fillC(double[:, :] C, double[:] T, double[:] k, bint irf, double mu, double delta, double delta_tilde):
    I = T.shape[0]
    J = k.shape[0]
    cdef int i, j
    cdef double t_i, k_j, term_1, term_2
    with nogil, parallel(num_threads=4):
        for i in prange(I, schedule=static):
            for j in range(J):
                t_i = T[i]
                k_j = k[j]
                C[i, j] = exp(-k_j * t_i)
                if irf:
                    term_1 = exp(k_j * (k_j * delta_tilde * delta_tilde / 2))
                    term_2 = 1 + erf((t_i - (mu + k_j * delta_tilde * delta_tilde)) / (sqrt(2) * delta_tilde))
                    C[i, j] *= .5 * term_1 * term_2    

@cython.boundscheck(False)
cpdef calculateC(double[:] k, double[:] T, bint irf, double mu_0,\
                 double delta_0, double[:] mu_disp,\
                 double[:] delta_disp, double l,\
                 double l_c_mu, double l_c_delta):
    C = np.empty((T.shape[0], k.shape[0]), dtype=np.float64)
    cdef double mu, delta, delta_tilde
    if irf:
        mu = disp(mu_0, mu_disp, l, l_c_mu)
        delta = disp(delta_0, delta_disp, l, l_c_delta)
        delta_tilde = delta / (2 * sqrt(2 * log(2)))
    fillC(C, T, k, irf, mu, delta, delta_tilde)
    return C
    