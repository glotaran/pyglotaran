
import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport exp, erf, erfc, pow, log, sqrt
from numpy.math cimport NAN

from cython.parallel import prange, parallel

def __init__():
    np.import_array()


#@cython.boundscheck(False)
#cdef disp(par_0, double[:] par_disp, double l, double l_c):
#    cdef double disp_s = par_0
#    cdef I = par_disp.shape[0]
#    cdef double disp_i
#    cdef int i
#    for i in range(I):
#        disp_i = par_disp[i]
#        disp_s += disp_i * pow(((l - l_c) / 100), i)
#    return disp_s
#
#@cython.boundscheck(False)
#cdef fillC(double[:, :] C, double[:] T, double[:] k, bint irf, double mu, double delta, double delta_tilde):
#    I = T.shape[0]
#    J = k.shape[0]
#    cdef int i, j
#    cdef double t_i, k_j, term_1, term_2
#    with nogil, parallel(num_threads=4):
#        for i in prange(I, schedule=static):
#            for j in range(J):
#                t_i = T[i]
#                k_j = k[j]
#                C[i, j] = exp(-k_j * t_i)
#                if irf:
#                    term_1 = exp(k_j * (k_j * delta_tilde * delta_tilde / 2))
#                    term_2 = 1 + erf((t_i - (mu + k_j * delta_tilde * delta_tilde)) / (sqrt(2) * delta_tilde))
#                    C[i, j] *= .5 * term_1 * term_2    
#
#@cython.boundscheck(False)
#cpdef calculateC(double[:] k, double[:] T, bint irf, double mu_0,\
#                 double delta_0, double[:] mu_disp,\
#                 double[:] delta_disp, double l,\
#                 double l_c_mu, double l_c_delta):
#    C = np.empty((T.shape[0], k.shape[0]), dtype=np.float64)
#    cdef double mu, delta, delta_tilde
#    if irf:
#        mu = disp(mu_0, mu_disp, l, l_c_mu)
#        delta = disp(delta_0, delta_disp, l, l_c_delta)
#        delta_tilde = delta / (2 * sqrt(2 * log(2)))
#    fillC(C, T, k, irf, mu, delta, delta_tilde)
#    return C

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
                C[i, j] = exp(-k_j * t_i)
    #return C

    
@cython.boundscheck(False)
@cython.wraparound(False)
def calculateCirf(double[:, :] C, double[:] k, double[:] T, double mu, double delta):
    I = T.shape[0]
    J = k.shape[0]
    cdef int i, j
    cdef double t_i, k_j, thresh, alpha, beta#term_1, term_2
    cdef double delta_tilde = delta / (2 * sqrt(2 * log(2)))
    with nogil, parallel(num_threads=4):
        for i in prange(I, schedule=static):
            for j in range(J):
                t_i = T[i]
                k_j = k[j]
                
                if k_j == 0:
                    C[i, j] = 0
                    continue
                
#                term_1 = exp(k_j * (mu + k_j * delta_tilde * delta_tilde / 2))
#                term_2 = 1 + erf((t_i - (mu + k_j * delta_tilde * delta_tilde)) / (sqrt(2) * delta_tilde))
#                C[i, j] = .5 * exp(-k_j * t_i) * term_1 * term_2
                alpha = k_j * delta / sqrt(2)
                beta = (t_i - mu) / (delta * sqrt(2))
                thresh = beta - alpha
                if thresh < -1 :
                    C[i, j] = .5 * erfce(-thresh) * exp(-beta * beta)
                else:
                    C[i, j] = .5 * (1 + erf(thresh)) * exp(alpha * (alpha - 2 * beta))
    #return C


###TODO: Update again!
@cython.boundscheck(False)    
@cython.wraparound(False)
def calculateCirf_multi(double[:, :] C, double[:] k, double[:] T, double[:] mu_disp, double[:] delta_disp):
    I = T.shape[0]
    J = k.shape[0]
    cdef int i, j
    cdef double t_i, k_j, term_1, term_2
    cdef double mu, delta, delta_tilde
    with nogil, parallel(num_threads=4):
        for i in prange(I, schedule=static):
            for j in range(J):
                t_i = T[i]
                k_j = k[j]
                
                if k_j == 0:
                    C[i, j] = 0
                    continue                
                mu = mu_disp[j]
                delta = delta_disp[j]                
                delta_tilde = delta / (2 * sqrt(2 * log(2)))
                term_1 = exp(k_j * (k_j * delta_tilde * delta_tilde / 2))
                term_2 = 1 + erf((t_i - (mu + k_j * delta_tilde * delta_tilde)) / (sqrt(2) * delta_tilde))
                C[i, j] = .5 * exp(-k_j * t_i) * term_1 * term_2
                    
    #return C