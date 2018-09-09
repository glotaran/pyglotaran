import sys

import cython
cimport cython

import numpy as np
cimport numpy as np

# Not sure why but the scipy.special.cython_special.erfcx function doesnt' sseem to work.
#from scipy.special import erf, erfc, erfcx
#from scipy.special.cython_special cimport erf, erfc, erfcx #try in 0.19.0
#cimport scipy.special.cython_special as csc

from libc.math cimport exp,  pow, log, sqrt, erf
from numpy.math cimport NAN

cdef extern from "erfce.c":
    double erfce(double x)

def __init__():
    np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_kinetic_matrix_gaussian_irf(double[:, :] matrix,
                                     double[:] rates,
                                     double[:] times,
                                     double center,
                                     double width,
                                     double scale,
                                     int backsweep,
                                     double backsweep_period):
    nr_times = times.shape[0]
    nr_rates = rates.shape[0]
    cdef int n_t, n_r
    cdef double t_n, r_n, thresh, alpha, beta#term_1, term_2
    for n_r in range(nr_rates):
        r_n = -rates[n_r]
        alpha = (r_n * width) / sqrt(2)
        for n_t in range(nr_times):
            t_n = times[n_t]
            beta = (t_n - center) / (width * sqrt(2))
            thresh = beta - alpha
            if thresh < -1 :
                matrix[n_r, n_t] += scale * .5 * erfce(-thresh) * exp(-beta * beta)
            else:
                matrix[n_r, n_t] += scale * .5 * (1 + erf(thresh)) * exp(alpha * (alpha - 2 * beta))
            if backsweep != 0:
                x1 = exp(-r_n * (t_n - center + backsweep_period))
                x2 = exp(-r_n * ((backsweep_period / 2) - (t_n - center)))
                x3 = exp(-r_n * backsweep_period)
                matrix[n_r, n_t] += (x1 + x2) / (1 - x3)
