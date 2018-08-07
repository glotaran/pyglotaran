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
def calculateCSingleGaussian(double[:, :] C, idxs, double[:] k, double[:] x, double mu, double
                  delta, double scale, int backsweep, double backsweep_period):
    nr_x = x.shape[0]
    nr_comps = k.shape[0]
    cdef int n_c, n_x, n_k
    cdef double x_n, k_n, thresh, alpha, beta#term_1, term_2
    for n_k in range(nr_comps):
        k_n = k[n_k]
        if k_n == 0:
            continue
        n_c = idxs[n_k]
        alpha = (k_n * delta) / sqrt(2)
        for n_x in range(nr_x):
            x_n = x[n_x]
            beta = (x_n - mu) / (delta * sqrt(2))
            thresh = beta - alpha
            if thresh < -1 :
                C[n_x, n_c] += scale * .5 * erfce(-thresh) * exp(-beta * beta)
            else:
                C[n_x, n_c] += scale * .5 * (1 + erf(thresh)) * exp(alpha * (alpha - 2 * beta))
            if backsweep != 0:
                x1 = exp(-k_n * (x_n - mu + backsweep_period))
                x2 = exp(-k_n * ((backsweep_period / 2) - (x_n - mu)))
                x3 = exp(-k_n * backsweep_period)
                C[n_x, n_c] += (x1 + x2) / (1 - x3)